from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _normalize_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


@dataclasses.dataclass
class Handle:
    handle_id: int
    offset: int
    nbytes: int
    reserved_nbytes: int
    align: int
    tag: str


class UnifiedGpuPool:
    def __init__(
        self,
        device: torch.device | str,
        total_budget_bytes: int = 0,
    ) -> None:
        self.device = _normalize_device(device)
        self.total_budget_bytes = max(0, int(total_budget_bytes))
        self._next_handle_id = 1
        self._handles: Dict[int, Handle] = {}
        self._used_bytes = 0
        self._peak_bytes = 0
        self._arena: Optional[torch.Tensor] = None
        self._free_spans: list[tuple[int, int]] = []

    def configure_budget(self, total_budget_bytes: int) -> None:
        total_budget_bytes = max(0, int(total_budget_bytes))
        if self._arena is not None and total_budget_bytes != self.total_budget_bytes:
            if total_budget_bytes < self._used_bytes:
                raise RuntimeError(
                    f"Cannot shrink UnifiedGpuPool below used bytes: used={self._used_bytes} "
                    f"new_budget={total_budget_bytes}"
                )
            raise RuntimeError(
                "UnifiedGpuPool arena is already bootstrapped; dynamic resize is not supported"
            )
        self.total_budget_bytes = total_budget_bytes
        if self._arena is None and total_budget_bytes > 0:
            self._bootstrap_arena()

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def is_bootstrapped(self) -> bool:
        return self._arena is not None

    def _bootstrap_arena(self) -> None:
        if self._arena is not None:
            return
        if self.total_budget_bytes <= 0:
            raise RuntimeError(
                "UnifiedGpuPool budget must be configured before the first allocation"
            )
        self._arena = torch.empty(
            self.total_budget_bytes, dtype=torch.uint8, device=self.device
        )
        self._free_spans = [(0, self.total_budget_bytes)]

    def _find_span(self, nbytes: int, align: int) -> tuple[int, int, int] | None:
        assert self._arena is not None
        arena_base = self._arena.data_ptr()
        best: tuple[int, int, int, int] | None = None
        for idx, (start, span_nbytes) in enumerate(self._free_spans):
            aligned_start = start + ((align - ((arena_base + start) % align)) % align)
            end = aligned_start + nbytes
            if end > start + span_nbytes:
                continue
            waste = span_nbytes - (end - start)
            if best is None or waste < best[3]:
                best = (idx, aligned_start, end, waste)
        if best is None:
            return None
        return best[:3]

    def _insert_free_span(self, start: int, nbytes: int) -> None:
        if nbytes <= 0:
            return
        spans = self._free_spans
        insert_at = 0
        while insert_at < len(spans) and spans[insert_at][0] < start:
            insert_at += 1
        spans.insert(insert_at, (start, nbytes))

        merged: list[tuple[int, int]] = []
        for span_start, span_nbytes in spans:
            if not merged:
                merged.append((span_start, span_nbytes))
                continue
            prev_start, prev_nbytes = merged[-1]
            prev_end = prev_start + prev_nbytes
            if span_start <= prev_end:
                merged[-1] = (
                    prev_start,
                    max(prev_end, span_start + span_nbytes) - prev_start,
                )
            else:
                merged.append((span_start, span_nbytes))
        self._free_spans = merged

    def alloc(self, nbytes: int, align: int = 256, tag: str = "") -> Handle:
        nbytes = int(nbytes)
        align = max(1, int(align))
        if nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")

        if self._arena is None:
            self._bootstrap_arena()

        span = self._find_span(nbytes, align)
        if span is None:
            raise RuntimeError(
                f"UnifiedGpuPool out of space: request={nbytes} align={align} "
                f"used={self._used_bytes} budget={self.total_budget_bytes} tag={tag}"
            )

        span_idx, aligned_start, end = span
        span_start, span_nbytes = self._free_spans.pop(span_idx)
        prefix_nbytes = aligned_start - span_start
        suffix_nbytes = span_start + span_nbytes - end
        if suffix_nbytes > 0:
            self._free_spans.insert(span_idx, (end, suffix_nbytes))
        if prefix_nbytes > 0:
            self._free_spans.insert(span_idx, (span_start, prefix_nbytes))

        handle = Handle(
            handle_id=self._next_handle_id,
            offset=aligned_start,
            nbytes=nbytes,
            reserved_nbytes=nbytes,
            align=align,
            tag=tag,
        )
        self._next_handle_id += 1
        self._handles[handle.handle_id] = handle
        self._used_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._used_bytes)
        return handle

    def free(self, handle: Handle) -> None:
        if handle.handle_id not in self._handles:
            return
        actual = self._handles[handle.handle_id]
        self._used_bytes -= actual.reserved_nbytes
        self._insert_free_span(actual.offset, actual.reserved_nbytes)
        del self._handles[handle.handle_id]

    def tensor_view(
        self,
        handle: Handle,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        expected_nbytes = _numel(shape) * torch.tensor([], dtype=dtype).element_size()
        if expected_nbytes > handle.nbytes:
            raise ValueError(
                f"Requested tensor view exceeds handle size: expected={expected_nbytes} "
                f"available={handle.nbytes} tag={handle.tag}"
            )
        assert self._arena is not None
        return self._arena.narrow(0, handle.offset, expected_nbytes).view(dtype).view(shape)

    def available_bytes(self) -> int:
        if self._arena is not None:
            return sum(span_nbytes for _, span_nbytes in self._free_spans)
        return self.total_budget_bytes


class ElasticLayoutStage(enum.Enum):
    PREFILL = "prefill"
    BATCH_BOUNDARY = "batch_boundary"
    REQUEST_FINISHED = "request_finished"


class ElasticVramCoordinator:
    def __init__(
        self,
        pool: UnifiedGpuPool,
        device: torch.device | str,
    ) -> None:
        self.pool = pool
        self.device = _normalize_device(device)
        self.offloader = None
        self.kv_managers = []
        self.expert_managers = []
        self.graph_rebuild_required = False
        self.graph_rebuild_reasons: list[str] = []

        self.kv_min_demand_bytes = 0
        self.kv_target_demand_bytes = 0
        self.hot_expert_target_bytes = 0

    def configure_budget(self, total_budget_bytes: int) -> None:
        self.pool.configure_budget(total_budget_bytes)
        logger.info(
            "Elastic VRAM budget configured: %.2f MiB",
            total_budget_bytes / 1024**2,
        )

    def register_kv_manager(self, kv_manager) -> None:
        if kv_manager not in self.kv_managers:
            self.kv_managers.append(kv_manager)

    def register_expert_manager(self, expert_manager) -> None:
        if expert_manager not in self.expert_managers:
            self.expert_managers.append(expert_manager)

    def register_offloader(self, offloader) -> None:
        self.offloader = offloader

    def mark_graph_rebuild_required(self, reason: str) -> None:
        self.graph_rebuild_required = True
        self.graph_rebuild_reasons.append(reason)

    def consume_graph_rebuild_required(self) -> tuple[bool, list[str]]:
        required = self.graph_rebuild_required
        reasons = list(self.graph_rebuild_reasons)
        self.graph_rebuild_required = False
        self.graph_rebuild_reasons.clear()
        return required, reasons

    def request_kv_growth(
        self,
        kv_manager,
        target_capacity_bytes: int,
        stage: ElasticLayoutStage,
    ) -> bool:
        self.kv_target_demand_bytes = max(
            self.kv_target_demand_bytes,
            int(target_capacity_bytes),
        )
        current_capacity_bytes = kv_manager.capacity_bytes()
        if target_capacity_bytes <= current_capacity_bytes:
            return True

        bytes_needed = int(target_capacity_bytes) - current_capacity_bytes
        self._evict_experts_until_available(bytes_needed, stage)
        if self.pool.available_bytes() < bytes_needed:
            return False

        changed = kv_manager.ensure_capacity_bytes(target_capacity_bytes)
        if changed:
            self.mark_graph_rebuild_required(f"kv_layout_changed:{stage.value}")
        return changed

    def notify_kv_released(self, kv_manager, target_capacity_bytes: int) -> None:
        changed = kv_manager.shrink_to_capacity_bytes(target_capacity_bytes)
        if changed:
            self.mark_graph_rebuild_required("kv_layout_changed:request_finished")
        self.rebalance_experts(ElasticLayoutStage.REQUEST_FINISHED)

    def rebalance_experts(self, stage: ElasticLayoutStage) -> None:
        available_bytes = self.pool.available_bytes()
        if available_bytes <= 0:
            return

        for manager in sorted(
            self.expert_managers,
            key=lambda item: getattr(item, "resident_priority", 0),
            reverse=True,
        ):
            grown = manager.grow_hot_experts(available_bytes, stage)
            if grown > 0:
                self.hot_expert_target_bytes += grown
                available_bytes -= grown
                self.mark_graph_rebuild_required(f"expert_layout_changed:{stage.value}")
            if available_bytes <= 0:
                return

    def _evict_experts_until_available(
        self,
        bytes_needed: int,
        stage: ElasticLayoutStage,
    ) -> None:
        if bytes_needed <= self.pool.available_bytes():
            return

        shortage = bytes_needed - self.pool.available_bytes()
        for manager in sorted(
            self.expert_managers,
            key=lambda item: getattr(item, "resident_priority", 0),
        ):
            released = manager.evict_cold_experts(shortage, stage)
            if released > 0:
                self.mark_graph_rebuild_required(f"expert_layout_changed:{stage.value}")
                shortage = max(0, bytes_needed - self.pool.available_bytes())
            if shortage == 0:
                return


_GLOBAL_UNIFIED_POOLS: Dict[str, UnifiedGpuPool] = {}
_GLOBAL_COORDINATORS: Dict[str, ElasticVramCoordinator] = {}


def get_or_create_unified_gpu_pool(
    device: torch.device | str,
    *,
    total_budget_bytes: int = 0,
) -> UnifiedGpuPool:
    normalized = str(_normalize_device(device))
    pool = _GLOBAL_UNIFIED_POOLS.get(normalized)
    if pool is None:
        pool = UnifiedGpuPool(device=device, total_budget_bytes=total_budget_bytes)
        _GLOBAL_UNIFIED_POOLS[normalized] = pool
    elif total_budget_bytes:
        pool.configure_budget(total_budget_bytes)
    return pool


def get_or_create_elastic_vram_coordinator(
    device: torch.device | str,
    *,
    pool: Optional[UnifiedGpuPool] = None,
) -> ElasticVramCoordinator:
    normalized = str(_normalize_device(device))
    coordinator = _GLOBAL_COORDINATORS.get(normalized)
    if coordinator is None:
        coordinator = ElasticVramCoordinator(
            pool=pool or get_or_create_unified_gpu_pool(device),
            device=device,
        )
        _GLOBAL_COORDINATORS[normalized] = coordinator
    return coordinator
