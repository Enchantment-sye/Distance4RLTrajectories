"""Distributed pairwise execution helpers.

The helpers in this module shard only the left-hand side of an already-fitted
metric.  They intentionally reuse the metric's existing pairwise tensor methods
so the distributed result is the row-wise concatenation of the non-distributed
algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reachability_metrics.torch_utils import (
    block_slices,
    maybe_numpy,
    require_torch,
    resolve_output_format,
    resolve_torch_device,
    torch_dtype,
)


@dataclass(frozen=True)
class _PairwisePlan:
    dist: Any | None
    group: Any | None
    rank: int
    world_size: int
    gather_mode: str
    A_full: Any
    B_full: Any
    context: dict[str, Any]
    start: int
    end: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class _TopKSpec:
    op: str
    k: int
    exclude_self: bool
    sorted: bool

    @property
    def largest(self) -> bool:
        return self.op == "similarity"


def distributed_pairwise_distance(
    metric: Any,
    A: Any | None = None,
    B: Any | None = None,
    *,
    gather: bool | str = True,
    row_block_size: int | None = None,
    process_group: Any | None = None,
    init_process_group: bool = False,
    backend: str | None = None,
    output_format: str | None = None,
    return_numpy: bool = False,
) -> Any:
    """Compute pairwise distances by sharding rows across torch.distributed ranks."""

    return _distributed_pairwise(
        metric,
        A,
        B,
        op="distance",
        gather=gather,
        row_block_size=row_block_size,
        process_group=process_group,
        init_process_group=init_process_group,
        backend=backend,
        output_format=output_format,
        return_numpy=return_numpy,
    )


def distributed_pairwise_similarity(
    metric: Any,
    A: Any | None = None,
    B: Any | None = None,
    *,
    gather: bool | str = True,
    row_block_size: int | None = None,
    process_group: Any | None = None,
    init_process_group: bool = False,
    backend: str | None = None,
    output_format: str | None = None,
    return_numpy: bool = False,
) -> Any:
    """Compute pairwise similarities by sharding rows across torch.distributed ranks."""

    return _distributed_pairwise(
        metric,
        A,
        B,
        op="similarity",
        gather=gather,
        row_block_size=row_block_size,
        process_group=process_group,
        init_process_group=init_process_group,
        backend=backend,
        output_format=output_format,
        return_numpy=return_numpy,
    )


def distributed_topk(
    metric: Any,
    A: Any | None = None,
    B: Any | None = None,
    *,
    k: int = 20,
    op: str = "distance",
    exclude_self: bool = False,
    sorted: bool = True,
    gather: bool | str = True,
    row_block_size: int | None = None,
    process_group: Any | None = None,
    init_process_group: bool = False,
    backend: str | None = None,
    output_format: str | None = None,
    return_numpy: bool = False,
) -> Any:
    """Return distributed top-k values and global column indices.

    ``op="distance"`` returns nearest values with ``largest=False``.
    ``op="similarity"`` returns largest similarities.
    """

    op_key = _normalize_topk_op(op)
    _validate_topk_request(k, exclude_self, B)
    plan = _prepare_pairwise_plan(
        metric,
        A,
        B,
        gather=gather,
        process_group=process_group,
        init_process_group=init_process_group,
        backend=backend,
    )
    spec = _topk_spec(op_key, k, exclude_self, sorted, plan.shape[1])
    local_values, local_indices = _compute_local_topk(metric, plan, spec, row_block_size)

    if plan.gather_mode == "none":
        return _convert_output(
            metric,
            (local_values, local_indices, *_plan_metadata(plan)),
            output_format,
            return_numpy,
        )
    values = _gather_tensor(
        local_values,
        plan.dist,
        plan.group,
        plan.rank,
        plan.world_size,
        plan.gather_mode,
    )
    indices = _gather_tensor(
        local_indices,
        plan.dist,
        plan.group,
        plan.rank,
        plan.world_size,
        plan.gather_mode,
    )
    if values is None or indices is None:
        return None
    return _convert_output(metric, (values, indices), output_format, return_numpy)


def _distributed_pairwise(
    metric: Any,
    A: Any | None,
    B: Any | None,
    *,
    op: str,
    gather: bool | str,
    row_block_size: int | None,
    process_group: Any | None,
    init_process_group: bool,
    backend: str | None,
    output_format: str | None,
    return_numpy: bool,
) -> Any:
    plan = _prepare_pairwise_plan(
        metric,
        A,
        B,
        gather=gather,
        process_group=process_group,
        init_process_group=init_process_group,
        backend=backend,
    )
    local = _compute_local_pairwise(
        metric,
        plan.A_full,
        plan.B_full,
        op,
        plan.context,
        plan.start,
        plan.end,
        row_block_size,
    )
    if plan.gather_mode == "none":
        return _convert_output(metric, (local, *_plan_metadata(plan)), output_format, return_numpy)
    result = _gather_tensor(
        local,
        plan.dist,
        plan.group,
        plan.rank,
        plan.world_size,
        plan.gather_mode,
    )
    return None if result is None else _convert_output(metric, result, output_format, return_numpy)


def _prepare_pairwise_plan(
    metric: Any,
    A: Any | None,
    B: Any | None,
    *,
    gather: bool | str,
    process_group: Any | None,
    init_process_group: bool,
    backend: str | None,
) -> _PairwisePlan:
    dist, group, rank, world_size = _distributed_context(
        process_group=process_group,
        init_process_group=init_process_group,
        backend=backend,
    )
    A_full, B_full, context = _resolve_inputs(metric, A, B)
    _validate_metric_safe(metric, A_full, B_full)
    n_rows = _num_rows(A_full)
    n_cols = _num_rows(B_full)
    start, end = _rank_range(n_rows, rank, world_size)
    return _PairwisePlan(
        dist=dist,
        group=group,
        rank=rank,
        world_size=world_size,
        gather_mode=_normalize_gather(gather),
        A_full=A_full,
        B_full=B_full,
        context=context,
        start=start,
        end=end,
        shape=(n_rows, n_cols),
    )


def _plan_metadata(plan: _PairwisePlan) -> tuple[int, int, tuple[int, int]]:
    return plan.start, plan.end, plan.shape


def _distributed_context(
    *,
    process_group: Any | None,
    init_process_group: bool,
    backend: str | None,
) -> tuple[Any | None, Any | None, int, int]:
    torch = require_torch()
    dist = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available():
        return None, None, 0, 1
    if process_group is not None:
        return dist, process_group, dist.get_rank(process_group), dist.get_world_size(process_group)
    if not dist.is_initialized():
        if not init_process_group:
            return dist, None, 0, 1
        chosen = _choose_backend(torch, backend)
        dist.init_process_group(backend=chosen)
    return dist, None, dist.get_rank(), dist.get_world_size()


def _choose_backend(torch: Any, backend: str | None) -> str:
    if backend and str(backend).lower() != "auto":
        return str(backend)
    return "nccl" if torch.cuda.is_available() else "gloo"


def _normalize_gather(gather: bool | str) -> str:
    if gather is True:
        return "all"
    if gather is False:
        return "none"
    key = str(gather).lower()
    if key in {"all", "rank0", "none"}:
        return key
    raise ValueError("gather must be True, False, 'all', 'rank0', or 'none'")


def _normalize_topk_op(op: str) -> str:
    op_key = str(op).lower()
    if op_key in {"pairwise_distance", "topk_distance"}:
        return "distance"
    if op_key in {"pairwise_similarity", "topk_similarity"}:
        return "similarity"
    if op_key in {"distance", "similarity"}:
        return op_key
    raise ValueError("op must be 'distance' or 'similarity'")


def _validate_topk_request(k: int, exclude_self: bool, B: Any | None) -> None:
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if exclude_self and B is not None:
        raise ValueError("exclude_self=True requires B=None")


def _topk_spec(op: str, k: int, exclude_self: bool, sorted: bool, n_cols: int) -> _TopKSpec:
    if n_cols <= 0:
        raise ValueError("top-k requires at least one column")
    k_eff = min(int(k), n_cols - (1 if exclude_self else 0))
    if k_eff <= 0:
        raise ValueError("top-k has no valid columns after exclude_self")
    return _TopKSpec(op=op, k=k_eff, exclude_self=bool(exclude_self), sorted=bool(sorted))


def _rank_range(n_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    base = int(n_rows) // int(world_size)
    rem = int(n_rows) % int(world_size)
    start = int(rank) * base + min(int(rank), rem)
    end = start + base + (1 if int(rank) < rem else 0)
    return start, end


def _effective_block_size(row_block_size: int | None, metric: Any, local_rows: int) -> int:
    if row_block_size is not None:
        return max(int(row_block_size), 1)
    candidate = getattr(metric, "block_size", None)
    if candidate is None:
        candidate = local_rows
    return max(int(candidate), 1)


def _resolve_inputs(metric: Any, A: Any | None, B: Any | None) -> tuple[Any, Any, dict[str, Any]]:
    context: dict[str, Any] = {}
    if _is_temporal_metric(metric) and _temporal_uses_indices(metric, A, B):
        torch = require_torch()
        n = int(metric.X_fit_.shape[0])
        indices = torch.arange(n, dtype=torch.long, device=metric.episode_ids_.device)
        context["temporal_indices"] = True
        return indices, indices, context

    A_full = _default_A(metric) if A is None else A
    B_full = _default_B(metric, A_full) if B is None else B
    if A_full is None:
        raise ValueError("A is required because no fitted default input was found")
    if B_full is None:
        raise ValueError("B is required because no fitted default right-hand input was found")
    return A_full, B_full, context


def _default_A(metric: Any) -> Any | None:
    if hasattr(metric, "windows_"):
        return metric.windows_
    if _is_state_to_trajectory_metric(metric) and hasattr(metric, "state_metric"):
        return getattr(metric.state_metric, "X_fit_", None)
    if _is_state_to_trajectory_kme_metric(metric) and hasattr(metric, "kme_"):
        return getattr(metric.kme_, "X_fit_", None)
    if _is_state_to_trajectory_set_metric(metric):
        inner = getattr(metric, "state_to_trajectory_metric", None)
        if inner is not None:
            return _default_A(inner)
    if _is_trajectory_to_set_metric(metric):
        inner = getattr(metric, "trajectory_metric_", None)
        if inner is not None and hasattr(inner, "trajectories_"):
            return inner.trajectories_
    if hasattr(metric, "trajectory_sets_"):
        return metric.trajectory_sets_
    if hasattr(metric, "trajectories_"):
        return metric.trajectories_
    if hasattr(metric, "X_fit_"):
        return metric.X_fit_
    return None


def _default_B(metric: Any, A_full: Any) -> Any | None:
    if _is_state_to_trajectory_metric(metric) or _is_state_to_trajectory_kme_metric(metric):
        return getattr(metric, "trajectories_", None)
    if _is_state_to_trajectory_set_metric(metric) or _is_trajectory_to_set_metric(metric):
        return getattr(metric, "trajectory_sets_", None)
    return A_full


def _compute_local_pairwise(
    metric: Any,
    A_full: Any,
    B_full: Any,
    op: str,
    context: dict[str, Any],
    start: int,
    end: int,
    row_block_size: int | None,
) -> Any:
    torch = require_torch()
    if _use_cpu_exact(A_full, B_full, metric):
        full = _compute_pairwise_tensor(metric, A_full, B_full, op, context, 0, _num_rows(A_full))
        return full[start:end]
    chunks = []
    for global_start, global_end, A_block in _iter_row_blocks(A_full, metric, start, end, row_block_size):
        chunks.append(_compute_pairwise_tensor(metric, A_block, B_full, op, context, global_start, global_end))
    if chunks:
        return torch.cat(chunks, dim=0)
    return torch.empty(
        (0, _num_rows(B_full)),
        dtype=_infer_dtype(metric, A_full, B_full),
        device=_infer_device(metric, A_full, B_full),
    )


def _compute_local_topk(
    metric: Any,
    plan: _PairwisePlan,
    spec: _TopKSpec,
    row_block_size: int | None,
) -> tuple[Any, Any]:
    torch = require_torch()
    if _use_cpu_exact(plan.A_full, plan.B_full, metric):
        full_values = _compute_pairwise_tensor(
            metric,
            plan.A_full,
            plan.B_full,
            spec.op,
            plan.context,
            0,
            plan.shape[0],
        )
        _mask_excluded_self(full_values, spec, row_start=0)
        values, indices = _topk_from_values(full_values, spec)
        return values[plan.start : plan.end], indices[plan.start : plan.end]

    chunks_v = []
    chunks_i = []
    for global_start, global_end, A_block in _iter_row_blocks(
        plan.A_full,
        metric,
        plan.start,
        plan.end,
        row_block_size,
    ):
        values = _compute_pairwise_tensor(
            metric,
            A_block,
            plan.B_full,
            spec.op,
            plan.context,
            global_start,
            global_end,
        )
        _mask_excluded_self(values, spec, row_start=global_start)
        vals, idx = _topk_from_values(values, spec)
        chunks_v.append(vals)
        chunks_i.append(idx)
    if chunks_v:
        return torch.cat(chunks_v, dim=0), torch.cat(chunks_i, dim=0)
    return _empty_topk_tensors(metric, plan.A_full, plan.B_full, spec.k)


def _iter_row_blocks(
    A_full: Any,
    metric: Any,
    start: int,
    end: int,
    row_block_size: int | None,
) -> Any:
    if end <= start:
        return
    block_size = _effective_block_size(row_block_size, metric, end - start)
    for rel_start, rel_end in block_slices(end - start, block_size):
        global_start = start + rel_start
        global_end = start + rel_end
        yield global_start, global_end, _slice_rows(A_full, global_start, global_end)


def _topk_from_values(values: Any, spec: _TopKSpec) -> tuple[Any, Any]:
    torch = require_torch()
    vals, idx = torch.topk(
        values,
        k=spec.k,
        dim=1,
        largest=spec.largest,
        sorted=spec.sorted,
    )
    return vals, idx.to(torch.long)


def _mask_excluded_self(values: Any, spec: _TopKSpec, *, row_start: int) -> None:
    if not spec.exclude_self:
        return
    torch = require_torch()
    fill = float("-inf") if spec.largest else float("inf")
    diag_cols = torch.arange(
        int(row_start),
        int(row_start) + int(values.shape[0]),
        dtype=torch.long,
        device=values.device,
    )
    valid = diag_cols < values.shape[1]
    if bool(torch.any(valid)):
        rows = torch.arange(values.shape[0], dtype=torch.long, device=values.device)[valid]
        values[rows, diag_cols[valid]] = fill


def _empty_topk_tensors(metric: Any, A_full: Any, B_full: Any, k: int) -> tuple[Any, Any]:
    torch = require_torch()
    device = _infer_device(metric, A_full, B_full)
    dtype = _infer_dtype(metric, A_full, B_full)
    return (
        torch.empty((0, int(k)), dtype=dtype, device=device),
        torch.empty((0, int(k)), dtype=torch.long, device=device),
    )


def _use_cpu_exact(A_full: Any, B_full: Any, metric: Any) -> bool:
    """Use full-matrix slicing on CPU to preserve bitwise equality.

    CPU BLAS kernels may produce slightly different roundoff for a row block
    than for the same rows inside the full matrix.  The project requires strict
    CPU/gloo equivalence, so CPU execution computes the original full tensor
    and then shards rows.  CUDA keeps the true row-sharded path.
    """

    try:
        return _infer_device(metric, A_full, B_full).type == "cpu"
    except Exception:
        return True


def _compute_pairwise_tensor(
    metric: Any,
    A_block: Any,
    B_full: Any,
    op: str,
    context: dict[str, Any],
    start: int,
    end: int,
) -> Any:
    if context.get("temporal_indices"):
        return _compute_temporal_tensor(metric, A_block, B_full, op)
    if op == "similarity":
        return metric.pairwise_similarity_tensor(A_block, B_full)
    return metric.pairwise_distance_tensor(A_block, B_full)


def _compute_temporal_tensor(metric: Any, A_indices: Any, B_indices: Any, op: str) -> Any:
    torch = require_torch()
    d = metric.pairwise_distance_indices_tensor(A_indices, B_indices)
    if op == "distance":
        return d
    sim = torch.zeros_like(d)
    finite = torch.isfinite(d)
    sim[finite] = 1.0 / (1.0 + d[finite])
    return sim


def _gather_tensor(
    local: Any,
    dist: Any | None,
    group: Any | None,
    rank: int,
    world_size: int,
    gather_mode: str,
) -> Any | None:
    if gather_mode == "none":
        return local
    if world_size == 1 or dist is None:
        return local if gather_mode == "all" or rank == 0 else None

    torch = require_torch()
    rows = torch.tensor([local.shape[0]], dtype=torch.long, device=local.device)
    gathered_rows = [torch.zeros_like(rows) for _ in range(world_size)]
    dist.all_gather(gathered_rows, rows, group=group)
    counts = [int(x.item()) for x in gathered_rows]
    max_rows = max(counts) if counts else 0
    padded_shape = (max_rows, *tuple(local.shape[1:]))
    padded = torch.zeros(padded_shape, dtype=local.dtype, device=local.device)
    if local.shape[0] > 0:
        padded[: local.shape[0]] = local
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    result = torch.cat([part[:count] for part, count in zip(gathered, counts)], dim=0)
    if gather_mode == "rank0" and rank != 0:
        return None
    return result


def _slice_rows(values: Any, start: int, end: int) -> Any:
    if isinstance(values, np.ndarray):
        return values[start:end]
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        return values[start:end]
    if isinstance(values, tuple):
        return list(values[start:end])
    if isinstance(values, list):
        return values[start:end]
    if isinstance(values, Sequence):
        return list(values)[start:end]
    raise TypeError(f"Cannot row-slice object of type {type(values)!r}")


def _num_rows(values: Any) -> int:
    if isinstance(values, np.ndarray):
        return int(values.shape[0])
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        return int(values.shape[0])
    if isinstance(values, Sequence):
        return len(values)
    raise TypeError(f"Cannot infer number of rows for {type(values)!r}")


def _convert_output(metric: Any, value: Any, output_format: str | None, return_numpy: bool) -> Any:
    if output_format is None and hasattr(metric, "_output_format"):
        fmt = metric._output_format()
    else:
        fmt = resolve_output_format(output_format, return_numpy=return_numpy)
    return maybe_numpy(value, output_format=fmt)


def _infer_dtype(metric: Any, A_full: Any, B_full: Any) -> Any:
    torch = require_torch()
    tensor = _find_tensor(A_full)
    if tensor is None:
        tensor = _find_tensor(B_full)
    if tensor is not None and tensor.is_floating_point():
        return tensor.dtype
    dtype = getattr(metric, "dtype", "float32")
    try:
        return torch_dtype(dtype)
    except Exception:
        return torch.float32


def _infer_device(metric: Any, A_full: Any, B_full: Any) -> Any:
    tensor = _find_tensor(A_full)
    if tensor is None:
        tensor = _find_tensor(B_full)
    if tensor is not None:
        return tensor.device
    return resolve_torch_device(getattr(metric, "device", "cpu"))


def _find_tensor(values: Any) -> Any | None:
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        return values
    if hasattr(values, "states"):
        return _find_tensor(values.states)
    if isinstance(values, np.ndarray):
        return None
    if isinstance(values, dict):
        for item in values.values():
            found = _find_tensor(item)
            if found is not None:
                return found
    if isinstance(values, (list, tuple)):
        for item in values:
            found = _find_tensor(item)
            if found is not None:
                return found
    return None


def _validate_metric_safe(metric: Any, A_full: Any, B_full: Any) -> None:
    _validate_task_conditioned(metric)
    _validate_trajectory_euclidean(metric, A_full, B_full)
    for attr in ("base_metric", "base_trajectory_metric", "trajectory_metric", "trajectory_metric_"):
        child = getattr(metric, attr, None)
        if child is not None and child is not metric:
            _validate_metric_safe(child, A_full, B_full)


def _validate_task_conditioned(metric: Any) -> None:
    name = metric.__class__.__name__
    if name not in {"TaskConditionedStateDistance", "TaskConditionedTrajectoryDistance"}:
        return
    value_fn = getattr(metric, "value_fn", None)
    if callable(value_fn) or hasattr(value_fn, "predict"):
        return
    raise ValueError(
        f"{name} with a precomputed value array is not supported by distributed pairwise; "
        "use a callable or an object with predict() so values can be recomputed per shard."
    )


def _validate_trajectory_euclidean(metric: Any, A_full: Any, B_full: Any) -> None:
    if metric.__class__.__name__ != "TrajectoryEuclideanDistance":
        return
    if getattr(metric, "target_length", None) is not None:
        return
    lengths = _trajectory_lengths(A_full) + _trajectory_lengths(B_full)
    if lengths and len(set(lengths)) > 1:
        raise ValueError(
            "Distributed TrajectoryEuclideanDistance with variable-length trajectories "
            "requires target_length to be set."
        )


def _trajectory_lengths(values: Any) -> list[int]:
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        if values.ndim == 2:
            return [int(values.shape[0])]
        if values.ndim == 3:
            return [int(values.shape[1])] * int(values.shape[0])
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            return [int(values.shape[0])]
        if values.ndim == 3:
            return [int(values.shape[1])] * int(values.shape[0])
    if hasattr(values, "states"):
        return [int(values.states.shape[0])]
    if isinstance(values, (list, tuple)):
        out: list[int] = []
        for item in values:
            out.extend(_trajectory_lengths(item))
        return out
    return []


def _is_temporal_metric(metric: Any) -> bool:
    return metric.__class__.__name__ == "TemporalDistance"


def _temporal_uses_indices(metric: Any, A: Any | None, B: Any | None) -> bool:
    if not hasattr(metric, "episode_ids_") or not hasattr(metric, "X_fit_"):
        return False
    if B is not None:
        return False
    if A is None:
        return True
    return _same_tensor_value(A, metric.X_fit_)


def _same_tensor_value(a: Any, b: Any) -> bool:
    torch = require_torch()
    try:
        aa = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=b.dtype, device=b.device)
        return aa.shape == b.shape and bool(torch.allclose(aa, b))
    except Exception:
        return False


def _is_state_to_trajectory_metric(metric: Any) -> bool:
    return metric.__class__.__name__ == "StateToTrajectoryDistance"


def _is_state_to_trajectory_kme_metric(metric: Any) -> bool:
    return metric.__class__.__name__ == "StateToTrajectoryKMEDistance"


def _is_state_to_trajectory_set_metric(metric: Any) -> bool:
    return metric.__class__.__name__ == "StateToTrajectorySetDistance"


def _is_trajectory_to_set_metric(metric: Any) -> bool:
    return metric.__class__.__name__ == "TrajectoryToSetDistance"
