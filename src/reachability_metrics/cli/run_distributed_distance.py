"""Torchrun entry point for distributed pairwise distance jobs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from reachability_metrics.distributed import (
    distributed_pairwise_distance,
    distributed_pairwise_similarity,
    distributed_topk,
)
from reachability_metrics.set_metrics import build_set_metric
from reachability_metrics.state_metrics import build_state_metric
from reachability_metrics.trajectory_metrics import build_trajectory_metric
from reachability_metrics.torch_utils import require_torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload_path", required=True)
    parser.add_argument("--payload_format", choices=["auto", "torch", "json"], default="auto")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "gloo", "nccl"])
    parser.add_argument("--map_location", default="cpu")
    parser.add_argument("--allow_pickle", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(f"output_path exists: {args.output_path}")
    torch = require_torch()
    dist = torch.distributed
    payload = _load_payload(args)
    backend = None if args.backend == "auto" else args.backend
    result = None
    try:
        metric = _build_metric(payload.get("metric"))
        A = _resolve_value(payload.get("A"), map_location=args.map_location, allow_pickle=args.allow_pickle)
        B = _resolve_value(payload.get("B"), map_location=args.map_location, allow_pickle=args.allow_pickle)
        fit_data = _resolve_value(payload.get("fit"), map_location=args.map_location, allow_pickle=args.allow_pickle)
        if fit_data is not None:
            metric.fit(fit_data)
        elif payload.get("auto_fit", False):
            metric.fit(A)
        op = str(payload.get("op", "pairwise_distance")).lower()
        row_block_size = payload.get("row_block_size", payload.get("row_chunk_size"))
        common = {
            "gather": "rank0",
            "row_block_size": row_block_size,
            "init_process_group": True,
            "backend": backend,
            "output_format": "torch",
        }
        if op == "pairwise_distance":
            result = distributed_pairwise_distance(metric, A, B, **common)
        elif op == "pairwise_similarity":
            result = distributed_pairwise_similarity(metric, A, B, **common)
        elif op in {"topk_distance", "topk_similarity"}:
            result = distributed_topk(
                metric,
                A,
                B,
                k=int(payload.get("k", 20)),
                op="similarity" if op == "topk_similarity" else "distance",
                exclude_self=bool(payload.get("exclude_self", False)),
                sorted=bool(payload.get("sorted", True)),
                **common,
            )
        else:
            raise ValueError("op must be pairwise_distance, pairwise_similarity, topk_distance, or topk_similarity")

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank == 0:
            output = _build_output(payload, result, op=op, world_size=world_size, backend=args.backend)
            _atomic_torch_save(torch, output, args.output_path)
            if not args.quiet:
                print(args.output_path)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    fmt = args.payload_format
    if fmt == "auto":
        fmt = "json" if args.payload_path.endswith(".json") else "torch"
    if fmt == "json":
        with open(args.payload_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        torch = require_torch()
        load_kwargs = {"map_location": args.map_location}
        try:
            payload = torch.load(args.payload_path, weights_only=not args.allow_pickle, **load_kwargs)
        except TypeError:
            payload = torch.load(args.payload_path, **load_kwargs)
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    return payload


def _build_metric(spec: Any) -> Any:
    if spec is None:
        raise ValueError("payload requires a metric")
    if hasattr(spec, "pairwise_distance_tensor"):
        return spec
    if isinstance(spec, str):
        kind, name = spec.split(":", 1) if ":" in spec else ("state", spec)
        return _build_metric({"kind": kind, "name": name, "kwargs": {}})
    if not isinstance(spec, dict):
        raise TypeError("metric must be a metric object, 'kind:name' string, or dict")
    kind = str(spec.get("kind", "state")).lower()
    name = spec.get("name")
    kwargs = dict(spec.get("kwargs") or {})
    if not name:
        raise ValueError("metric spec requires name")
    if kind == "state":
        return build_state_metric(str(name), **kwargs)
    if kind == "trajectory":
        return build_trajectory_metric(str(name), **kwargs)
    if kind == "set":
        return build_set_metric(str(name), **kwargs)
    raise ValueError("metric kind must be state, trajectory, or set")


def _resolve_value(value: Any, *, map_location: str, allow_pickle: bool) -> Any:
    if isinstance(value, dict) and "path" in value:
        torch = require_torch()
        try:
            loaded = torch.load(value["path"], map_location=map_location, weights_only=not allow_pickle)
        except TypeError:
            loaded = torch.load(value["path"], map_location=map_location)
        key = value.get("key")
        return loaded if key is None else loaded[key]
    return value


def _build_output(payload: dict[str, Any], result: Any, *, op: str, world_size: int, backend: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "op": op,
        "world_size": int(world_size),
        "backend": backend,
    }
    if op.startswith("topk"):
        values, indices = result
        output["values"] = values.detach().cpu()
        output["indices"] = indices.detach().cpu()
        output["k"] = int(payload.get("k", values.shape[1]))
        output["shape"] = list(values.shape)
    else:
        output["values"] = result.detach().cpu()
        output["shape"] = list(result.shape)
    return output


def _atomic_torch_save(torch: Any, value: Any, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(value, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()

