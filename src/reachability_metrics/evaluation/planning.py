"""Planning helpers and maze geodesic utilities."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MazeSpec:
    """Simple maze-free geodesic fallback over XY coordinates."""

    dataset_id: str = "synthetic"

    def geodesic_distances(self, xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
        a = np.asarray(xy_a, dtype=np.float32)
        b = np.asarray(xy_b, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).astype(np.float32)

    def geodesic_for_pairs(self, xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
        a = np.asarray(xy_a, dtype=np.float32)
        b = np.asarray(xy_b, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
            b = b[None, :]
        return np.linalg.norm(a - b, axis=-1).astype(np.float32)


def multi_source_dijkstra(
    graph: dict[str, Any],
    num_nodes: int,
    source_ids: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """Dijkstra search on adjacency arrays."""
    distances = np.full(int(num_nodes), np.inf, dtype=np.float64)
    predecessors = np.full(int(num_nodes), -1, dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for source in np.asarray(source_ids, dtype=np.int64):
        if 0 <= source < num_nodes:
            distances[source] = 0.0
            heapq.heappush(heap, (0.0, int(source)))
    expanded = 0
    while heap:
        current, node = heapq.heappop(heap)
        if current > distances[node]:
            continue
        expanded += 1
        if bool(target_mask[node]):
            path = [node]
            while predecessors[path[-1]] >= 0:
                path.append(int(predecessors[path[-1]]))
            path.reverse()
            return {"found": True, "path_nodes": np.asarray(path), "path_cost": float(current), "expanded_nodes": expanded}
        for neigh, cost in zip(graph["edge_targets"][node], graph["edge_costs"][node]):
            nd = current + float(cost)
            if nd < distances[int(neigh)]:
                distances[int(neigh)] = nd
                predecessors[int(neigh)] = int(node)
                heapq.heappush(heap, (nd, int(neigh)))
    return {"found": False, "expanded_nodes": expanded}


def nearest_nodes(points: np.ndarray, queries: np.ndarray, radius: float) -> list[np.ndarray]:
    """Find node ids within radius for each query point."""
    tree = cKDTree(points)
    return [np.asarray(tree.query_ball_point(q, r=float(radius)), dtype=np.int64) for q in queries]

