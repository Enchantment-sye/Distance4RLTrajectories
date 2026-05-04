"""Shared sklearn-style metric lifecycle helpers."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import maybe_numpy, resolve_output_format


class TensorOutputMixin:
    """Mixin for torch-first estimators with an optional NumPy output adapter."""

    def _set_output_options(
        self,
        *,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.return_numpy = return_numpy
        self.output_format = output_format

    def _output_format(self) -> str:
        return resolve_output_format(
            getattr(self, "output_format", None),
            return_numpy=bool(getattr(self, "return_numpy", False)),
        )

    def _return(self, value: Any) -> Any:
        return maybe_numpy(value, output_format=self._output_format())


class TransformTensorMixin(TensorOutputMixin):
    """Template method for estimators exposing a tensor transform path."""

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        return self._return(self.transform_tensor(*args, **kwargs))


class PairwiseTensorMetricMixin(TensorOutputMixin):
    """Template methods for pairwise tensor metrics.

    Subclasses keep the sklearn-style public methods and implement the tensor
    core methods. This gives state, trajectory, cross, and set metrics a shared
    output lifecycle without forcing a single input representation.
    """

    def pairwise_distance(self, A: Any | None = None, B: Any | None = None) -> Any:
        return self._return(self.pairwise_distance_tensor(A, B))

    def pairwise_similarity_tensor(self, A: Any | None = None, B: Any | None = None) -> Any:
        return -self.pairwise_distance_tensor(A, B)

    def pairwise_similarity(self, A: Any | None = None, B: Any | None = None) -> Any:
        return self._return(self.pairwise_similarity_tensor(A, B))
