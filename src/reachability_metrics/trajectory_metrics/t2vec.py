"""Trainable PyTorch t2vec trajectory distance."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from reachability_metrics.data import StatePreprocessor
from reachability_metrics.models.t2vec import T2VecModel
from reachability_metrics.torch_utils import (
    as_trajectory_tensor_list,
    cosine_distance_matrix,
    pairwise_euclidean,
    resolve_torch_device,
    torch_resample_trajectory,
)
from .base import TrajectoryMetric


def _require_torch() -> object:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install reachability-metrics[t2vec] to use T2VecDistance") from exc


class T2VecDistance(TrajectoryMetric):
    """Continuous-state t2vec distance with in-package PyTorch training."""

    def __init__(
        self,
        model_path: str | None = None,
        train_if_missing: bool = False,
        config: dict[str, Any] | None = None,
        normalize: bool = True,
        normalization: str = "standard",
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        embedding_mode: str = "last",
        distance: str = "euclidean",
        batch_size: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        validation_split: float = 0.1,
        noise_std: float = 0.01,
        point_dropout: float = 0.05,
        downsample_keep_prob: float = 0.7,
        loss: str = "mse",
        gradient_clip: float = 1.0,
        amp: bool = False,
        num_workers: int = 0,
        device: str = "auto",
        random_state: int = 0,
        verbose: bool = False,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.train_if_missing = train_if_missing
        self.config = config
        self.normalize = normalize
        self.normalization = normalization
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_mode = embedding_mode
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.noise_std = noise_std
        self.point_dropout = point_dropout
        self.downsample_keep_prob = downsample_keep_prob
        self.loss = loss
        self.gradient_clip = gradient_clip
        self.amp = amp
        self.num_workers = num_workers
        super().__init__(
            device=device,
            dtype="float32",
            batch_size=batch_size,
            return_numpy=return_numpy,
            output_format=output_format,
        )
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _model_kwargs(self, input_dim: int) -> dict[str, Any]:
        cfg = dict(self.config or {})
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("embedding_dim", self.embedding_dim)
        cfg.setdefault("hidden_size", self.hidden_size)
        cfg.setdefault("num_layers", self.num_layers)
        cfg.setdefault("dropout", self.dropout)
        cfg.setdefault("embedding_mode", self.embedding_mode)
        return cfg

    def fit(self, trajectories: Any, y: Any = None) -> "T2VecDistance":
        torch = _require_torch()
        raw_trajs = as_trajectory_tensor_list(trajectories, dtype=torch.float32, device=resolve_torch_device(self.device))
        self.trajectories_ = raw_trajs
        if self.model_path and os.path.exists(self.model_path):
            return self.load(self.model_path)
        if not self.train_if_missing:
            raise FileNotFoundError("model_path is missing and train_if_missing=False")
        self.device_ = resolve_torch_device(self.device)
        self.preprocessor_ = StatePreprocessor(normalize=self.normalize, normalization=self.normalization, device=self.device_)
        self.preprocessor_.fit(raw_trajs)
        proc = [self.preprocessor_.transform_trajectory(t).to(dtype=torch.float32, device=self.device_) for t in raw_trajs]
        self.input_dim_ = int(proc[0].shape[1])
        torch.manual_seed(int(self.random_state))
        if self.device_.type == "cuda":
            torch.cuda.manual_seed_all(int(self.random_state))
        self.model_ = T2VecModel(**self._model_kwargs(self.input_dim_)).to(self.device_)
        self._train(proc)
        if self.model_path:
            self.save(self.model_path)
        return self

    def _pad_batch(self, trajectories: list[Any], sources: list[Any]) -> tuple[Any, Any, Any, Any]:
        torch = _require_torch()
        lengths_list = [int(t.shape[0]) for t in trajectories]
        max_len = max(lengths_list)
        dim = trajectories[0].shape[1]
        target = torch.zeros((len(trajectories), max_len, dim), dtype=torch.float32, device=self.device_)
        source = torch.zeros_like(target)
        mask = torch.zeros((len(trajectories), max_len, 1), dtype=torch.float32, device=self.device_)
        for i, (traj, src) in enumerate(zip(trajectories, sources)):
            target[i, : traj.shape[0]] = traj
            source[i, : src.shape[0]] = src
            mask[i, : traj.shape[0], 0] = 1.0
        return (
            source,
            target,
            torch.as_tensor(lengths_list, dtype=torch.long, device=self.device_),
            mask,
        )

    def _degrade_batch(self, trajectories: list[Any], rng: np.random.Generator) -> list[Any]:
        torch = _require_torch()
        degraded = []
        for traj in trajectories:
            length = int(traj.shape[0])
            if length == 0:
                degraded.append(traj.clone())
                continue
            keep_np = rng.random(length) < float(self.downsample_keep_prob)
            keep_np[0] = True
            keep_np[-1] = True
            keep = torch.as_tensor(keep_np, dtype=torch.bool, device=traj.device)
            sampled = traj[keep]
            restored = torch_resample_trajectory(sampled, length)
            if self.point_dropout > 0:
                drop = torch.as_tensor(rng.random(length) < float(self.point_dropout), dtype=torch.bool, device=traj.device)
                restored = restored.clone()
                restored[drop] = 0.0
            if self.noise_std > 0:
                noise = torch.as_tensor(
                    rng.normal(scale=float(self.noise_std), size=tuple(restored.shape)),
                    dtype=restored.dtype,
                    device=restored.device,
                )
                restored = restored + noise
            degraded.append(restored.to(dtype=torch.float32))
        return degraded

    def _train(self, trajectories: list[Any]) -> None:
        torch = _require_torch()
        rng = np.random.default_rng(self.random_state)
        order = np.arange(len(trajectories))
        rng.shuffle(order)
        val_count = int(round(float(self.validation_split) * len(order)))
        val_idx = set(order[:val_count].tolist())
        train = [trajectories[i] for i in order if int(i) not in val_idx] or trajectories
        val = [trajectories[i] for i in order if int(i) in val_idx] or train
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        use_huber = str(self.loss).lower() == "huber"
        scaler = torch.cuda.amp.GradScaler(enabled=bool(self.amp) and self.device_.type == "cuda")
        best_state = None
        best_val = float("inf")
        self.training_history_ = []
        for epoch in range(int(self.epochs)):
            self.model_.train()
            epoch_losses = []
            rng.shuffle(train)
            for start in range(0, len(train), int(self.batch_size)):
                batch = train[start : start + int(self.batch_size)]
                degraded = self._degrade_batch(batch, rng)
                source, target, lengths, mask = self._pad_batch(batch, degraded)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(self.amp) and self.device_.type == "cuda"):
                    recon, _ = self.model_(source, target, lengths)
                    if use_huber:
                        loss_raw = torch.nn.functional.smooth_l1_loss(recon, target, reduction="none")
                    else:
                        loss_raw = (recon - target) ** 2
                    loss = torch.sum(loss_raw * mask) / torch.clamp(torch.sum(mask) * target.shape[-1], min=1.0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if self.gradient_clip and self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), float(self.gradient_clip))
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().cpu()))
            val_loss = self._validation_loss(val)
            self.training_history_.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss})
            if self.verbose:
                print(f"[t2vec] epoch={epoch} train={np.mean(epoch_losses):.6f} val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_validation_loss_ = float(best_val)

    def _validation_loss(self, trajectories: list[Any]) -> float:
        torch = _require_torch()
        self.model_.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(trajectories), int(self.batch_size)):
                batch = trajectories[start : start + int(self.batch_size)]
                source, target, lengths, mask = self._pad_batch(batch, batch)
                recon, _ = self.model_(source, target, lengths)
                loss = torch.sum(((recon - target) ** 2) * mask) / torch.clamp(torch.sum(mask) * target.shape[-1], min=1.0)
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def transform_tensor(self, trajectories: Any):
        torch = _require_torch()
        if not hasattr(self, "model_"):
            if self.model_path and os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                raise RuntimeError("T2VecDistance must be fitted or loaded before transform")
        raw = as_trajectory_tensor_list(trajectories, dtype=torch.float32, device=self.device_)
        proc = [self.preprocessor_.transform_trajectory(t).to(dtype=torch.float32, device=self.device_) for t in raw]
        self.model_.eval()
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(proc), int(self.batch_size)):
                batch = proc[start : start + int(self.batch_size)]
                source, _, lengths, _ = self._pad_batch(batch, batch)
                emb = self.model_.encode(source, lengths)
                embeddings.append(emb.detach())
        return torch.cat(embeddings, dim=0)

    def transform(self, trajectories: Any):
        return self._return(self.transform_tensor(trajectories))

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        ea = self.transform_tensor(A)
        eb = ea if B is None else self.transform_tensor(B)
        if str(self.distance).lower() == "cosine":
            return cosine_distance_matrix(ea, eb)
        return pairwise_euclidean(ea, eb)

    def pairwise_similarity_tensor(self, A: Any, B: Any | None = None):
        return -self.pairwise_distance_tensor(A, B)

    def save(self, path: str) -> None:
        """Save model, config, and preprocessor."""
        torch = _require_torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model_state": self.model_.state_dict(),
            "model_kwargs": self._model_kwargs(self.input_dim_),
            "preprocessor": self.preprocessor_,
            "input_dim": self.input_dim_,
            "params": self.get_params(deep=False),
        }
        torch.save(payload, path)

    def load(self, path: str) -> "T2VecDistance":
        """Load model, config, and preprocessor."""
        torch = _require_torch()
        try:
            payload = torch.load(path, map_location=resolve_torch_device(self.device), weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=resolve_torch_device(self.device))
        self.preprocessor_ = payload["preprocessor"]
        self.input_dim_ = int(payload["input_dim"])
        self.device_ = resolve_torch_device(self.device)
        self.model_ = T2VecModel(**payload["model_kwargs"]).to(self.device_)
        self.model_.load_state_dict(payload["model_state"])
        self.model_.eval()
        return self
