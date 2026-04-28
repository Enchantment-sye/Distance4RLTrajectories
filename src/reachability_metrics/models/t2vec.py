"""PyTorch continuous-state t2vec model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class T2VecBatch:
    """Padded t2vec training batch."""

    source: object
    target: object
    lengths: object
    mask: object


def _require_torch() -> object:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install reachability-metrics[t2vec] to use t2vec") from exc


def make_degraded_batch(
    trajectories: list[np.ndarray],
    *,
    noise_std: float,
    point_dropout: float,
    downsample_keep_prob: float,
    random_state: np.random.Generator,
) -> list[np.ndarray]:
    """Create degraded trajectories for denoising sequence training."""
    degraded = []
    for traj in trajectories:
        values = np.asarray(traj, dtype=np.float32)
        if values.shape[0] == 0:
            degraded.append(values.copy())
            continue
        keep = random_state.random(values.shape[0]) < float(downsample_keep_prob)
        keep[0] = True
        keep[-1] = True
        sampled = values[keep]
        old_t = np.linspace(0.0, 1.0, sampled.shape[0], dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
        restored = np.empty_like(values)
        for dim in range(values.shape[1]):
            restored[:, dim] = np.interp(new_t, old_t, sampled[:, dim])
        if point_dropout > 0:
            mask = random_state.random(restored.shape[0]) < float(point_dropout)
            restored[mask] = 0.0
        if noise_std > 0:
            restored = restored + random_state.normal(scale=float(noise_std), size=restored.shape).astype(np.float32)
        degraded.append(restored.astype(np.float32))
    return degraded


class T2VecModel:
    """GRU encoder/decoder for continuous-state t2vec.

    This wrapper delays importing torch until construction.
    """

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(
                self,
                input_dim: int,
                embedding_dim: int = 128,
                hidden_size: int = 256,
                num_layers: int = 3,
                dropout: float = 0.0,
                embedding_mode: str = "last",
            ) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.embedding_dim = int(embedding_dim)
                self.hidden_size = int(hidden_size)
                self.num_layers = int(num_layers)
                self.embedding_mode = str(embedding_mode)
                gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0
                self.encoder = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=gru_dropout,
                )
                self.to_embedding = nn.Linear(hidden_size, embedding_dim)
                self.from_embedding = nn.Linear(embedding_dim, hidden_size)
                self.decoder = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=gru_dropout,
                )
                self.output = nn.Linear(hidden_size, input_dim)

            def encode(self, x, lengths):  # type: ignore[no-untyped-def]
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

                lengths_cpu = lengths.detach().cpu()
                packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
                packed_out, hidden = self.encoder(packed)
                if self.embedding_mode == "mean":
                    out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
                    mask = (
                        torch.arange(x.shape[1], device=x.device)[None, :]
                        < lengths.to(x.device)[:, None]
                    ).float()
                    pooled = torch.sum(out * mask[:, :, None], dim=1) / torch.clamp(lengths.float()[:, None], min=1.0)
                    return self.to_embedding(pooled)
                return self.to_embedding(hidden[-1])

            def forward(self, source, target, lengths):  # type: ignore[no-untyped-def]
                emb = self.encode(source, lengths)
                h0_single = torch.tanh(self.from_embedding(emb))
                h0 = h0_single.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
                out, _ = self.decoder(target, h0)
                return self.output(out), emb

        return _Impl(*args, **kwargs)

