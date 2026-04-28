"""Train the continuous-state t2vec wrapper on synthetic trajectories."""

from __future__ import annotations

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.trajectory_metrics import T2VecDistance


def main() -> None:
    dataset = TrajectoryDataset.synthetic(num_trajectories=16, length=20, seed=0)
    trajectories = [traj.states for traj in dataset.trajectories]
    metric = T2VecDistance(
        train_if_missing=True,
        normalize=True,
        embedding_dim=16,
        hidden_size=24,
        num_layers=1,
        batch_size=8,
        epochs=2,
        device="cpu",
        random_state=0,
    ).fit(trajectories)
    embeddings = metric.transform(trajectories[:4])
    print("t2vec embeddings", embeddings.shape)


if __name__ == "__main__":
    main()

