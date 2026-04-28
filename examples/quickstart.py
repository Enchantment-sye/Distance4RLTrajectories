"""Minimal reachability-metrics quickstart."""

from __future__ import annotations

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.state_metrics import IsolationKernelDistance
from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance
from reachability_metrics.cross_metrics import StateToTrajectoryDistance


def main() -> None:
    dataset = TrajectoryDataset.synthetic(num_trajectories=12, length=24, seed=0)
    states = dataset.states()
    trajectories = [traj.states for traj in dataset.trajectories]

    state_metric = IsolationKernelDistance(ensemble_size=16, subsample_size=8, random_state=0, device="cpu")
    state_metric.fit(states)
    print("state distance", state_metric.pairwise_distance(states[:2], states[2:5]).shape)

    traj_metric = IDKTrajectoryDistance(ensemble_size=16, subsample_size=8, random_state=0, device="cpu")
    traj_metric.fit(trajectories)
    print("trajectory distance", traj_metric.pairwise_distance(trajectories[:3]).shape)

    state_to_traj = StateToTrajectoryDistance(state_metric, aggregation="min")
    print("state-to-trajectory", state_to_traj.pairwise_distance(states[:2], trajectories[:4]).shape)


if __name__ == "__main__":
    main()

