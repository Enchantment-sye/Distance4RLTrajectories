# reachability-metrics

`reachability-metrics` 是一个 sklearn 风格的 Python 库，用于离线 MDP 轨迹数据中的相似度、可达性和检索实验。它包含状态距离、轨迹距离、轨迹集合距离、状态到轨迹距离、kNN 重标注、kNN 规划、H 步后继分布匹配、trajectory novelty 和逻辑。


## 适合解决什么问题

| 任务 | 推荐模块 |
| --- | --- |
| 比较两个状态是否局部相似 | `state_metrics` |
| 比较两条轨迹形状是否相似 | `trajectory_metrics` |
| 比较两个未来窗口或后继状态分布 | `HSuccessorDistance`、`IDKTrajectoryDistance`、`GDKTrajectoryDistance`、`TrajectoryWassersteinDistance` |
| 从状态检索最接近的轨迹或目标 | `cross_metrics` |
| 比较两个轨迹集合或技能库 | `set_metrics` |
| 轨迹新奇度、异常轨迹检测 | `TrajectoryNoveltyScorer` |
| kNN goal relabeling | `reachability_metrics.cli.run_relabel` |
| 离线 kNN planning | `reachability_metrics.cli.run_planning` |


## 安装

在新库目录下：

```bash
pip install -e ".[dev,torch,t2vec]"
```

在当前迁移机器上，推荐使用用户指定的 `metra_idk` 环境：

```bash
python -m pip install -e ".[dev,torch,t2vec]"
python -m pytest -q
```

可选依赖：

```bash
pip install -e ".[torch]"              # Isolation Kernel GPU / t2vec 相关 torch 支持
pip install -e ".[d4rl]"               # Minari / Gymnasium 数据加载
pip install -e ".[optimal_transport]"  # POT 最优传输支持
pip install -e ".[t2vec]"              # t2vec 训练依赖
```

核心 numpy/scipy/sklearn 距离不依赖 D4RL、Minari、POT 或 torch。

## 快速开始

```python
import numpy as np

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.state_metrics import IsolationKernelDistance
from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance

trajectories = [np.cumsum(np.random.randn(32, 2), axis=0) for _ in range(20)]
dataset = TrajectoryDataset.from_arrays(trajectories)

# 状态到状态距离
state_metric = IsolationKernelDistance(
    ensemble_size=16,
    subsample_size=8,
    temperature=0.1,
    device="cpu",
    random_state=0,
)
state_metric.fit(dataset.states())
D_state = state_metric.pairwise_distance(
    dataset.trajectories[0].states[:4],
    dataset.states()[:10],
)

# 轨迹到轨迹距离
traj_metric = IDKTrajectoryDistance(
    ensemble_size=16,
    subsample_size=8,
    temperature=0.1,
    device="cpu",
    random_state=0,
)
traj_metric.fit(dataset.trajectories)
D_traj = traj_metric.pairwise_distance(dataset.trajectories[:3])
```

## 数据结构和数据加载

### `Trajectory`

一条离线 MDP 轨迹：

```python
from reachability_metrics.data import Trajectory

traj = Trajectory(
    states=states,       # shape=(T, D)
    actions=actions,     # 可选，shape=(T, A) 或 (T-1, A)
    rewards=rewards,     # 可选
    dones=dones,         # 可选
    timesteps=None,      # 可选；默认 0, 1, ..., T-1
    episode_id=0,
)
```

适合在你已经有 numpy 轨迹数组时使用。`states` 必须是二维数组 `(T, D)`。

### `TrajectoryDataset`

多条轨迹的集合：

```python
from reachability_metrics.data import TrajectoryDataset

dataset = TrajectoryDataset.from_arrays([traj1, traj2, traj3])
X = dataset.states()
S, S_next = dataset.transition_pairs()
windows = dataset.windows(horizon=10, include_current=False)
train, val, test = dataset.split_by_trajectory(0.7, 0.15, 0.15, seed=0)
```

常用方法：

| 方法 | 用途 |
| --- | --- |
| `states()` | 把所有状态堆成 `(N, D)` |
| `transition_pairs()` | 取一步转移 `(s_t, s_{t+1})`，用于动力学距离 |
| `windows(horizon)` | 取同一轨迹内的未来窗口，用于 H 步后继实验 |
| `split_by_trajectory()` | 按整条轨迹切分 train/val/test，避免同轨迹泄漏 |

### synthetic fallback

没有 D4RL/Minari 数据时，可以直接生成合成轨迹：

```python
dataset = TrajectoryDataset.synthetic(
    num_trajectories=40,
    length=32,
    dim=2,
    seed=0,
)
```

实验 CLI 也会在真实数据不可用时 fallback 到 synthetic dataset，保证 smoke test 能跑通。

### D4RL / Minari 加载

```python
from reachability_metrics.data import load_dataset_or_synthetic

dataset = load_dataset_or_synthetic(
    "D4RL/pointmaze/umaze-v2",
    minari_datasets_path=None,
    use_achieved_goal=True,
    synthetic_seed=0,
)
```

支持目标包括：

```text
D4RL/pointmaze/umaze-v2
D4RL/pointmaze/large-v2
D4RL/antmaze/umaze-diverse-v1
```

如果本地没有 Minari/D4RL 数据，函数会返回 synthetic fallback，而不是让核心库 import 失败。

## 状态预处理 `StatePreprocessor`

`StatePreprocessor` 用于在距离计算前做可选归一化、时间特征拼接和滑动窗口展开。

```python
from reachability_metrics.data import StatePreprocessor

pre = StatePreprocessor(
    normalize=True,
    normalization="standard",   # standard, minmax, robust, none
    temporal_feature=None,      # None, sinusoidal, rope, learned_index
    sliding_window=None,        # None 或正整数
    flatten_window=True,
    padding="repeat_first",     # repeat_first, zero, drop
)
pre.fit(dataset.trajectories)
X = pre.transform_states(dataset.states())
T = pre.transform_trajectory(dataset.trajectories[0])
```

如何选择：

| 选项 | 什么时候用 |
| --- | --- |
| `normalize=False` | 状态本身尺度已经有物理意义，例如二维 maze 坐标 |
| `standard` | 默认推荐；不同维度尺度不同但无明显异常值 |
| `minmax` | 状态有固定上下界，希望压到相近范围 |
| `robust` | 有离群点，想用 median/IQR 降低异常值影响 |
| `temporal_feature="sinusoidal"` | 希望距离感知轨迹中的时间位置 |
| `temporal_feature="rope"` | 想使用 RoPE 风格时间编码 |
| `temporal_feature="learned_index"` | 只想拼接归一化时间 `t/T` |
| `sliding_window=w` | 单个状态不足以表达局部动态，需要把最近 `w` 个状态拼起来 |

注意：归一化是可选的。对 PointMaze/AntMaze 的二维位置类实验，常常会显式使用 position-only 或关闭额外状态归一化，以避免无关维度干扰几何可达性。

## 统一 API 风格

状态距离继承 sklearn 风格 `BaseEstimator`，一般有：

```python
metric.fit(X_train)
D = metric.pairwise_distance(X_query, X_ref)
S = metric.pairwise_similarity(X_query, X_ref)
dist, ind = metric.kneighbors(X_query, X_ref, k=20)
```

输入支持：

| 输入 | shape |
| --- | --- |
| 单个状态 | `(D,)` |
| 多个状态 | `(N, D)` |
| 单条轨迹 | `(T, D)` |
| 多条轨迹 | `list[np.ndarray]` 或 `list[Trajectory]` |

轨迹距离一般有：

```python
metric.fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
S = metric.pairwise_similarity(trajectories_a, trajectories_b)
```

注意：`kneighbors` 目前是状态距离基类的接口；轨迹检索可以对 `pairwise_distance` 的结果自行 `argsort`。

## 状态到状态距离

导入：

```python
from reachability_metrics.state_metrics import (
    EuclideanDistance,
    GaussianKernelDistance,
    AdaptiveGaussianDistance,
    MahalanobisDistance,
    TemporalDistance,
    IsolationKernelDistance,
    OneStepDynamicsDistance,
    HSuccessorDistance,
    TaskConditionedStateDistance,
)
```

### `EuclideanDistance`

最直接的欧氏距离。

```python
metric = EuclideanDistance().fit(X_train)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 低维连续状态，例如 PointMaze 的二维坐标。
- 各维度尺度已经可比。
- 需要一个最朴素、可解释的 baseline。

不适合：

- 状态维度尺度差异很大且未归一化。
- 数据密度变化明显，局部结构比全局几何更重要。

### `GaussianKernelDistance`

固定带宽高斯核诱导距离。

```python
metric = GaussianKernelDistance(
    sigma="median",          # median 或 fixed
    sigma_value=None,
    distance_mode="rkhs",    # rkhs 或 one_minus_kernel
).fit(X_train)
S = metric.pairwise_similarity(X_query, X_ref)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 想把距离转成平滑局部相似度。
- 数据分布比较均匀。
- 需要和 IK、Adaptive Gaussian 做核函数 baseline 对比。

注意：

- `sigma="median"` 会用训练集距离中位数估计带宽。
- 固定带宽在密集区和稀疏区可能表现不一致。

### `AdaptiveGaussianDistance`

密度自适应高斯核。每个点的局部尺度来自训练集中第 `k` 近邻距离。

```python
metric = AdaptiveGaussianDistance(
    k=10,
    eps=1e-6,
    distance_mode="one_minus_kernel",
).fit(X_train)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 采样密度不均的离线数据。
- 同一空间中有密集房间和稀疏房间。
- 固定带宽 Gaussian 在某些区域过宽或过窄时。

注意：

- `k` 太小会噪声大；太大又会退化成较全局的尺度。
- 对非常稀疏的区域，局部带宽估计仍然可能不稳定。

### `MahalanobisDistance`

协方差白化后的距离。

```python
metric = MahalanobisDistance(
    covariance_estimator="ledoitwolf",
    implementation="whitening",
    eps=1e-6,
).fit(X_train)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 状态维度相关性明显。
- 不同维度尺度不同，例如位置、速度、角度混在一起。
- 想要一个考虑全局协方差结构的 baseline。

注意：

- 它是全局线性白化，不能自动发现复杂非线性局部结构。
- 如果可达性由障碍物、门、瓶颈等局部结构决定，Mahalanobis 可能不够。

### `TemporalDistance`

基于同一条轨迹内时间间隔的距离。

```python
metric = TemporalDistance(
    mode="same_trajectory_min_gap",
    max_window=None,
)
metric.fit(dataset.trajectories)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 分析 replay buffer 中经验时间邻近性。
- 做 temporal baseline。
- 检查时序采样是否造成虚假的可达性信号。

不适合：

- 作为跨轨迹真实可达性的唯一 ground truth。
- 数据有 sample starvation 或 window mismatch 时，Temporal 容易失败。

### `IsolationKernelDistance`

孤立核距离。每个 ensemble 内从训练状态池采样 anchors，对每个状态做 soft assignment，similarity 为特征内积除以 ensemble 数。

```python
metric = IsolationKernelDistance(
    ensemble_size=100,
    subsample_size=32,
    temperature=0.01,
    device="auto",       # auto, cpu, cuda
    batch_size=4096,
    block_size=4096,
    random_state=0,
).fit(X_train)

Phi = metric.transform(X_query)
S = metric.pairwise_similarity(X_query, X_ref)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 局部结构重要的可达性排序。
- PointMaze/AntMaze 这类有瓶颈、门、局部密度差异的离线数据。
- kNN 重标注、局部目标检索、trajectory novelty。
- 想避免固定带宽核在不同密度区域表现不稳。

注意：

- `ensemble_size` 越大越稳定，但计算更重。
- `subsample_size` 控制每个 ensemble 的局部 partition 粒度。
- `temperature` 小时更接近 hard assignment；大时更平滑。
- GPU 可选使用 torch；没有 torch 时不要导入或使用 IK/t2vec 相关功能。

### `OneStepDynamicsDistance`

比较两个状态的一步后继分布是否相似。

```python
S, S_next = dataset.transition_pairs()
metric = OneStepDynamicsDistance(
    backend="local_knn_nextstate",
    local_knn_m=20,
    distance_metric="jsd",
).fit(S, S_next)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 状态几何接近不等于动力学接近的场景。
- 想比较局部 transition behavior。
- 作为 Dyn-1 baseline。

注意：

- 需要 transition pairs。
- 离线数据覆盖不足时，一步后继估计会受局部样本质量影响。

### `HSuccessorDistance`

比较 H 步未来窗口。

```python
metric = HSuccessorDistance(
    horizon=10,
    gamma=None,
    aggregation="raw_l2",
).fit(dataset)
D = metric.pairwise_distance()
```

适合：

- H 步后继状态距离实验。
- 比较两个状态后续未来窗口是否相似。
- 作为 successor distribution matching 的 raw-H baseline。

注意：

- 只使用同一轨迹内有效未来窗口。
- 如果轨迹太短，窗口数量会减少。

### `TaskConditionedStateDistance`

在基础状态距离上加入 value/task 差异。

```python
base = EuclideanDistance()
value_fn = lambda X: X[:, :1]  # 示例：用户自己的 value function

metric = TaskConditionedStateDistance(
    base_metric=base,
    value_fn=value_fn,
    gamma=1.0,
    value_norm="l2",
    combine="add",
).fit(X_train)
D = metric.pairwise_distance(X_query, X_ref)
```

适合：

- 相似性与任务目标、奖励、价值函数有关。
- 同样几何接近的状态，可能任务价值完全不同。

注意：

- `value_fn` 可以是 Python callable、带 `predict` 的 estimator，或预计算数组。
- 这是任务相关距离，不再是纯几何或纯分布距离。

## 轨迹到轨迹距离

导入：

```python
from reachability_metrics.trajectory_metrics import (
    TrajectoryEuclideanDistance,
    DTWDistance,
    HausdorffDistance,
    FrechetDistance,
    TrajectoryWassersteinDistance,
    KernelMeanEmbedding,
    IDKTrajectoryDistance,
    GDKTrajectoryDistance,
    AdaptiveGDKTrajectoryDistance,
    T2VecDistance,
    TaskConditionedTrajectoryDistance,
)
```

### `TrajectoryEuclideanDistance`

把轨迹对齐长度后 flatten，再算欧氏距离。

```python
metric = TrajectoryEuclideanDistance(
    target_length=32,
    resample="linear",
).fit(trajectories)
D = metric.pairwise_distance(trajectories[:4])
```

适合：

- 轨迹长度相同，或者可以合理重采样到固定长度。
- 想比较整体形状的简单 baseline。

不适合：

- 轨迹速度不同但路径形状类似的情况；这时优先考虑 DTW/Frechet。

### `DTWDistance`

动态时间规整距离，允许时间轴非线性对齐。

```python
metric = DTWDistance(
    point_metric="euclidean",
    window=None,
    normalize=True,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 两条轨迹走了相似路径，但速度不同。
- demonstration、技能片段、路径模板匹配。

注意：

- DTW 计算比普通欧氏距离重。
- `window` 可以限制对齐带宽，加速并避免过度扭曲。

### `HausdorffDistance`

把轨迹看成点集，比较两个点集之间的最坏最近点距离。

```python
metric = HausdorffDistance(
    point_metric="euclidean",
    directed=False,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 关心两条轨迹覆盖区域是否一致。
- 不太关心时间顺序。

注意：

- 对离群点敏感。
- 如果时间顺序重要，优先用 Frechet 或 DTW。

### `FrechetDistance`

离散 Frechet 距离，同时考虑路径顺序和形状。

```python
metric = FrechetDistance(point_metric="euclidean").fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 需要比较曲线形状，并保留顺序。
- 路径规划、轨迹模板、导航路径比较。

注意：

- 相比 Hausdorff，它更尊重轨迹先后顺序。
- 相比 DTW，它更像曲线匹配距离。

### `TrajectoryWassersteinDistance`

把轨迹状态看成经验分布，计算 Wasserstein 距离。

```python
metric = TrajectoryWassersteinDistance(
    point_metric="euclidean",
    p=2,
    regularization=None,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 比较轨迹访问状态的分布，而不是严格时间顺序。
- H 步后继状态集合分布匹配。
- 想要一个最优传输 baseline。

注意：

- 如果安装了 POT，可以使用更完整的 OT 支持。
- 没有 POT 时会使用 scipy fallback 或给出明确提示。

### `KernelMeanEmbedding`

用状态核的 mean embedding 表示一条轨迹。

```python
from reachability_metrics.state_metrics import GaussianKernelDistance
from reachability_metrics.trajectory_metrics import KernelMeanEmbedding

base_kernel = GaussianKernelDistance(sigma="median")
kme = KernelMeanEmbedding(base_kernel, normalize=True).fit(trajectories)
emb = kme.transform(trajectories)
K = kme.pairwise_kernel(trajectories)
D = kme.pairwise_distance(trajectories)
```

适合：

- 把轨迹看成状态分布。
- 需要用状态核扩展到轨迹核。
- 实现 IDK/GDK/Adaptive GDK 的基础思想。

### `IDKTrajectoryDistance`

用 Isolation Kernel 作为 base kernel 的轨迹分布距离。

```python
metric = IDKTrajectoryDistance(
    ensemble_size=100,
    subsample_size=32,
    temperature=0.01,
    device="cpu",
    random_state=0,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
emb = metric.transform(trajectories)
```

适合：

- 轨迹的状态访问分布相似度。
- 局部密度结构重要的轨迹比较。
- H 步后继分布匹配、trajectory novelty。

### `GDKTrajectoryDistance`

用 Gaussian kernel 的 trajectory distribution kernel。

```python
metric = GDKTrajectoryDistance(
    sigma="median",
    sigma_value=None,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 分布核 baseline。
- 数据密度较均匀、固定带宽足够合理的场景。

### `AdaptiveGDKTrajectoryDistance`

用 Adaptive Gaussian kernel 的 trajectory distribution kernel。

```python
metric = AdaptiveGDKTrajectoryDistance(k=10, eps=1e-6).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 状态采样密度不均的轨迹分布比较。
- 想和 IDK/GDK 做分布距离对比。

### `T2VecDistance`

PyTorch 连续状态版 t2vec。它训练 GRU encoder/decoder，从 degraded trajectory 重建原轨迹，再用 encoder embedding 计算距离。

```python
metric = T2VecDistance(
    model_path=None,
    train_if_missing=True,
    normalize=True,
    normalization="standard",
    embedding_dim=32,
    hidden_size=64,
    num_layers=1,
    epochs=2,
    batch_size=8,
    device="cpu",
    random_state=0,
).fit(trajectories)

emb = metric.transform(trajectories)
D = metric.pairwise_distance(trajectories[:4])
metric.save("outputs/t2vec.pt")
```

适合：

- 你希望学习轨迹表征，而不是只用手工距离。
- 轨迹较长、模式复杂、需要压缩成 embedding。
- 后续要做聚类、检索、异常检测。

注意：

- t2vec 需要训练或加载模型，不是免训练距离。
- 小数据 smoke 可以 `epochs=1-2`；正式实验需要更长训练和验证集。
- state normalization 可通过 `normalize` 和 `normalization` 控制。

### `TaskConditionedTrajectoryDistance`

在基础轨迹距离上加入轨迹 value 差异。

```python
base = IDKTrajectoryDistance(device="cpu")
value_fn = lambda X: X[:, :1]

metric = TaskConditionedTrajectoryDistance(
    base_trajectory_metric=base,
    value_fn=value_fn,
    gamma=1.0,
    aggregation="mean",   # mean, endpoint, discounted_sum
    beta=0.99,
).fit(trajectories)
D = metric.pairwise_distance(trajectories_a, trajectories_b)
```

适合：

- 轨迹相似性需要和任务价值、目标进度或 reward proxy 绑定。
- 两条轨迹几何相似但任务结果不同。

## 状态到轨迹、轨迹到集合、集合到集合

导入：

```python
from reachability_metrics.cross_metrics import (
    StateToTrajectoryDistance,
    StateToTrajectoryKMEDistance,
    StateToTrajectorySetDistance,
    TrajectoryToSetDistance,
)
from reachability_metrics.set_metrics import (
    IDK2SetDistance,
    GDK2SetDistance,
    AdaptiveGDK2SetDistance,
    TrajectoryNoveltyScorer,
)
```

### `StateToTrajectoryDistance`

定义状态到轨迹的距离，例如 `min_t d(s, x_t)`。

```python
from reachability_metrics.state_metrics import EuclideanDistance
from reachability_metrics.cross_metrics import StateToTrajectoryDistance

metric = StateToTrajectoryDistance(
    state_metric=EuclideanDistance(),
    aggregation="min",   # min, mean, softmin, kmin_mean
    k=3,
).fit(trajectories)
D = metric.pairwise_distance(states, trajectories)
```

适合：

- 查询某个状态离哪条轨迹最近。
- goal candidate relabeling。
- 判断状态是否被某个 demonstration 或 skill 覆盖。

### `StateToTrajectoryKMEDistance`

用 KME 形式计算状态到轨迹分布的距离。

```python
from reachability_metrics.state_metrics import IsolationKernelDistance
from reachability_metrics.cross_metrics import StateToTrajectoryKMEDistance

metric = StateToTrajectoryKMEDistance(
    base_kernel=IsolationKernelDistance(device="cpu")
).fit(trajectories)
D = metric.pairwise_distance(states, trajectories)
```

适合：

- 不只关心最近点，还关心状态和整条轨迹分布的相似度。
- 与 IDK/GDK 轨迹距离保持一致的核视角。

### `StateToTrajectorySetDistance`

状态到一组轨迹集合的距离。

```python
st = StateToTrajectoryDistance(EuclideanDistance(), aggregation="min")
metric = StateToTrajectorySetDistance(st, aggregation="min").fit(trajectory_sets)
D = metric.pairwise_distance(states, trajectory_sets)
```

适合：

- 查询状态属于哪个技能库、轨迹簇或示范集合。
- 状态到策略库/轨迹库的最近覆盖关系。

### `TrajectoryToSetDistance`

轨迹到轨迹集合的距离。

```python
from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance
from reachability_metrics.cross_metrics import TrajectoryToSetDistance

base = IDKTrajectoryDistance(device="cpu")
metric = TrajectoryToSetDistance(base, aggregation="min").fit(trajectory_sets)
D = metric.pairwise_distance(query_trajectories, trajectory_sets)
```

适合：

- 判断一条新轨迹最接近哪个轨迹簇。
- skill library retrieval。
- imitation dataset 分组检索。

### `IDK2SetDistance`、`GDK2SetDistance`、`AdaptiveGDK2SetDistance`

二层 KME：先把状态集合变成轨迹 embedding，再把轨迹集合变成集合 embedding。

```python
metric = IDK2SetDistance(
    ensemble_size=64,
    subsample_size=16,
    temperature=0.01,
    device="cpu",
).fit(trajectory_sets)

D = metric.pairwise_distance(trajectory_sets)
emb_set = metric.transform_set(trajectory_sets[0])
score = metric.novelty_score(trajectory_sets[0][0])
```

适合：

- 比较两个 trajectory dataset / replay buffer / skill set。
- 评估一批轨迹整体是否分布相似。
- trajectory novelty 和分布漂移检测。

如何选：

| 类 | 什么时候用 |
| --- | --- |
| `IDK2SetDistance` | 局部结构和密度变化重要，推荐默认尝试 |
| `GDK2SetDistance` | 需要固定带宽高斯核 baseline |
| `AdaptiveGDK2SetDistance` | 采样密度不均，希望高斯核自适应 |

### `TrajectoryNoveltyScorer`

轨迹新奇度评分，分数越高表示越不像训练参考轨迹。

```python
scorer = TrajectoryNoveltyScorer(
    method="idk2",
    ensemble_size=64,
    subsample_size=16,
    temperature=0.01,
    device="cpu",
).fit(reference_trajectories)

scores = scorer.novelty_score(query_trajectories)
```

适合：

- 检测离线数据中的异常轨迹。
- 判断新 rollout 是否偏离 reference dataset。
- 主动采样或数据筛选。

## 如何选择距离

| 场景 | 推荐距离 | 原因 |
| --- | --- | --- |
| 低维坐标、尺度一致 | `EuclideanDistance` | 简单稳定，可解释 |
| 平滑局部几何相似度 | `GaussianKernelDistance` | 固定带宽核 baseline |
| 采样密度不均 | `AdaptiveGaussianDistance`、`IsolationKernelDistance` | 局部尺度或孤立划分更稳 |
| 局部可达排序 | `IsolationKernelDistance` | 对 PointMaze/AntMaze 局部结构更敏感 |
| 一步转移行为相似 | `OneStepDynamicsDistance` | 比较局部后继分布 |
| H 步未来窗口 | `HSuccessorDistance`、`IDKTrajectoryDistance`、`GDKTrajectoryDistance`、`TrajectoryWassersteinDistance` | 后继分布匹配实验 |
| 等长轨迹整体形状 | `TrajectoryEuclideanDistance` | 快速 baseline |
| 速度不同但路径类似 | `DTWDistance` | 允许时间轴弯曲 |
| 点集覆盖相似 | `HausdorffDistance` | 不关心时间顺序 |
| 曲线顺序和形状 | `FrechetDistance` | 保留轨迹先后顺序 |
| 状态访问分布 | `IDKTrajectoryDistance`、`GDKTrajectoryDistance`、`AdaptiveGDKTrajectoryDistance` | KME 分布距离 |
| 最优传输分布比较 | `TrajectoryWassersteinDistance` | 直接比较经验分布 |
| 需要学习轨迹 embedding | `T2VecDistance` | 训练 encoder 表征 |
| 轨迹集合比较 | `IDK2SetDistance`、`GDK2SetDistance`、`AdaptiveGDK2SetDistance` | 二层分布 embedding |
| 轨迹新奇度 | `TrajectoryNoveltyScorer` | 距 reference distribution 的距离 |
| 有任务价值函数 | `TaskConditionedStateDistance`、`TaskConditionedTrajectoryDistance` | 把 value 差异纳入距离 |

## 实验 CLI

所有 CLI 都会把表保存到 `tables/`，图保存到 `figures/`，并生成 `report.md`。

### kNN 重标注

```bash
python -m reachability_metrics.cli.run_relabel \
  --datasets D4RL/pointmaze/umaze-v2 D4RL/pointmaze/large-v2 D4RL/antmaze/umaze-diverse-v1 \
  --num_anchors 200 \
  --num_candidates 1000 \
  --top_k 20 \
  --horizon 20 \
  --output_dir outputs/knn_relabeling
```

输出指标包括 Spearman、NDCG@k、goal precision、mean ground-truth reachability、diversity、unique goal ratio 等。

### kNN 规划

```bash
python -m reachability_metrics.cli.run_planning \
  --datasets D4RL/pointmaze/umaze-v2 D4RL/pointmaze/large-v2 D4RL/antmaze/umaze-diverse-v1 \
  --retrieval_top_k 20 \
  --num_queries 200 \
  --output_dir outputs/knn_planning
```

输出指标包括 success rate、path suboptimality、precision，并保存 query path 图。

### H 步后继状态距离

```bash
python -m reachability_metrics.cli.run_successor \
  --datasets D4RL/pointmaze/umaze-v2 D4RL/pointmaze/large-v2 D4RL/antmaze/umaze-diverse-v1 \
  --horizon_values 10 20 50 \
  --output_dir outputs/successor_distance
```

比较 raw H、IDK、GDK、Adaptive GDK、Wasserstein，输出 AUROC、AUPRC 和 Recall@k。

### Reachability alignment

```bash
python -m reachability_metrics.cli.run_alignment \
  --datasets D4RL/pointmaze/umaze-v2 D4RL/pointmaze/large-v2 D4RL/antmaze/umaze-diverse-v1 \
  --output_dir outputs/reachability_alignment
```

用于比较 IK、Gaussian、Adaptive Gaussian、Euclidean、Mahalanobis、Temporal、Dyn-1 与可达性 proxy ground truth 的排序一致性。

### 论文 Fig.2 和 Tables 1-6 复现

```bash
python -m reachability_metrics.cli.reproduce_paper \
  --legacy_outputs_dir /share/shangyy/codes/metra/outputs \
  --output_dir outputs/paper_reproduction \
  --include_figures \
  --verify-paper-values
```

生成：

```text
outputs/paper_reproduction/report.md
outputs/paper_reproduction/paper_source_manifest.json
outputs/paper_reproduction/figures/figure2_simple_data.png
outputs/paper_reproduction/figures/figure2_pointmaze_data_diagram.png
outputs/paper_reproduction/tables/paper_hyperparameters.csv
outputs/paper_reproduction/tables/table1_pointmaze_ndcg_reconstructed.csv
outputs/paper_reproduction/tables/table1_reconstruction_notes.md
outputs/paper_reproduction/tables/table2_relabel_spearman.csv
outputs/paper_reproduction/tables/table3_planning_success.csv
outputs/paper_reproduction/tables/table4_planning_path_suboptimality.csv
outputs/paper_reproduction/tables/table5_planning_precision.csv
outputs/paper_reproduction/tables/table6_successor_h10_auroc.csv
```

复现状态：

- Fig.2 来自旧 `outputs` 中的成品图。
- Tables 2-6 有完整 legacy CSV 来源，并按论文小数位对齐。
- Table 1 没有找到单个完整最终 CSV，因此使用可追踪重构：
  - `direct_csv`：直接来自旧 CSV。
  - `recomputed_from_pair_or_cache`：从 pair summary 或 cache 重算/追踪。
  - `paper_expected_version_conflict`：旧产物存在版本差异。
  - `paper_expected_unresolved_source`：论文值可列出，但未找到完整同源 artifact。

## Smoke test

没有真实数据时，这些命令会 fallback 到 synthetic dataset。

```bash
python -m reachability_metrics.cli.run_successor \
  --datasets D4RL/pointmaze/umaze-v2 \
  --horizon_values 10 \
  --eval_num_pairs 1000 \
  --num_queries 16 \
  --num_candidates 64 \
  --output_dir outputs/smoke_successor

python -m reachability_metrics.cli.run_relabel \
  --datasets D4RL/pointmaze/umaze-v2 \
  --num_anchors 20 \
  --num_candidates 100 \
  --top_k 10 \
  --output_dir outputs/smoke_relabel

python -m reachability_metrics.cli.run_planning \
  --datasets D4RL/pointmaze/umaze-v2 \
  --num_queries 20 \
  --retrieval_top_k 10 \
  --output_dir outputs/smoke_planning
```

## 验证

当前仓库已用以下命令验证：

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python -m pip install -e ".[dev,torch,t2vec]"
/home/shangyy/miniconda3/envs/metra_idk/bin/python -m pytest -q
```

额外 import smoke：

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python - <<'PY'
from reachability_metrics.state_metrics import IsolationKernelDistance
from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance, T2VecDistance
from reachability_metrics.set_metrics import TrajectoryNoveltyScorer
print("imports ok")
PY
```

## 迁移边界

本库只迁移和相似度、距离度量、kNN、后继分布匹配、排序评测、可视化、复现报告相关的代码。以下内容不在本库中：

- METRA / DADS / SAC / DrQ 训练算法。
- IsaacLab / Galaxea / ROS。
- 环境封装、视频生成、在线 RL 训练脚本。
- 硬编码的本地数据路径。

如果只是想使用距离度量，推荐从 `reachability_metrics.state_metrics`、`reachability_metrics.trajectory_metrics`、`reachability_metrics.cross_metrics`、`reachability_metrics.set_metrics` 开始。如果想复现实验，优先使用 `reachability_metrics.cli.*` 命令。

