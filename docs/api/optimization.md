# 非线性优化 API

PlaMatrix 的 optimization 模块提供与具体业务无关的非线性最小二乘基础设施。它不知道相机、点云
或摄影测量模型；调用方负责计算残差和雅可比，并把线性化结果写入块法方程。

## 鲁棒损失

`evaluateHuberLoss(squaredResidualNorm, delta)` 返回 `0.5 * rho(s)` 约定下的代价和
`rho'(s)` 法方程权重。`delta=0` 表示普通平方损失。负数或非有限输入会抛出
`std::invalid_argument`。

## 二分块法方程

`BlockNormalEquations<Scalar>` 描述残差连接一个或多个 primary 块，以及至多一个 eliminated 块的稀疏问题。
典型用途包括相机/三维点、位姿/地标等，但 API 不绑定这些语义。雅可比采用行优先布局：
`residual_size x block_size`。

- `addResidualBlock()`：残差同时连接两类变量。
- `addResidualBlocks()`：残差连接多个 primary 块和一个 eliminated 块，并累计全部直接 primary 交叉项。
- `addPrimaryResidualBlock()`：只连接 primary 块的先验或固定 eliminated 块残差。
- `addPrimaryResidualBlocks()`：只连接多个 primary 块的残差。
- `addEliminatedResidualBlock()`：只连接 eliminated 块的先验或固定 primary 块残差。
- 同一变量对的多条残差会先累计到同一个 cross block，再执行消元。

## Schur-PCG

`solveDampedSchurComplement()` 对两类对角块施加按原始对角尺度缩放的 LM 阻尼，使用 Cholesky
分解 eliminated 块，求 primary 增量后回代 eliminated 增量。`SchurComplementSolverOptions::linearBackend`
显式选择以下约化系统求解路径：

- `Cpu`：矩阵自由 Schur 乘法和块 Jacobi-PCG，不构造完整 CSR。
- `Cuda`：CPU 校验/缓存 CSR 拓扑，CUDA kernel 装配 Schur 数值，再使用 cuSPARSE 支撑的块 Jacobi-PCG。
- `OpenCl`：CPU 校验/缓存相同拓扑，OpenCL kernel 装配 Schur 数值，再在所选 GPU 上完成块 Jacobi-PCG。

CUDA/OpenCL 不可用或设备索引不匹配时会抛出明确异常，不会隐式执行 CPU。报告包含实际后端、设备名、
收敛状态、迭代数、初末残差范数、Schur CSR pattern 是否复用、数值是否在设备装配、组装耗时和线性求解耗时。三条路径都支持显式实例化的
`float` 和 `double`；OpenCL `double` 需要设备支持 FP64。

调用方需要重复求解相同变量邻接、不同数值或阻尼的系统时，可复用
`SchurComplementSolverWorkspace`。workspace 只缓存经过完整拓扑签名校验的 CSR row/column pattern、
块位置和设备数值装配索引，不缓存法方程值、阻尼或 eliminated 块逆；拓扑变化时自动重建。通用 CUDA/OpenCL
`blockPcg()` 也可接收调用方提供的连续 row-major 逆对角块。

## LM 阻尼策略

`LevenbergMarquardtStrategy` 只管理阻尼上下界和接受/拒绝计数。调用方负责计算候选目标函数并调用
`acceptStep()` 或 `rejectStep()`，从而保持业务终止条件、回调和状态发布规则在业务层。
