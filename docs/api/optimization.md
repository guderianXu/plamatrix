# 非线性优化 API（内部实现）

本页描述 `plamatrix::internal` 实现接口，需显式包含 `plamatrix/internal/...` 头文件；不属于 Eigen 风格公开契约。BA 业务由 PlaBundle 负责，点云体素归组由 PlaPoint 负责。

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

## Schur 约化求解

`solveDampedSchurComplement()` 对两类对角块施加按原始对角尺度缩放的 LM 阻尼，使用 Cholesky
分解 eliminated 块，求 primary 增量后回代 eliminated 增量。`SchurComplementSolverOptions::linearBackend`
显式选择以下约化系统求解路径：

- `Cpu`：矩阵自由 Schur 乘法和块 Jacobi-PCG，不构造完整 CSR。
- `DenseCpu`：直接按稳定 block slot 装配 row-major 下三角稠密 Schur，不经过 CSR；再使用原生分块 Cholesky。
- `SparseCpu`：装配完整对称 CSR，按参数块构造图并执行确定性最小度排序，再进行 PlaMatrix 原生块稀疏
  LLT；符号结构随 workspace 复用，每次 LM 阻尼或法方程数值变化时只更新数值分解。
- `Cuda`：CPU 校验/缓存 CSR 拓扑，CUDA kernel 先按 cross row 复用 3×3 消元变换并装配 Schur 数值，
  再使用 cuSPARSE 支撑的块 Jacobi-PCG；PCG 合并向量更新，并按批读取设备端首个收敛状态。
- `OpenCl`：CPU 校验/缓存相同拓扑，OpenCL kernel 装配 Schur 数值，再在所选 GPU 上完成块 Jacobi-PCG。
- `Vulkan`：上传紧凑块后在 GPU 内补零并转换为 FP16，使用 KHR cooperative matrix 和 FP32 累加装配
  Schur values；PCG 直接绑定装配输出，装配、初始化和首批 8 次迭代共用一次提交。固定拓扑会跳过索引
  转换与上传；9×9 Schur 矩阵自动使用 BSR-style block SpMV，numerical-only 首批路径和后续完整迭代
  批次均复用 command buffer。GPU 会从 Schur 对角块直接执行 9×9 Cholesky 求逆，供 block-Jacobi 使用，
  因而不再计算或上传 CPU primary 逆块。

`SparseCpu` 是不依赖第三方稀疏库的内置能力；CUDA/OpenCL/Vulkan 不可用或设备选择不匹配时会抛出明确异常，
不会隐式执行 CPU。调用方可用 `hasSparseDirectSchurSolver()` 查询能力。报告包含实际后端、设备名、
收敛状态、迭代数、初末残差范数、Schur CSR pattern 是否复用、数值是否在设备装配、组装耗时和线性求解耗时。`blockSpmvUsed` 和 `deviceBlockJacobiUsed` 分别标识 Vulkan 块 SpMV 与 GPU 逆块构造。报告还分别记录 eliminated 小块求逆、Schur 数值累加、CSR 转换、Cholesky 分解、三角回代、残差复核和 eliminated 变量回代；`DenseCpu` 的 CSR 转换时间为零。各路径都支持显式实例化的
`float` 和 `double`；OpenCL `double` 需要设备支持 FP64。Vulkan Schur 当前只接受 `float`、块边长
1 到 16，并要求调用方设置 `useMixedPrecision=true`，不满足条件时抛出明确异常。

调用方需要重复求解相同变量邻接、不同数值或阻尼的系统时，可复用
`SchurComplementSolverWorkspace`。workspace 缓存经过完整拓扑签名校验的 CSR row/column pattern、
块位置和设备数值装配索引，并复用稠密 Schur、参考矩阵、块排列/填充图/稀疏因子、变换后的 cross block、
对角块、逆块、约化 RHS 和 scratch 容量；每次求解都会重写数值，不缓存旧法方程或阻尼，拓扑变化时自动重建。
CPU 稀疏装配以哈希去重、排序恢复确定性 block slot，并用 prefix-sum 连续保存每个 slot 的消元 term；常见
9x3 cross/3 维 eliminated 路径使用固定尺寸内核。固定宽度块中全局零行列会从有效子系统等价消除，但若对应
RHS 非零则求解明确失败。直接解未通过严格残差复核时，最多复用同一因子执行 3 次迭代精化。
稠密装配把每个
下三角 block slot 分配给唯一线程，slot 内按固定 term 顺序累加，所以 OpenMP 线程数变化不会改变求和顺序。
通用 CUDA/OpenCL/Vulkan
`blockPcg()` 也可接收调用方提供的连续 row-major 逆对角块。

## LM 阻尼策略

`LevenbergMarquardtStrategy` 只管理阻尼上下界和接受/拒绝计数。调用方负责计算候选目标函数并调用
`acceptStep()` 或 `rejectStep()`，从而保持业务终止条件、回调和状态发布规则在业务层。
