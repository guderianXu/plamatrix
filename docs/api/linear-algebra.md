# 线性代数 API

应用层的浮点 `Matrix` 可按问题类型选择 Eigen 5.0.1 风格分解。分解对象可重复求解多个右端；
支持 GPU 分解的入口会服从线程级自动后端策略。大型 BA 系统仍使用专用块 Schur 求解器。

| 入口 | 适用问题 | 执行与限制 |
|------|----------|------------|
| `partialPivLu().solve(b)` | 一般非奇异方阵 | CPU；不处理秩亏 |
| `fullPivLu().solve(b)` | 一般方阵、秩与核分析 | 完整行列选主元；CPU |
| `llt().solve(b)` | 对称正定方阵 | CUDA 可用时可自动使用 cuSOLVER |
| `ldlt().solve(b)` | 对称半正定或半负定方阵 | CPU；奇异系统要求右端在值域内 |
| `householderQr().solve(b)` | 满秩最小二乘 | CUDA 使用 cuSOLVER；OpenCL/Vulkan 使用并行 Householder |
| `colPivHouseholderQr().solve(b)` | 宽、超定或秩亏最小二乘 | CPU；提供秩、核维数和列置换 |
| `jacobiSvd().solve(b)` | 秩亏或欠定最小二乘 | 最小范数解；支持 thin/full U、V |
| `bdcSvd().solve(b)` | 大矩阵 SVD 入口 | CPU 使用并行 Jacobi；CUDA 使用 cuSOLVER；OpenCL/Vulkan 使用并行 one-sided Jacobi |
| `completeOrthogonalDecomposition().solve(b)` | 秩亏最小二乘和伪逆 | CPU；当前以 SVD 实现等价求解语义 |

```cpp
plamatrix::Matrix3d coefficients = plamatrix::Matrix3d::Identity();
plamatrix::Vector3d right(1.0, 2.0, 3.0);
plamatrix::Vector3d solution = coefficients.fullPivLu().solve(right);
plamatrix::Vector3d spd_solution = coefficients.llt().solve(right);
auto svd = coefficients.jacobiSvd<plamatrix::ComputeFullU | plamatrix::ComputeFullV>();
```

公开类名和矩阵成员入口与 Eigen 5.0.1 对齐，包括 `LLT`、`FullPivLU`、`HouseholderQR`、`ColPivHouseholderQR`、`JacobiSVD`、`BDCSVD` 和 `CompleteOrthogonalDecomposition`。`lu()` 是 `partialPivLu()` 的项目便利入口；需要与 Eigen 源码逐字迁移时应使用 `partialPivLu()`。SVD 提供 `rank()`、`threshold()`、`setThreshold()`、`singularValues()`、`matrixU()` 和 `matrixV()`；COD 提供 `rank()`、`matrixQTZ()`、`colsPermutation()`、`householderQ()`、`matrixZ()`、`solve()` 和 `pseudoInverse()`。当前 COD 使用 SVD 形成等价的正交分解和最小范数解，因此紧凑内部因子不保证与 Eigen 的 QR 实现逐元素相同。

动态 `PartialPivLU` 的列优先路径使用连续 rank-1 更新和连续三角求解；固定尺寸路径继续使用编译期工作区。
分解对象可复用同一份因子求解多个右端，避免重复分解。

特征值接口包括任意尺寸 `SelfAdjointEigenSolver`、一般实矩阵 `EigenSolver`、复矩阵
`ComplexEigenSolver`、`GeneralizedSelfAdjointEigenSolver` 和 `GeneralizedEigenSolver`。
`HessenbergDecomposition`、`Tridiagonalization`、`RealSchur` 和 `ComplexSchur` 公开 Eigen 同名的
分解对象及 `matrixH()`/`matrixT()`/`matrixQ()`/`matrixU()`、`computeFromHessenberg()` 等入口。
一般矩阵接口返回复数特征值和复数特征向量。`RealQZ` 对实矩阵 pencil 执行正交等价变换，
返回满足 `Q.transpose() * A * Z == S`、`Q.transpose() * B * Z == T` 的广义实 Schur 形式；
`GeneralizedEigenSolver` 从 1x1/2x2 Schur 块提取齐次 `alphas()`/`betas()`，因此保留
`beta == 0` 的无穷特征值，不再通过 `B.inverse() * A` 求解。自伴问题读取下三角并按升序返回
特征值。CUDA、OpenCL 和 Vulkan 后端可加速自伴特征分解；一般、Schur、QZ 和广义问题在 CPU 执行。

实数和复数矩阵都支持 `PartialPivLU`、`FullPivLU`、`HouseholderQR`、
`ColPivHouseholderQR` 与 `JacobiSVD`。复数 QR 使用 Hermitian 投影，复数 SVD 使用单边
Jacobi 核并返回实非负奇异值；thin/full U、V 与 `solve()` 沿用相同接口。

旋转、位姿、欧拉角和相似变换接口见 [Geometry API](geometry.md)。

`PartialPivLU` 与 `FullPivLU` 提供 `rcond()` 的倒 1-范数条件数估计；`FullPivLU` 还提供 `rank()`、`dimensionOfKernel()`、`kernel()`、`image()`、`permutationP()`、`permutationQ()` 和 `determinant()`。`ColPivHouseholderQR` 提供 `matrixQR()`、`householderQ()`、`colsPermutation()` 与阈值控制。`SelfAdjointEigenSolver` 还提供 `operatorSqrt()` 和 `operatorInverseSqrt()`。

稀疏系统可使用 `ConjugateGradient`、`BiCGSTAB`、`SparseLU`、`SparseQR`、`SimplicialLLT` 和 `SimplicialLDLT`；直接分解支持独立的结构分析和数值分解阶段，并可组合 `IncompleteCholesky`、`IncompleteLUT` 与 AMD/COLAMD/Natural ordering，详见[稀疏矩阵](sparse-matrix.md)。显式设备数值核属于[后端实现接口](backend-internals.md)。

自动策略只会选择有等价稳定原语的后端。LLT 支持 CUDA；无选主元 QR、SVD 和自伴特征分解支持
CUDA、OpenCL 与 Vulkan。OpenCL 路径支持 `float`/`double`，其中 FP64 取决于设备能力；Vulkan
路径当前支持 `float`。自动模式按后端使用独立的经验门槛，允许中等规模问题跳过尚未达到收益点的
CUDA 后端并选择 OpenCL/Vulkan；`GpuRequired` 始终严格使用指定后端，不受门槛影响。它指向不支持的
组合时会明确返回后端错误，不会静默改走 CPU。

固定尺寸 LU、FullPivLU 和 QR 的 pivot、Householder 系数及求解临时区使用编译期数组；
动态尺寸矩阵继续使用动态工作区。因而摄影测量常见的 2x2、3x3、6x6、9x9 与 15x15
分解不为这些内部工作区申请堆内存。
