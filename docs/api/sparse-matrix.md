# 稀疏矩阵 API

PlaMatrix 的公开稀疏接口采用 Eigen 5.0.1 的常用类名和调用方式。应用层只依赖
`SparseMatrix`、稀疏表达式、ordering、preconditioner 和 solver；CSR、resident storage、
workspace 及设备句柄属于内部实现。

## 矩阵构建与表达式

```cpp
#include <plamatrix/plamatrix.h>

std::vector<plamatrix::Triplet<double>> entries{
    {0, 0, 4.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 3.0}};

plamatrix::SparseMatrix<double> matrix(2, 2);
matrix.setFromTriplets(entries.begin(), entries.end());
matrix.makeCompressed();

plamatrix::MatrixXd dense = plamatrix::MatrixXd::Zero(100, 3);
plamatrix::SparseMatrix<double> selected = dense.sparseView();
```

`Triplet` 支持自定义重复坐标合并函数。`SparseMatrix` 支持 `coeff()`、`coeffRef()`、
`insert()`、`resize()`、`setZero()`、`setIdentity()`、`reserve()`、`makeCompressed()`、
`prune()`、`pruned()`、`sum()`、`diagonal()`、`transpose()` 和 `adjoint()`。ColMajor 与
RowMajor 矩阵可直接互相构造或赋值。

压缩后可通过 `valuePtr()`、`innerIndexPtr()` 和 `outerIndexPtr()` 访问与 Eigen 相同顺序的
压缩数组；非 const 指针写入会在后续矩阵访问时同步回矩阵语义。`uncompress()` 恢复可装配状态。
已按 outer/inner 索引排序的数据也可以使用 `startVec()`、`insertBack()`、
`insertBackByOuterInner()` 和 `finalize()` 构建。`Map<SparseMatrix<...>>` 与
`Map<const SparseMatrix<...>>` 可零复制映射调用方拥有的压缩存储：

```cpp
int outer[] = {0, 2, 3};
int inner[] = {0, 1, 1};
double values[] = {2.0, 3.0, 4.0};
plamatrix::Map<plamatrix::SparseMatrix<double>> mapped(2, 2, 3, outer, inner, values);
mapped.valuePtr()[1] = 8.0;
```

调用方必须保持映射的三个数组有效，并保证 outer offsets 单调、inner indices 有序且位于范围内。
拥有型 `SparseMatrix` 在装配状态用连续 entry 数组保存坐标和值，`reserve()` 会实际预留容量；
`makeCompressed()` 后，CSC（ColMajor）或 CSR（RowMajor）的 values、inner indices 和 outer offsets
成为唯一主存储，不再保留逐元素树节点或额外压缩缓存。再次插入会进入未压缩装配状态，
`startVec()`/`insertBack()` 则可直接生成有序压缩数组。压缩数组接口和稀疏 `Map` 可零复制接入
CHOLMOD、SuiteSparse、cuSPARSE 等外部后端。

稀疏乘稠密直接遍历上述 CSC/CSR 主存储，不再为 SpMV/SpMM 建立第二份 CSR。迭代求解器的
`compute()` 仍按 Eigen 的分解对象语义保存一份独立数值快照，使原矩阵后续修改不会改变既有求解器；
RowMajor 快照按数组线性复制，ColMajor 则直接由 CSC 计数并填充 CSR，不经过 entry 或 COO 中间副本。

稀疏矩阵支持加减、逐元素乘法、稀疏矩阵乘法和稀疏乘稠密。`SparseView` 是对原稠密
表达式的只读视图，构造视图时不复制数据；赋给 `SparseMatrix` 时才保存满足阈值条件的元素。
`InnerIterator` 按实际存储顺序遍历非零元。

`PermutationMatrix` 可左右作用于稀疏矩阵，并提供 `indices()`、`inverse()`、`transpose()`、
`determinant()` 和有效性检查。

## 直接求解器

| 接口 | 适用问题 | 默认 ordering |
|---|---|---|
| `SimplicialLLT` | 对称正定 | AMD |
| `SimplicialLDLT` | 对称、允许非正主元 | AMD |
| `SparseLU` | 一般方阵 | COLAMD |
| `SparseQR` | 矩形、最小二乘、秩亏 | COLAMD |

```cpp
plamatrix::SparseLU<plamatrix::SparseMatrix<double>> solver;
solver.analyzePattern(matrix);

// 只改变已有坐标的数值时，可以复用符号分析和 ordering。
solver.factorize(matrix);
plamatrix::VectorXd solution = solver.solve(rhs);
```

`compute()` 等价于 `analyzePattern()` 后接 `factorize()`。`factorize()` 会验证尺寸和坐标集合；
结构改变时返回 `InvalidInput`。Simplicial 分解使用稀疏 left-looking 数值分解，`SparseLU`
使用带数值行选主元的稀疏消元，均只保存消元产生的非零因子。`SparseQR` 从稀疏列开始执行
列选主元正交化；Q、R 可能因问题本身的填充而变稠密。

ordering 不是空模板标记：`AMDOrdering` 对对称消元图执行确定性近似最小度排序，
`COLAMDOrdering` 对列交图执行近似最小度排序，`NaturalOrdering` 保留原顺序。求解器公开实际
使用的置换和因子信息。

直接分解目前在 CPU 执行。强制 GPU 策略会返回明确的 `UnsupportedOperation`。
CHOLMOD 可作为今后的可选 factorization provider，但 PlaMatrix 的矩阵、阶段状态、错误语义和
求解器调用形式不依赖 CHOLMOD。

## 迭代求解器与预条件器

```cpp
using Sparse = plamatrix::SparseMatrix<double>;
using Preconditioner = plamatrix::IncompleteCholesky<double>;

plamatrix::ConjugateGradient<Sparse,
                            plamatrix::Lower | plamatrix::Upper,
                            Preconditioner> cg;
cg.setTolerance(1.0e-10).setMaxIterations(500).compute(matrix);
plamatrix::VectorXd x = cg.solve(rhs);

plamatrix::BiCGSTAB<Sparse, plamatrix::IncompleteLUT<double>> bicg;
bicg.preconditioner().setDroptol(1.0e-4).setFillfactor(10);
bicg.compute(nonsymmetric_matrix);
plamatrix::VectorXd y = bicg.solve(nonsymmetric_rhs);
```

可组合预条件器包括：

- `IdentityPreconditioner`
- `DiagonalPreconditioner<Scalar>`
- `IncompleteCholesky<Scalar, UpLo, Ordering>`
- `IncompleteLUT<Scalar, StorageIndex>`

`ConjugateGradient` 和 `BiCGSTAB` 提供 `compute()`、`analyzePattern()`、`factorize()`、
`solve()`、`solveWithGuess()`、`preconditioner()`、`iterations()`、`error()`、`tolerance()`、
`setTolerance()`、`maxIterations()`、`setMaxIterations()` 和 `info()`。两者接受一个或多个稠密
右端。CG 用于 SPD 系统；BiCGSTAB 用于一般非对称系统。

默认 Identity/Diagonal CG 可以自动进入 resident PCG：CUDA 支持 float/double，OpenCL 在设备
能力允许时支持 float/double，Vulkan 当前支持 float。`Auto` 只为足够大的系统支付上传成本；
`GpuPreferred` 优先尝试 GPU 后允许 CPU，`GpuRequired` 要求选中的设备路径成功。使用 IC/ILUT
的 CG 和 BiCGSTAB 当前在 CPU 执行，因为设备后端尚无等价预条件器或 BiCGSTAB 内核。

大型 BA 问题仍应使用 PlaMatrix 的块 Schur 求解器；通用稀疏求解器适合点云、Poisson、
图优化和项目中其他 CSR 线性系统。
