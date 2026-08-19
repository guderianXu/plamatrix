# 稀疏矩阵 API

PlaMatrix 提供 CPU/CUDA CSR 构建、传输、乘法和 SPD 迭代求解。公开入口位于
`plamatrix/sparse/sparse_ops.h` 与 `plamatrix/sparse/iterative_solver.h`。

## COO 转 CSR

```cpp
std::vector<Index> rows{0, 0, 1, 1};
std::vector<Index> cols{0, 1, 0, 1};
std::vector<float> values{4, -1, -1, 4};
auto csr = cooToCsr(2, 2, rows, cols, values);
```

输出按行、列排序。重复坐标会合并，值按原始输入顺序累加。越界坐标、长度不一致
或无法表示的尺寸会抛出明确异常。

CUDA 同步重载接收三个 GPU 列向量。需要完全异步时，预分配准确 nnz 的 GPU CSR：

```cpp
SparseOpsWorkspace workspace;
CSRMatrix<float, Device::GPU> output(rows_count, cols_count, combined_nnz);
cooToCsrAsync(rows_count, cols_count, row_gpu, col_gpu, value_gpu,
              output, workspace, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
workspace.checkStatus("COO conversion");
```

`checkStatus()` 不只是读取设备错误，也完成输出结构的可信状态。检查前不得在另一
stream 消费输出；同一个普通 GPU 存储也不能被两个未完成 stream 重叠写入。

## CSR 传输与生命周期

同步 `toGpu()/toCpu()` 返回可立即使用的矩阵。异步 `copyToGpuAsync()`、
`copyToCpuAsync()` 和 `toGpuAsync()` 要求 pinned host 存储，并记录 producer stream。
必须先同步 producer stream，之后才能跨 stream 读取、覆盖或释放。

调用非 const `values()/colIndices()/rowOffsets()` 会使结构可信状态失效，因为外部别名
可以继续修改存储。自适应 CG/PCG 会重新验证；固定迭代异步接口要求可信结构。

## SpMV 与 SpMM

```cpp
DenseMatrix<float, Device::CPU> x(2, 1);
DenseMatrix<float, Device::CPU> y(2, 1);
spmv(csr, x, y);

DenseMatrix<float, Device::CPU> block(2, 8);
auto product = spmm(csr, block);
```

GPU 热循环应复用输出和 `SparseOpsWorkspace`：

```cpp
SparseOpsWorkspace workspace;
spmvAsync(csr_gpu, x_gpu, y_gpu, workspace, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
workspace.closeAsyncAllocation();
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

输入和输出不能使用同一数据指针。矩阵、输入、输出、workspace 和 stream 必须存活到
异步操作完成。

## CG 与 Jacobi-PCG

```cpp
DenseMatrix<float, Device::CPU> solution(csr.rows(), 1);
solution.fill(0.0f);
IterativeSolverOptions options;
options.maxIterations = 500;
options.relativeTolerance = 1.0e-6;
options.requireConvergence = true;
const auto report = pcg(csr, rhs, solution, options);
```

矩阵必须为 SPD。`pcg()` 默认启用 Jacobi 预条件，并拒绝缺失、非正或接近零的对角元。
`solution` 同时是初始猜测和输出。报告包含 `converged`、`iterations`、
`initialResidual` 和 `finalResidual`。

CUDA 自适应接口额外接收可复用的 `IterativeSolverWorkspace<Scalar>` 和 stream。
`blockPcg()` 接收连续 row-major 逆对角块和显式 `block_size`，在 CUDA/OpenCL 设备上执行块
Jacobi 预条件；逆块向量尺寸必须为 `matrix.rows() * block_size`。
固定轮数 pipeline 可使用 `cgFixedIterationsAsync()` 或 `pcgFixedIterationsAsync()`，
随后调用 `finalizeIterativeSolverReport()`。workspace 绑定首次使用的 stream；切换或销毁
非默认 stream 前，先同步、`closeAsyncAllocation()`，再同步一次完成有序释放。

## 基准回归

```bash
plamatrix_benchmark --mode all --size smoke \
  --case coo_to_csr,spmv,spmm,cg,pcg
```

稀疏 CUDA 行分别记录冷分配、热 workspace、计算/求解和传输时间。自适应 CG/PCG
包含主机收敛检查，因此其计算列表示完整 GPU 求解总时间。
