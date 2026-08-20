# 线性代数 API

## 矩阵乘法 (gemm)

```cpp
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gemm(A_cpu, B_cpu);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void gemm(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemmAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void gemmAsync(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);
```

- **CPU**：使用项目内 B-panel packing、SIMD 寄存器微内核和 OpenMP tile 内核
- **同步 GPU 接口**：`cublasSgemm` / `cublasDgemm`，支持指定 CUDA stream；
  返回前会同步该 stream，结果可立即传回 CPU 或继续参与默认 stream 运算
- **输出复用**：`gemm(A_gpu, B_gpu, C_gpu, stream)` 写入已有输出矩阵，适合循环里避免反复分配
- **异步接口**：`gemmAsync` 只提交 cuBLAS 工作，不主动同步；
  调用方需要用 CUDA event、`cudaStreamSynchronize()` 或后续传输来等待结果
- **示例**：`auto C = gemm(A, B);`

## 逐元素运算

```cpp
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> add(A, B);  // C[i] = A[i] + B[i]

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sub(A, B);  // C[i] = A[i] - B[i]

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> add(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> sub(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void add(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void sub(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> addAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> subAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void addAsync(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void subAsync(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

// elementwise.h：CPU/GPU 同名重载；GPU 另有 Async 和输出复用重载
auto scaled   = scalarMultiply(A, value);
auto shifted  = scalarAdd(A, value);
auto divided  = scalarDivide(A, value);
auto product  = hadamardMultiply(A, B);
auto quotient = hadamardDivide(A, B);
auto absolute = absElements(A);
auto rooted   = sqrtElements(A);
auto clipped  = clampElements(A, min_value, max_value);
```

- CPU 侧小矩阵保持串行，超过内部阈值后使用 OpenMP 并行
- 同步 GPU 接口使用自定义 CUDA kernel (256 threads/block)，支持可选 CUDA stream；
  返回前会同步该 stream
- 所有二元逐元素运算都要求左右矩阵 `rows` 和 `cols` 完全相同；输出复用重载也必须同形。
  当前不支持行向量、列向量或标量矩阵广播
- GPU 输出复用重载要求输出与输入矩阵同尺寸；`*Async` 不主动同步，输入和输出必须保持
  有效直到 stream 完成
- **标量运算**：CPU 矩阵支持 `2.0f * A`，矩阵加法使用 `add(A, B)`；
  标量乘加写作 `auto C = add(2.0f * A, B);`
- **错误和特殊值**：`scalarDivide` 拒绝 `+0/-0` 标量除数；Hadamard 除法按 IEEE 规则产生
  `+/-Inf` 或 NaN。`sqrtElements` 对负数产生 NaN，其他算术、`absElements` 和
  `clampElements` 保留/传播输入 NaN。`clampElements` 的闭区间要求 `min_value <= max_value`

## 归约

```cpp
enum class ReductionAxis { All, Rows, Columns };

auto totals   = sum(input, axis);
auto averages = mean(input, axis);
auto minima   = min(input, axis);
auto maxima   = max(input, axis);
auto min_with_offsets = argMin(input, axis);
auto max_with_offsets = argMax(input, axis);
```

| `ReductionAxis` | 归约方向 | 输出形状 | `argMin/argMax` 索引含义 |
|-----------------|----------|----------|-------------------------|
| `All` | 全部元素 | `1 x 1` | 列优先线性 offset |
| `Rows` | 每行归约所有列 | `input.rows() x 1` | 该行内的列 offset |
| `Columns` | 每列归约所有行 | `1 x input.cols()` | 该列内的行 offset |

- `sum` 的空 lane 为零；`mean/min/max/argMin/argMax` 遇到实际存在但长度为零的 lane 会抛
  `std::invalid_argument`。当输出 lane 数本身为零时，返回对应的 `0 x 1` 或 `1 x 0`
- `float` 的 `sum/mean` 使用 `double` 累积。`mean` 先处理 NaN 和正负 infinity；CPU 在普通求和
  有溢出或舍入风险时切换到按最大绝对值缩放的补偿求和，CUDA 直接使用缩放补偿求和，最后
  除以 lane 长度并转换为输出类型。这避免“有限输入的平均值因中间和溢出”
- NaN 在 value reduction 中传播；`min/max/argMin/argMax` 的相等值和多个 NaN 都选择最低
  source offset。`mean` 对同时含 `+Inf` 和 `-Inf` 的 lane 返回 NaN
- 非法 `ReductionAxis`、输出复用形状错误和空 lane 错误在 GPU launch 前同步抛出

GPU API 为每个操作提供同步 allocating、同步 workspace、同步输出复用，以及对应的
`*Async` allocating/输出复用重载。`ReductionWorkspace` 是 move-only、grow-only、非线程安全
的调用方临时存储：同一 stream 可顺序复用；普通分配不能在异步 launch 中增长，需先同步并
调用 `reserveBytes()`；stream-ordered 分配只能在其所属 stream 增长，跨 stream 前必须同步并
reset 或 `closeAsyncAllocation()`。

归约没有延迟业务状态，也没有 `ReductionWorkspace::checkStatus()`。同步重载等待 stream；
异步重载只入队。allocating async 返回值使用 stream-ordered 分配，输入、输出和 workspace
必须存活到 stream 完成，并在销毁所属 stream 前显式关闭这些异步分配。

## 索引、扫描和稳定紧缩

```cpp
auto offsets = exclusiveScan(counts);
auto rows = gatherRows(input, indices);       // indices: K x 1
scatterRows(values, indices, output);         // values: K x C
auto exact = compactRows(input, keep_mask);   // keep_mask: input.rows() x 1, uint8_t
```

- `exclusiveScan` 保持输入形状，按底层列优先线性顺序执行 exclusive scan；负 count 合法，
  任一前缀加法超出 `Index` 范围时抛 `std::overflow_error`
- `gatherRows` 按 `indices` 顺序生成 `K x input.cols()`，保留重复索引
- `scatterRows` 不清空未命中的目标行；多个源行写同一目标时，最低 source row 获胜。
  CPU 和同步 GPU 重载在发现非法索引时保证输出不变
- `compactRows` 把任意非零 mask 当作 keep，按 source row 升序稳定输出，并返回对应的
  `sourceIndices`

GPU 的 capacity 重载要求 `capacity_output` 为 `R x C`、`capacity_source_indices` 为
`R x 1`、`selected_count` 为 `1 x 1`，其中 `R=input.rows()`、`C=input.cols()`。只有前
`selected_count` 行有效，容量尾部不属于结果。同步 exact wrapper 先写容量输出，再根据
count 分配精确的 `selected_count x C` 和 `selected_count x 1` 结果；异步 API 不提供
exact-size allocating compact，因为精确行数在设备工作完成前未知。

形状错误在 launch 前抛出。GPU scan overflow 和 gather/scatter 越界由
`IndexingWorkspace` 聚合：同步重载内部等待并检查；异步调用方必须先同步所属 stream，再
调用 `workspace.checkStatus(operation)`。状态 batch 保留最低 source offset，可跨同一 stream
上的连续调用累积；overflow 和越界同时存在时 overflow 优先。`checkStatus()` 会消费状态，
即使它抛出异常；未消费状态会阻止 `reserveBytes()` reset 和 `closeAsyncAllocation()`。

`IndexingWorkspace` 是 move-only、grow-only、非线程安全，并绑定异步复用 stream。它的
move assignment 为 `noexcept`：目标现有存储和尚未消费的目标状态会被释放/丢弃，再接管源
workspace 的存储、状态和 stream provenance。因此调用方必须在覆盖目标 workspace 前同步并
消费其状态。

## SVD 分解

```cpp
/// A = U * diag(S) * Vt
/// 返回 (U, S_vector, Vt)，奇异值降序排列
template <typename Scalar, Device Dev>
std::tuple<DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>> svd(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：使用项目内 one-sided Jacobi SVD；64 列以上使用确定性的 round-robin 独立列对并行和两遍正交化补全基
- **GPU**：`cusolverDnSgesvd` (float) / `cusolverDnDgesvd` (double)，legacy cuSOLVER 路径要求 `rows >= cols`
- 返回形状为 `U(m,m)`, `S(min(m,n),1)`, `Vt(n,n)`。

## QR 分解

```cpp
/// A = Q * R (Q 正交, R 上三角)
template <typename Scalar, Device Dev>
std::tuple<DenseMatrix<Scalar, Dev>, DenseMatrix<Scalar, Dev>>
qr(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：尺度安全的 Householder 反射；大矩阵的尾随列更新和 Q 回放按互不重叠的列/行确定性并行
- **GPU**：`cusolverDnSgeqrf` + `cusolverDnSorgqr`

## 对称特征值

```cpp
/// 对称矩阵特征值分解，返回特征值向量 (降序)
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> eigh(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：使用 Householder 对称三对角化加隐式 QL，并按降序返回特征值
- **GPU**：`cusolverDnSsyevd` (分治算法)

### 批量对称 3x3 特征分解

```cpp
auto result = symmetricEigh3x3Batched(compact); // compact: N x 6
// result.eigenvalues:  N x 3
// result.eigenvectors: N x 9
```

单矩阵热路径可使用不分配 `DenseMatrix` 的固定尺寸接口：

```cpp
std::array<Scalar, 3> eigenvalues;
std::array<Scalar, 9> eigenvectors;
symmetricEigh3x3(compact_covariance, &eigenvalues, &eigenvectors);

std::array<Scalar, 9> u;
std::array<Scalar, 3> singular_values;
std::array<Scalar, 9> vt;
svd3x3(matrix, &u, &singular_values, &vt);
```

`symmetricEigh3x3` 与批量接口使用相同的 compact covariance 和升序特征对布局；`svd3x3`
输入/输出为 row-major，奇异值降序，适合局部 PCA、刚体配准和小块优化内核。

每个输入行按 `[xx, xy, xz, yy, yz, zz]` 表示
`[[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]`。每个特征值输出行按升序排列；特征向量输出行为
`[v0x,v0y,v0z,v1x,v1y,v1z,v2x,v2y,v2z]`，即三个与特征值对应的单位列向量依次打包。
`N=0` 返回 `0 x 3` 和 `0 x 9`。

CPU 和 CUDA 都运行固定 8 sweep 的稳定 Jacobi 旋转，每个 sweep 依次处理 `(0,1)`、
`(0,2)`、`(1,2)`。旋转角计算先按相关元素的最大绝对值缩放。排序后，对数值上重复的
特征值组把固定坐标轴投影到该特征空间，并用两遍正交化构造确定性基；每个向量归一化，
最大绝对分量被规范为非负，绝对值并列时选择最低分量下标。

CPU 会先验证完整输入，再分配输出；列数不是 6 或任一元素非有限时抛
`std::invalid_argument`；若确定性重复基无法构造则抛 `std::runtime_error`。GPU 输出复用
同步/异步入口要求输出严格为 `N x 3` 和 `N x 9`，
形状错误在 launch 前抛出。GPU 设备端遇到非有限行或重复特征空间基构造失败时把该行两个
输出都置零，并在 workspace 中记录最低失败行；非有限错误优先于 basis failure。

同步 GPU 重载等待 stream 并调用 `checkStatus()`。异步重载只接受调用方输出和
`SymmetricEigh3x3Workspace`；调用方必须保持输入、两个输出和 workspace 存活，等待 stream
后调用 `checkStatus()`。未消费状态会阻止 reset/close。workspace move assignment 在目标含
未消费状态时抛 `std::logic_error` 且保持源、目标不变；否则释放目标并转移源的存储、状态和
stream provenance。

CPU-only 构建保留 GPU 重载和 workspace 方法的可编译桩：GPU 运算、reserve 和
`checkStatus()` 抛出需要 `PLAMATRIX_WITH_CUDA=ON` 的错误；空 workspace 的
`closeAsyncAllocation()` 是 no-op。CPU `symmetricEigh3x3Batched` 始终可用。

## 线性求解

```cpp
/// 求解 A * X = B，A 必须是方阵且可逆
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> solve(
    const DenseMatrix<Scalar, Dev>& A,
    const DenseMatrix<Scalar, Dev>& B);
```

- **CPU**：列主元高斯消元 (LU 分解)
- **GPU**：`cusolverDnSgetrf` + `cusolverDnSgetrs`
- B 可以是向量 (n×1) 或多右端项矩阵 (n×nrhs)

## 性能建议

| 运算 | GPU 临界尺寸 | 说明 |
|------|-------------|------|
| gemm | 依 CPU/CUDA 和矩阵尺寸而定 | 原生 CPU 使用 packing、SIMD 和 OpenMP |
| add/sub | ~4096 | 内存带宽受限，GPU 优势出现较晚 |
| svd | 依 CPU/CUDA 和矩阵尺寸而定 | CPU 使用原生 one-sided Jacobi；CUDA 使用 cuSOLVER |
| solve | ~256 | GPU LU 分解远超 CPU 高斯消元 |

*注：临界尺寸因硬件而异，请使用 `plamatrix_benchmark` 工具实际测量。*

CUDA benchmark 会在 CUDA run 边界启用 GPU memory pool。
GEMM/add/sub 使用 CUDA event 计时并复用输出矩阵，报告里的 CUDA 时间主要反映设备端计算，
不包含每轮输出分配成本；传输时间使用 pinned host 输入矩阵，单独记录在 `transfer_ms`。
