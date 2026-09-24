# 后端实现接口（非公开契约）

本页供 PlaMatrix、PlaPoint、PlaBundle 的后端维护者使用。以下类型与函数位于 `plamatrix::internal`，须显式包含 `plamatrix/internal/...` 头文件，不承诺兼容性。代码片段中的未限定名称按该命名空间解释；应用代码使用公开矩阵、表达式与求解器。

旧 `DenseMatrix`、`DeviceMatrix`、`CSRMatrix`、`COOMatrix`、`Vec3` 名称及旧头文件路径已删除，没有兼容别名。其实现分别归为 `DenseStorage`、`DeviceStorage`、`CsrStorage`、`CooStorage`、`PackedVector3`。这些类只描述后端布局与资源生命周期，不是另一套公开矩阵 API。

自动求值由 `Matrix` 内部选择 CPU/CUDA/OpenCL/Vulkan。`MatrixAccess`、`SparseAccess` 仅供后端求值及验证读取私有存储/诊断；业务层不要使用它们。

## 执行策略和驻留资源

默认 `ExecutionPolicy::Auto` 对小型或逐元素的主机数据使用 CPU；对计算量足够大的
矩阵乘法及已驻留设备的输入使用 GPU。自动候选顺序为 CUDA、OpenCL、Vulkan；不可用或不支持当前
标量的后端会被跳过。阈值是启发式策略，不保证每台设备上都是最佳选择。
没有兼容 GPU 时自动回退 CPU；通过 `MatrixAccess::executionInfo(result)` 查看 `backend`、`reason`、
`bytesUploaded`、`bytesDownloaded` 和 `reusedDeviceInput`。用
`ScopedExecutionPolicy(ExecutionPolicy::GpuRequired)` 可禁止回退并显式检查设备可用性。
调用 `const data()` 会在必要时下载 GPU 结果；非 const `data()` 与可变系数访问还会使设备缓存失效。
列优先 `float` 基础算术可在三个 GPU 后端执行，OpenCL 在设备支持 FP64 时也支持 `double`；
不受支持的运算按执行策略回退或报错。
为自定义内核或专用稀疏求解明确绑定设备时，使用 `ResidentMatrix<Scalar>::copyFrom(host, context)`
和 `toHostMatrix()` 在新 `Matrix` 与 `ExecutionContext` 的设备存储之间传输；这些入口不是
日常密集算术所必需的。

`ExecutionContext::createShared(device)` 可供点云等拥有型对象共享设备上下文。向
`ResidentMatrix(rows, cols, context)` 或 `copyFrom(host, context)` 传入 shared_ptr 时，矩阵
自身保留上下文，返回结果可以比其生产者存活更久；传入 `ExecutionContext&` 时仍是借用，
调用方须保持上下文存活。`ResidentMatrix::clone()` 在同一上下文内深拷贝并继承拥有方式，
`copyFromResident(source)` 写入形状和上下文相同的已有矩阵。CPU 使用内存拷贝，
CUDA/OpenCL/Vulkan 使用各自的设备拷贝命令，没有主机中转。两者返回时拷贝已完成。
CUDA 上传、下载和克隆按该上下文的 stream 排序；在其它 stream 中写入的数据需要调用方先建立同步。

`view<plamatrix::internal::Device::GPU>()` 返回 CUDA 非拥有视图，const 对象返回只读视图；CPU 上下文使用
`view<plamatrix::internal::Device::CPU>()`。视图的设备必须匹配后端；OpenCL/Vulkan 的原生缓冲区不能转换成
CPU/CUDA 指针视图。视图不延长矩阵或上下文的生命周期。

`ResidentVector::copyFrom(vector, context)` / `toHostVector()` 连接新向量与设备向量存储；
`ResidentCsrMatrix::copyFrom(sparse, context)` 可直接上传 Eigen 风格的 `SparseMatrix`，
按排序、去重后的 64 位零起始 CSR 供设备求解器使用。它们用于模块内部的显式设备执行边界，
上层 CPU 算术和稀疏装配无需使用旧容器。


下述 `DenseStorage<Scalar, Device>` 是用于明确管理设备、流和工作区的底层容器。

`DenseStorage` 表示列优先密集存储，供显式设备运算和工作区复用使用。

## 类型定义

```cpp
template <typename Scalar, Device Dev>
class DenseStorage : public DeviceStorage<Scalar, Dev>;
```

- `Scalar`: 元素类型 (`float` 或 `double`)
- `Dev`: 设备类型 (`Device::CPU` 或 `Device::GPU`)

## 构造函数

```cpp
DenseStorage();                        // 空矩阵 (0×0)
DenseStorage(Index rows, Index cols);  // 创建 rows×cols 矩阵，零初始化
```

```cpp
DenseStorage<float, Device::CPU> A(100, 200);  // 100行 200列，全零
DenseStorage<double, Device::GPU> B(50, 50);   // GPU 50×50 double矩阵
```

`DenseStorage()` 是有效的空对象。行列必须非负，`rows * cols` 发生 `Index` 溢出时构造失败。

CUDA 构建还提供未初始化、按 stream 排序的 GPU 分配：

```cpp
auto output = DenseStorage<float, Device::GPU>::uninitializedAsync(rows, cols, stream);
```

该入口只适用于 `Device::GPU`，元素初值未指定。分配、后续使用和释放都必须遵守传入
stream 的顺序；CPU-only 构建会抛出明确的 `PLAMATRIX_WITH_CUDA=ON` 错误。

## 元素访问

```cpp
// CPU 矩阵 — 支持 operator() 直接读写
A(2, 3) = 5.0f;           // 设置第2行第3列 (0-based, 列优先)
float val = A(2, 3);      // 读取

// GPU 矩阵 — 使用 setValue/getValue
B.setValue(0, 1, 42.0f);  // GPU 矩阵设置单个元素 (触发 cudaMemcpy)
float v = B.getValue(0, 1); // GPU 矩阵读取单个元素
```

**列优先索引**: `data[row + col * rows]`

## 填充

```cpp
A.fill(3.14f);  // 所有元素设为 3.14
A.fill(0.0f);   // 清零 (CPU 用 memset, GPU 用 cudaMemset)
```

## 转置

```cpp
auto At = A.transpose();  // 返回新矩阵，维度交换 (cols×rows)
```

- CPU: 嵌套循环
- GPU: 2D CUDA kernel

## 非拥有视图

`MatrixView<Scalar, Dev>` 和 `ConstMatrixView<Scalar, Dev>` 只保存指针、形状与二维 stride，
不分配或释放内存。原始存储必须比视图及其所有派生视图存活得更久。

```cpp
DenseStorage<double, Device::CPU> matrix(6, 8);
auto block = matrix.block(1, 2, 3, 4);       // 零拷贝 3 x 4 子矩阵
auto row = matrix.row(2);                    // 1 x 8
auto col = matrix.col(3);                    // 6 x 1
auto transposed = matrix.transposeView();    // 零拷贝转置视图

block(0, 0) = 42.0; // 直接修改 matrix(1, 2)
```

外部内存可以按列优先、行优先或显式 stride 映射：

```cpp
auto image = makeRowMajorView<float, Device::CPU>(pixels, height, width);
auto packed = makeColumnMajorView<double, Device::CPU>(values, rows, cols);
auto padded = makeMatrixView<float, Device::CPU>(data, rows, cols,
                                                  row_stride, col_stride);
```

视图可以继续调用 `block()`、`row()`、`col()` 和 `transpose()`。常量矩阵只生成
`ConstMatrixView`，可变视图可以隐式转换为常量视图，反向转换不允许。CPU 视图支持
`operator()`；GPU 视图只表达 device pointer 与布局，不能从主机直接解引用。

`DenseStorage::transpose()` 仍返回拥有独立存储的转置矩阵；需要零拷贝布局交换时使用
`transposeView()`。当前算术算子仍接收拥有型 `DenseStorage`，视图算子重载将在后续融合执行层中补充。

CPU-only 构建 (`PLAMATRIX_WITH_CUDA=OFF`) 下仍可构造 `Device::GPU` 矩阵并做显式传输测试，
但 GPU 算法入口（非零 `fill()`、GPU `transpose()`、GPU `add/sub/gemm` 等）
会抛出明确异常或不可用；
实际业务应使用 `Device::CPU`，需要 GPU 算法时启用 CUDA 构建。

## 设备传输

```cpp
auto A_gpu = A.toGpu();  // CPU → GPU (cudaMemcpy HostToDevice)
auto A_cpu = A_gpu.toCpu();  // GPU → CPU (cudaMemcpy DeviceToHost)

auto B_gpu = A.toGpuAsync(stream);  // CPU → GPU (cudaMemcpyAsync)
auto B_cpu = B_gpu.toCpuAsync(stream);  // GPU → CPU (cudaMemcpyAsync)
```

在此底层 API 中传输是一次性的、显式的、昂贵的；上层 `Matrix` 会按需自动搬移数据。

普通 CPU 矩阵使用 pageable host memory。需要让 `cudaMemcpyAsync` 满足真正异步传输条件时，
使用 pinned/page-locked CPU 矩阵：

```cpp
auto pinned = DenseStorage<float, Device::CPU>::pinned(rows, cols);
bool is_pinned = pinned.isPinnedHost(); // true
```

异步传输返回后不会主动同步；调用方必须在读取 CPU 结果或跨 stream 使用 GPU 结果前
等待对应 stream。
热循环里可以复用输出矩阵：

```cpp
auto host = DenseStorage<float, Device::CPU>::pinned(A.rows(), A.cols());
DenseStorage<float, Device::GPU> gpu(A.rows(), A.cols());
auto back = DenseStorage<float, Device::CPU>::pinned(A.rows(), A.cols());

host.copyToGpuAsync(gpu, stream);
gpu.copyToCpuAsync(back, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

如果 host 指针不是 pinned memory，`cudaMemcpyAsync` 仍可能在驱动内部阻塞。
异步传输 API 仍提供明确的 stream ordering 和输出复用入口。

## GPU 内存池

默认 GPU 分配继续直接使用 `cudaMalloc/cudaFree`。需要减少同尺寸矩阵反复分配成本时，
可显式开启进程内 GPU memory pool：

```cpp
GpuAllocator<float>::setMemoryPoolEnabled(true);

{
    DenseStorage<float, Device::GPU> tmp(rows, cols);
} // tmp 的 block 进入 pool，后续同字节数分配可复用

GpuAllocator<float>::releaseMemoryPool();
GpuAllocator<float>::setMemoryPoolEnabled(false);
```

内存池按字节数缓存空闲 block，`releaseMemoryPool()` 会释放当前缓存。
异步 kernel 或异步传输仍要求调用方保证相关矩阵在 stream 完成前保持有效。
`uninitializedAsync()` 创建的 stream-ordered 分配不进入普通 GPU memory pool。

## GPU 异步计算和输出复用

同步 GPU 运算会在返回前同步传入的 CUDA stream，适合直接取结果：

```cpp
auto C_gpu = gemm(A_gpu, B_gpu);
auto D_gpu = add(A_gpu, B_gpu);
```

需要在循环或 benchmark 中减少分配和同步开销时，可复用输出矩阵并使用异步接口：

```cpp
DenseStorage<float, Device::GPU> C_gpu(A_gpu.rows(), B_gpu.cols());
DenseStorage<float, Device::GPU> D_gpu(A_gpu.rows(), A_gpu.cols());

gemmAsync(A_gpu, B_gpu, C_gpu, stream);
addAsync(A_gpu, A_gpu, D_gpu, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

`gemmAsync`、`addAsync`、`subAsync` 只提交 GPU 工作，不主动同步；
调用方负责保证输入/输出矩阵在对应 stream 完成前保持有效。

部分 allocating async API（例如 `sumAsync`、`argMinAsync`、`exclusiveScanAsync` 和
`gatherRowsAsync` 的返回值重载）使用 `uninitializedAsync()`。这类返回矩阵必须一直存活到
stream 完成，并在 stream 销毁前显式释放：

```cpp
auto result = sumAsync(input, ReductionAxis::All, workspace, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
// 此时可消费 result；在销毁 stream 前入队释放。
result.closeAsyncAllocation();
workspace.closeAsyncAllocation();
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

`closeAsyncAllocation()` 在保留的创建 stream 上调用 `cudaFreeAsync`。成功后矩阵变为
`0 x 0`，重复关闭空矩阵是 no-op；对普通非空 GPU 矩阵调用会抛出 `std::logic_error`。
若需要观察释放错误，应显式调用该方法；析构只提供 `noexcept` 后备释放，因此保留的
stream 必须至少存活到显式关闭或析构入队完成。

## 基类方法

```cpp
Index rows() const;       // 行数
Index cols() const;       // 列数
Index size() const;       // 总元素数 (rows × cols)
Scalar* data();           // 原始指针
const Scalar* data() const;
static constexpr Device device();  // 设备类型
```

## Move 语义

DenseStorage 禁止拷贝，支持移动：

```cpp
DenseStorage<float, Device::CPU> A(100, 100);
DenseStorage<float, Device::CPU> B(std::move(A));  // A 变为空
// A.data() == nullptr   (移动后源矩阵置空)
```

移动构造和移动赋值都会转移数据指针、pinned/普通 host 分配类型，以及 GPU 普通/
stream-ordered 分配类型和创建 stream。移动赋值先释放目标对象原有存储，再接管源对象；
源对象统一置为 `0 x 0`、`data() == nullptr`，不再保留 stream provenance。因而对尚有
异步工作使用的目标矩阵做移动赋值，同样要求调用方先完成相关 stream 生命周期管理。

## 完整示例

```cpp
#include <plamatrix/internal/backend.h>

// 创建 CPU 矩阵
plamatrix::internal::DenseStorage<float, plamatrix::internal::Device::CPU> A(3, 4);
A(0, 0) = 1.0f;  A(1, 0) = 2.0f;  A(2, 0) = 3.0f;
A(0, 1) = 4.0f;  A(1, 1) = 5.0f;  A(2, 1) = 6.0f;
// ... 填充剩余元素 ...

// 转移到 GPU 计算
auto A_gpu = A.toGpu();
auto C_gpu = plamatrix::internal::gemm(A_gpu, A_gpu.transpose());

// 取回结果
auto C = C_gpu.toCpu();
```

## 稀疏后端


入口位于 `plamatrix/internal/sparse/sparse_ops.h` 与 `plamatrix/internal/sparse/iterative_solver.h`。

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
CsrStorage<float, Device::GPU> output(rows_count, cols_count, combined_nnz);
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
DenseStorage<float, Device::CPU> x(2, 1);
DenseStorage<float, Device::CPU> y(2, 1);
spmv(csr, x, y);

DenseStorage<float, Device::CPU> block(2, 8);
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
DenseStorage<float, Device::CPU> solution(csr.rows(), 1);
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

CPU CSR 的 `cg()`/`pcg()` 也可直接接收 `Matrix<Scalar, Dynamic, 1>` 类型的
`rhs` 和 `solution`，无需转成 `DenseStorage`；两个重载使用同一个求解内核。
目前只实例化 float/double，`GpuRequired` 会报错。GPU 稀疏求解仍使用下述设备接口。

CUDA 自适应接口额外接收可复用的 `IterativeSolverWorkspace<Scalar>` 和 stream。
`blockPcg()` 接收连续 row-major 逆对角块和显式 `block_size`，在 CUDA/OpenCL/Vulkan 设备上执行块
Jacobi 预条件；逆块向量尺寸必须为 `matrix.rows() * block_size`。
`IterativeSolverOptions::convergenceCheckInterval` 控制 CUDA 自适应求解器隔多少轮把状态同步到主机，
默认值 1 保持逐轮检查。大于 1 时，设备仍在首个满足容差的轮次记录迭代数并冻结解，批内其余已提交轮次
只做数值 no-op；CPU/OpenCL 只校验该值为正，不改变现有判停频率。
固定轮数 pipeline 可使用 `cgFixedIterationsAsync()` 或 `pcgFixedIterationsAsync()`，
随后调用 `finalizeIterativeSolverReport()`。workspace 绑定首次使用的 stream；切换或销毁
非默认 stream 前，先同步、`closeAsyncAllocation()`，再同步一次完成有序释放。

## 基准回归

```bash
plamatrix_benchmark --mode all --size smoke \
  --case coo_to_csr,spmv,spmm,cg,pcg
```

稀疏 CUDA 行分别记录冷分配、热 workspace、计算/求解和传输时间。自适应 CG/PCG
包含主机收敛检查，因此其计算列表示完整 GPU 求解总时间。设置批量检查时，该时间也包含最后一个批次中
冻结后的设备 no-op。

## 密集数值核


## 矩阵乘法 (gemm)

```cpp
template <typename Scalar>
DenseStorage<Scalar, Device::CPU> gemm(A_cpu, B_cpu);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> gemm(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void gemm(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> gemmAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

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
DenseStorage<Scalar, Device::CPU> add(A, B);  // C[i] = A[i] + B[i]

template <typename Scalar>
DenseStorage<Scalar, Device::CPU> sub(A, B);  // C[i] = A[i] - B[i]

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> add(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> sub(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void add(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
void sub(A_gpu, B_gpu, C_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> addAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

template <typename Scalar>
DenseStorage<Scalar, Device::GPU> subAsync(A_gpu, B_gpu, cudaStream_t stream = nullptr);

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

// 单次遍历计算 alpha * A + beta * B，不生成中间矩阵
auto combined = axpby(alpha, A, beta, B);
axpby(alpha, A, beta, B, output); // 复用输出，允许 output 与 A 或 B 完全同址

// 三项线性组合仍只遍历和写入一次
auto combined3 = linearCombination(alpha, A, beta, B, gamma, C);
linearCombination(alpha, A, beta, B, gamma, C, output);
```

- CPU 侧小矩阵保持串行，超过内部阈值后使用 OpenMP 并行
- 同步 GPU 接口使用自定义 CUDA kernel (256 threads/block)，支持可选 CUDA stream；
  返回前会同步该 stream
- 所有二元逐元素运算都要求左右矩阵 `rows` 和 `cols` 完全相同；输出复用重载也必须同形。
  当前不支持行向量、列向量或标量矩阵广播
- GPU 输出复用重载要求输出与输入矩阵同尺寸；`*Async` 不主动同步，输入和输出必须保持
  有效直到 stream 完成
- CPU `axpby` 把两个缩放和一次相加融合为单次遍历，并使用未初始化结果存储，避免
  `scalarMultiply(add(A, B), scale)` 的中间矩阵、清零和第二次内存遍历。它提供 allocating、
  `DenseStorage` 输出复用和任意正 stride 的 `MatrixView` 重载；输出与任一输入完全相同的 view mapping
  可安全原地执行，其他输入/输出重叠映射及自身元素重叠的输出 view 会在写入前抛出
  `std::invalid_argument`
- CPU `linearCombination` 把 `alpha*A + beta*B + gamma*C` 融合为一次遍历，提供与 `axpby`
  相同的 allocating、输出复用、任意正 stride view 和安全别名语义
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
auto inner_product = dot(A, B);  // 不生成 hadamardMultiply(A, B) 中间矩阵
auto energy = squaredNorm(A);    // 直接归约逐元素平方
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
- CPU `dot/squaredNorm` 直接融合逐元素乘法与全量归约，不分配中间矩阵；它们支持任意正 stride
  view，空输入返回零，`float` 输入使用 `double` 累积
- NaN 在 value reduction 中传播；`min/max/argMin/argMax` 的相等值和多个 NaN 都选择最低
  source offset。`mean` 对同时含 `+Inf` 和 `-Inf` 的 lane 返回 NaN
- 非法 `ReductionAxis`、输出复用形状错误和空 lane 错误在 GPU launch 前同步抛出

GPU API 为上述轴归约操作提供同步 allocating、同步 workspace、同步输出复用，以及对应的
`*Async` allocating/输出复用重载。`ReductionWorkspace` 是 move-only、grow-only、非线程安全
的调用方临时存储：同一 stream 可顺序复用；普通分配不能在异步 launch 中增长，需先同步并
调用 `reserveBytes()`；stream-ordered 分配只能在其所属 stream 增长，跨 stream 前必须同步并
reset 或 `closeAsyncAllocation()`。

`sum/min/max` 还提供连续列优先的 GPU `ConstMatrixView` 输入和 caller-owned `MatrixView`
输出，同步和 `*Async` 形式都直接复用设备 kernel，输出形状按上表确定。输入、输出不得重叠；
可直接传入 `ResidentMatrix::view<Device::GPU>()`，无需构造旧矩阵或复制到主机。

归约没有延迟业务状态，也没有 `ReductionWorkspace::checkStatus()`。同步重载等待 stream；
异步重载只入队。allocating async 返回值使用 stream-ordered 分配，输入、输出和 workspace
必须存活到 stream 完成，并在销毁所属 stream 前显式关闭这些异步分配。

### 有限行掩码和列范围

```cpp
auto mask = finiteRowMask(input);        // input.rows() x 1, uint8_t
auto bounds = finiteColumnBounds(input); // minimum/maximum: 1 x C, validRowCount: 1 x 1
auto combined = finiteColumnBoundsWithMask(input); // 同时返回 rowMask 和 bounds
```

`finiteRowMask` 仅当一行的全部元素都是有限值时把该行标为 `1`；NaN、`+Inf` 或 `-Inf`
都会使整行无效。`finiteColumnBounds` 跳过这些无效行，对剩余行逐列计算最小值和最大值，
并返回参与统计的有效行数。没有有效行时，每列上下界都是 NaN，计数为零；`N x 0`
输入的每一行按空集约定视为有效，上下界形状为 `1 x 0`，计数为 `N`。
需要同时使用掩码和边界时，`finiteColumnBoundsWithMask` 从同一次掩码计算返回两者，避免重复
扫描输入；返回字段为 `rowMask`、`minimum`、`maximum` 和 `validRowCount`。

CPU 与 CUDA 接口均支持 `float` 和 `double`。CUDA 接口提供同步 allocating、同步输出复用、
`*Async` allocating 和异步输出复用重载；它们复用 `ReductionWorkspace`，生命周期、stream
绑定和 stream-ordered 分配规则与其他归约一致。异步重载只把计算加入指定 stream，不执行
隐式同步。

CUDA 有限值统计也接受调用方拥有的连续 `ConstMatrixView` 输入和 `MatrixView` 输出，
可直接使用 `ResidentMatrix::view<Device::GPU>()`，无需构造旧的 `DenseStorage`。
输出形状分别为 `N×1` 掩码、`1×C` 最小/最大值和 `1×1` 有效行数，输入输出必须两两不重叠。
同步及 `*Async` 重载共用同一套 kernel；无 CUDA 构建调用这些设备接口会明确报错。

## 索引、扫描和稳定紧缩

GPU `gatherRows[Async]`、`scatterRows[Async]` 和 `compactRows[Async]` 支持连续的调用方拥有
视图，可直接借用 `ResidentMatrix` 存储；输出不能与输入或其它输出重叠。紧缩输出保持
`N×C` 和 `N×1` 容量，并把有效行数写入 `1×1` 的 `Index` 视图；有效行是每列的前缀，
列步长仍为 `N`。如需精确大小，可在读取计数后进行设备内二维拷贝。
同步入口等待并检查索引状态，异步入口仍要求调用方同步后执行 `workspace.checkStatus()`。

```cpp
auto offsets = exclusiveScan(counts);
exclusiveScanAsync(counts_view, offsets_view, workspace, stream); // 外部 GPU 缓冲区
auto rows = gatherRows(input, indices);       // indices: K x 1
scatterRows(values, indices, output);         // values: K x C
auto exact = compactRows(input, keep_mask);   // keep_mask: input.rows() x 1, uint8_t
```

- `exclusiveScan` 保持输入形状，按底层列优先线性顺序执行 exclusive scan；负 count 合法，
  任一前缀加法超出 `Index` 范围时抛 `std::overflow_error`
- GPU caller-owned 重载接受连续的 `ConstMatrixView<Index, Device::GPU>` 输入和
  `MatrixView<Index, Device::GPU>` 输出，两者必须是相同的 `N x 1` 形状。它们可直接包装外部
  设备 byte buffer，并沿用 `IndexingWorkspace` 的非默认 stream 绑定、状态批次和最低溢出 offset
  语义；异步调用完成后仍需先同步 stream，再调用 `workspace.checkStatus()`。输入和输出不得
  重叠，包括完全同址的原地扫描，因为溢出检查需要读取原始 counts
- `gatherRows` 按 `indices` 顺序生成 `K x input.cols()`，保留重复索引
- `scatterRows` 不清空未命中的目标行；多个源行写同一目标时，最低 source row 获胜。
  CPU 和同步 GPU 重载在发现非法索引时保证输出不变
- `compactRows` 把任意非零 mask 当作 keep，按 source row 升序稳定输出，并返回对应的
  `sourceIndices`
- `gatherRows`、`scatterRows` 和 `compactRows` 的 CPU/CUDA 接口支持 `float`、`double`、
  `Index`、`std::uint8_t` 和 `std::uint16_t` 元素；顺序、重复 scatter 和越界语义在各类型间一致

GPU 的 capacity 重载要求 `capacity_output` 为 `R x C`、`capacity_source_indices` 为
`R x 1`、`selected_count` 为 `1 x 1`，其中 `R=input.rows()`、`C=input.cols()`。只有前
`selected_count` 行有效，容量尾部不属于结果。同步 exact wrapper 先写容量输出，再根据
count 分配精确的 `selected_count x C` 和 `selected_count x 1` 结果；异步 API 不提供
exact-size allocating compact，因为精确行数在设备工作完成前未知。

`gatherRows/scatterRows/compactRows` 同样支持连续列优先 GPU view 的同步/异步重载。compact
的 view 接口使用上述容量形状，只有每列前 `selected_count` 行有效，列间 stride 仍为 `R`。
所有可写 view 必须与其他输入、输出分离；非法形状、stride 或重叠在 launch 前抛出。

形状错误在 launch 前抛出。GPU scan overflow 和 gather/scatter 越界由
`IndexingWorkspace` 聚合：同步重载内部等待并检查；异步调用方必须先同步所属 stream，再
调用 `workspace.checkStatus(operation)`。状态 batch 保留最低 source offset，可跨同一 stream
上的连续调用累积；overflow 和越界同时存在时 overflow 优先。`checkStatus()` 会消费状态，
即使它抛出异常；未消费状态会阻止 `reserveBytes()` reset 和 `closeAsyncAllocation()`。

`IndexingWorkspace` 是 move-only、grow-only、非线程安全，并绑定异步复用 stream。它的
move assignment 为 `noexcept`：目标现有存储和尚未消费的目标状态会被释放/丢弃，再接管源
workspace 的存储、状态和 stream provenance。因此调用方必须在覆盖目标 workspace 前同步并
消费其状态。

## 键值分组与分段归约

```cpp
auto sorted = sortByKey(keys, values);
auto runs = runLengthEncode(sorted.keys);
auto grouped = reduceByKey(sorted.keys, sorted.values, GroupReduction::Sum);
auto totals = segmentedReduce(values, offsets, GroupReduction::Sum);

Int32Key3 key{x, y, z};
auto voxel_order = sortByKey(keys3, source_indices); // source_indices: int32_t 或 Index
auto voxel_runs = runLengthEncode(voxel_order.keys);
```

- 所有输入都是列向量。标量键支持 `std::uint32_t`、`std::uint64_t` 和 `Index`；值支持
  `float`、`double` 和 `Index`
- `Int32Key3` 是公开的三分量有符号整数复合键，字段为 `x/y/z`。它是 standard-layout、
  trivially-copyable 的固定 12 字节类型，按 `x`、`y`、`z` 字典序比较，三个分量均保留完整
  `int32_t` 范围，不进行有碰撞风险的 64 位压缩
- `Int32Key3` 当前支持稳定 `sortByKey` 和 `runLengthEncode`；排序 payload 支持
  `std::int32_t` 与 `Index`，RLE 的 `counts/runCount` 使用 `Index`
- `sortByKey` 是稳定升序排序，因此相同键保持原输入顺序。CUDA 后端使用稳定的 CUB radix sort
- `runLengthEncode` 编码连续 run，返回键、长度和 run 数；需要全局按键合并时先调用
  `sortByKey`
- `reduceByKey` 对连续相同键执行 `Sum`、`Minimum` 或 `Maximum`；浮点求和在同一后端和设备
  配置下可复现，但 CPU 与 CUDA 的归约树不同，不承诺跨后端 bitwise 相同
- `segmentedReduce` 使用 `(S + 1) x 1` offsets 描述 `S` 个半开区间。offsets 必须从零开始、
  单调不减，并以 `values.rows()` 结束。空 segment 的 sum/min/max 分别返回 `0`、类型最大值和
  类型最小值。CPU 会验证 offsets；CUDA 接口把有效设备 offsets 作为调用前置条件

同步 CUDA 的 RLE 和 reduce-by-key 返回精确尺寸结果。异步版本无法在不等待设备的情况下知道
run 数，因此返回与输入等长的容量数组，只有前 `runCount[0]` 项有效。所有 CUDA 操作还提供
输出复用重载，并共享 `GroupingWorkspace`；它是 `ReductionWorkspace` 的语义别名，遵循相同的
grow-only、单 stream 复用和 stream-ordered 关闭规则。`*Async` 只入队，不隐式同步。

标量键的 `sortByKey/runLengthEncode` 和 `segmentedReduce` 也支持 caller-owned GPU view
同步/异步重载，类型范围与上文一致。所有 view 必须是连续列优先列向量，可写输出不能与输入
或其他输出重叠。RLE 的 keys/counts 输出容量均为 `N x 1`，runCount 为 `1 x 1`；分段归约
输出为 `S x 1`。这些重载直接复用 CUB 和设备 kernel，不创建旧矩阵存储。

CUDA 的 `Int32Key3` 排序在 PlaMatrix 内按 `z -> y -> x` 执行三个稳定的 signed-int32 radix pass，
最终结果等价于 `(x, y, z)` 字典序；CUB/临时 permutation 均封装在 `GroupingWorkspace` 内。专用的
caller-owned 重载接受连续 `ConstMatrixView<..., Device::GPU>` 输入和
`MatrixView<..., Device::GPU>` 输出，因此外部设备 byte buffer 可直接包装为 `N x 1` 视图：

```cpp
auto key_input = makeColumnMajorView<Int32Key3, Device::GPU>(device_keys, count, 1);
auto index_input = makeColumnMajorView<std::int32_t, Device::GPU>(device_indices, count, 1);
auto key_output = makeColumnMajorView<Int32Key3, Device::GPU>(sorted_keys, count, 1);
auto index_output = makeColumnMajorView<std::int32_t, Device::GPU>(sorted_indices, count, 1);

sortByKeyAsync(key_input, index_input, key_output, index_output, workspace, stream);
runLengthEncodeAsync(ConstMatrixView<Int32Key3, Device::GPU>(key_output),
                     unique_keys, counts, run_count, workspace, stream);
```

这些 view 不拥有内存；输入、输出和 workspace 必须保持有效到 stream 完成。view 必须是连续列优先
的 `N x 1`，RLE 的 unique/count 容量至少为 `N x 1`，runCount 为 `1 x 1`。

## SVD 分解

```cpp
/// A = U * diag(S) * Vt
/// 返回 (U, S_vector, Vt)，奇异值降序排列
template <typename Scalar, Device Dev>
std::tuple<DenseStorage<Scalar, Dev>,
           DenseStorage<Scalar, Dev>,
           DenseStorage<Scalar, Dev>> svd(const DenseStorage<Scalar, Dev>& A);
```

- **CPU**：使用项目内 one-sided Jacobi SVD；64 列以上使用确定性的 round-robin 独立列对并行和两遍正交化补全基
- **GPU**：`cusolverDnSgesvd` (float) / `cusolverDnDgesvd` (double)，legacy cuSOLVER 路径要求 `rows >= cols`
- 返回形状为 `U(m,m)`, `S(min(m,n),1)`, `Vt(n,n)`。

Eigen 风格 `Matrix::jacobiSvd()`/`bdcSvd()` 的自动后端还提供 OpenCL 与 Vulkan 实现。设备端按
round-robin 将互不相交的列对分组，每轮并行执行 one-sided Jacobi 旋转；OpenCL 支持
`float`/`double`，Vulkan 支持 `float`。奇异值排序、退化列归一化和 full-U 基补全在设备结果回传后完成，
主导的 `O(mn²)` 旋转留在 GPU。float 固定执行 8 个 sweep，OpenCL double 执行 12 个 sweep。

## QR 分解

```cpp
/// A = Q * R (Q 正交, R 上三角)
template <typename Scalar, Device Dev>
std::tuple<DenseStorage<Scalar, Dev>, DenseStorage<Scalar, Dev>>
qr(const DenseStorage<Scalar, Dev>& A);
```

- **CPU**：尺度安全的 Householder 反射；大矩阵的尾随列更新和 Q 回放按互不重叠的列/行确定性并行
- **GPU**：`cusolverDnSgeqrf` + `cusolverDnSorgqr`

Eigen 风格 `Matrix::householderQr()` 还可在 OpenCL `float`/`double` 与 Vulkan `float` 上执行。
反射向量、尾随列更新、显式 Q 回放及秩统计都在设备端完成；宽矩阵和秩亏矩阵沿用相同入口。

## 对称特征值

```cpp
/// 对称矩阵特征值分解，返回特征值向量 (降序)
template <typename Scalar, Device Dev>
DenseStorage<Scalar, Dev> eigh(const DenseStorage<Scalar, Dev>& A);
```

- **CPU**：使用 Householder 对称三对角化加隐式 QL，并按降序返回特征值
- **GPU**：`cusolverDnSsyevd` (分治算法)

Eigen 风格 `SelfAdjointEigenSolver` 在 OpenCL `float`/`double` 和 Vulkan `float` 上使用两侧并行
Jacobi：每个 round 的旋转列对互不相交，列更新、行更新和特征向量累积通过设备内存屏障排序。
结果按 Eigen 语义升序排列；一般矩阵、广义特征、Schur 与 QZ 仍由 CPU 执行。

### 批量对称 3x3 特征分解

```cpp
auto result = symmetricEigh3x3Batched(compact); // compact: N x 6
// result.eigenvalues:  N x 3
// result.eigenvectors: N x 9
```

单矩阵热路径可使用不分配 `DenseStorage` 的固定尺寸接口：

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

GPU 同步/异步输出复用还接受 `ConstMatrixView` 输入和两个 `MatrixView` 输出，直接复用
相同 kernel 与状态机制；要求连续列优先，输入与两个输出两两不重叠。可直接借用
`ResidentMatrix::view<Device::GPU>()`，不需要构造旧矩阵。

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
DenseStorage<Scalar, Dev> solve(
    const DenseStorage<Scalar, Dev>& A,
    const DenseStorage<Scalar, Dev>& B);
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
