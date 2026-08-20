# PlaMatrix 代码架构文档

本文档面向希望深入理解库内部实现的开发者，按模块逐一说明算法选择、数据流、内存管理和关键设计决策。

---

## 1. 整体架构

```
include/plamatrix/plamatrix.h          # 总入口
├── core/         基础类型 + 内存管理
│   ├── types.h          Device 枚举, Index 类型
│   ├── error_check.h    CUDA/cuBLAS/cuSOLVER 错误宏
│   ├── no_cuda_stubs.h  无 CUDA 时的桩实现
│   ├── allocator.h      CpuAllocator / GpuAllocator
│   └── device_matrix.h  RAII 矩阵基类
├── dense/        密集矩阵
│   ├── dense_matrix.h   DenseMatrix (列优先, move-only)
│   ├── dense_ops.h      基础逐元素加减
│   └── elementwise.h    标量/逐元素乘除、abs/sqrt/clamp
├── sparse/       稀疏矩阵
│   ├── coo_matrix.h     COO 格式 + toCsr()
│   └── csr_matrix.h     CSR 格式 (三数组)
├── ops/          运算层
    ├── gemm.h            矩阵乘法 (原生 CPU/cuBLAS)
    ├── decomposition.h   SVD / QR / Eigh
    ├── reduction.h       按轴 value/indexed reduction + workspace
    ├── indexing.h        scan、gather/scatter、stable compact + workspace
    ├── small_matrix.h    固定阶小系统求解、批量对称 3x3 特征分解 + workspace
    ├── statistics.h      忽略非有限样本的稳健中位数
    ├── solver.h          线性求解
    ├── vector.h          Vec3 小向量表示与算术
    └── point_cloud.h     旋转矩阵, 刚体变换, 协方差
└── optimization/ 非线性最小二乘基础设施
    ├── robust_loss.h          Huber 块损失与 IRLS 权重
    ├── block_schur.h          二分块法方程与 Schur-PCG
    └── levenberg_marquardt.h  LM 阻尼状态策略

src/                            实现文件
├── dense/*.cu + *_cpu.cpp      密集矩阵 CPU/CUDA 运算
├── ops/*_dispatch.cu           reduction/indexing/small-matrix GPU 调度
├── ops/*_workspace.cu          grow-only workspace 与 stream/status 生命周期
├── ops/*.cu + *_cpu.cpp        其他 GPU + CPU 运算实现
├── sparse/csr_matrix.cpp       稀疏矩阵模板实例化
└── optimization/*              多 primary 块累计、CPU 矩阵自由、可复用 CSR 拓扑、设备数值装配及块 Jacobi Schur-PCG
```

**核心设计原则**：
- `DeviceMatrix<Scalar, Device>` 作为 RAII 基类，编译期 `if constexpr` 分发 CPU/GPU 代码
- 列优先 (column-major) 存储，兼容 cuBLAS Fortran 序
- 显式设备管理：`toCpu()` / `toGpu()` 触发 `cudaMemcpy`
- GPU 加速通过 cuBLAS / cuSOLVER 库 + 少量自定义 kernel
- CPU 后端只使用自包含原生算法；GEMM 使用 B-panel packing、寄存器微内核、AVX2/FMA 运行时分派和 OpenMP

---

## 2. 核心模块

### 2.1 类型系统 (`core/types.h`)

```cpp
enum class Device : int { CPU = 0, GPU = 1 };
using Index = std::int64_t;
```

`Device` 作为模板参数实现编译期设备分发。选择 `int` 作为底层类型以便在 `enum class` 上用 `if constexpr`。`Index` 使用 64 位有符号整数，支持千万级点云的索引。

### 2.2 错误处理 (`core/error_check.h`)

三个检查宏，每个捕获 `__FILE__`、`__LINE__` 和字符串化的表达式：

| 宏 | 检查对象 | 失败行为 |
|----|---------|---------|
| `PLAMATRIX_CHECK_CUDA(call)` | `cudaError_t` | 提取 `cudaGetErrorString()` → `throw runtime_error` |
| `PLAMATRIX_CHECK_CUBLAS(call)` | `cublasStatus_t` | 状态码 → 可读名称 + 数字 → 抛异常 |
| `PLAMATRIX_CHECK_CUSOLVER(call)` | `cusolverStatus_t` | 同上，含 `ZERO_PIVOT` 等 12 种状态 |

`cublasStatusString()` / `cusolverStatusString()` 对全部状态码做 exhaustive switch，输出如 `CUBLAS_STATUS_EXECUTION_FAILED (13)`。

### 2.3 无 CUDA 时的桩实现 (`core/no_cuda_stubs.h`)

当 `-DPLAMATRIX_WITH_CUDA=OFF`，`PLAMATRIX_NO_CUDA=1` 被定义，此文件在 `error_check.h` 中替换 `<cuda_runtime.h>` 等头文件：

| 真实 API | 桩实现 |
|----------|--------|
| `cudaMalloc(&ptr, size)` | `ptr = malloc(size)` |
| `cudaFree(ptr)` | `free(ptr)` |
| `cudaMemcpy(dst, src, n, dir)` | `memcpy(dst, src, n)` |
| `cudaMemset(ptr, v, n)` | `memset(ptr, v, n)` |
| `cublasCreate/Destroy` | 返回 `CUBLAS_STATUS_SUCCESS` |
| `cusolverDnCreate/Destroy` | 返回 `CUSOLVER_STATUS_SUCCESS` |

无 CUDA 构建下，`Device::GPU` 矩阵的存储和传输桩使用 CPU 内存，以便公共头文件和 CPU-only 测试可编译。`.cu` 中的 GPU 算法不会构建；真实业务路径应使用 `Device::CPU`，需要 GPU 加速时重新启用 CUDA 构建。

### 2.4 内存分配器 (`core/allocator.h`)

```cpp
template <typename Scalar>
struct CpuAllocator {
    static Scalar* allocate(size_t count) {
        posix_memalign(&ptr, 32, count * sizeof(Scalar));  // 32字节对齐
    }
    static void deallocate(Scalar* ptr) { free(ptr); }
};

template <typename Scalar>
struct GpuAllocator {
    static Scalar* allocate(size_t count) { cudaMalloc(&ptr, ...); }
    static void deallocate(Scalar* ptr) { cudaFree(ptr); }
    static Scalar* allocateAsync(size_t count, cudaStream_t stream);
    static void deallocateAsync(Scalar* ptr, cudaStream_t stream);
};
```

CPU 分配器使用 32 字节对齐（`posix_memalign`），适配 AVX-256 向量化。GPU 分配器包装 `cudaMalloc`/`cudaFree`，无 CUDA 时通过桩回退到 `malloc`/`free`。

此外还有三条互不混用的分配路径：

- `PinnedCpuAllocator` 使用 `cudaHostAlloc/cudaFreeHost`，供真正异步的 host-device 传输
- 可选进程内 GPU memory pool 只缓存普通 `cudaMalloc` block，按字节数复用
- `allocateAsync/deallocateAsync` 直接使用 `cudaMallocAsync/cudaFreeAsync`，由 owner 保留创建
  stream provenance，不进入普通 memory pool；CPU-only 构建明确拒绝该入口

### 2.5 矩阵基类 (`core/device_matrix.h`)

```cpp
template <typename Scalar, Device Dev>
class DeviceMatrix {
protected:
    Index _rows, _cols;
    Scalar* _data;

    void allocate(Index count) {
        if constexpr (Dev == CPU) _data = CpuAllocator<Scalar>::allocate(count);
        else                     _data = GpuAllocator<Scalar>::allocate(count);
    }
    void release() {
        if (_data) {
            if constexpr (Dev == CPU) CpuAllocator<Scalar>::deallocate(_data);
            else                     GpuAllocator<Scalar>::deallocate(_data);
            _data = nullptr;  // 二次释放安全
        }
    }
};
```

**关键设计决策**：
- **空对象有效**：`DenseMatrix()` 为 `0 x 0`，零元素矩阵不分配存储
- **禁止拷贝**：大矩阵拷贝昂贵且隐式，强制用户显式操作
- **支持移动**：move 构造/赋值转移指针、host/GPU allocation kind 和 async stream
  provenance；源矩阵置为 `0 x 0`
- **RAII**：析构自动释放，`release()` 置 nullptr 防止二次释放
- **`if constexpr` 编译期分发**：CPU/GPU 代码路径在编译期确定，零运行时开销

普通 GPU allocation 走同步释放或 memory pool；stream-ordered allocation 必须在其创建
stream 上入队释放。`closeAsyncAllocation()` 是可报告错误的显式路径，成功后清空 owner；
析构为 `noexcept` 后备路径。移动赋值会先释放目标原有 owner，再接管源 owner，因此调用方
必须保证目标和源关联的异步工作、stream 都满足各自生命周期约束。

---

## 3. 密集矩阵模块

### 3.1 DenseMatrix (`dense/dense_matrix.h`)

继承 `DeviceMatrix`，增加：

**构造和初始化**：
- 参数化构造后立即零初始化：CPU 用 `memset`，GPU 用 `cudaMemset`
- `uninitializedAsync(rows, cols, stream)`：GPU-only 的 stream-ordered 未初始化分配
- `fill()`：CPU 用 `fill_n`；GPU 零值用 `cudaMemset`，非零值启动自定义 kernel

**元素访问**：
- `operator()(row, col)`：CPU only，`static_assert` 阻止 GPU 调用。列优先索引公式 `data[row + col * rows]`
- `setValue(row, col, val)` / `getValue(row, col)`：双设备通用。GPU 路径通过单元素 `cudaMemcpy` 实现

**设备传输**：
- `toGpu()`：`cudaMemcpyHostToDevice`，`static_assert(Dev == CPU)`
- `toCpu()`：`cudaMemcpyDeviceToHost`，`static_assert(Dev == GPU)`
- `copyToGpuAsync/copyToCpuAsync` 和 allocating async 重载只建立 stream ordering，不等待
- 返回新矩阵（move），传输是一次性的、昂贵的、显式的

**转置**：
- CPU：双重 `for` 循环
- GPU：2D kernel，`dst[j + i*src_cols] = src[i + j*src_rows]`

### 3.2 GPU Kernel 启动配置

| 运算 | Kernel | block 大小 | grid 大小 |
|------|--------|-----------|-----------|
| fill | `fillKernel` | 256 | `ceil(N/256)` |
| transpose | `transposeKernel` | `dim3(16,16)` | `ceil(rows/16), ceil(cols/16)` |
| add/sub | `elementWiseAdd/SubKernel` | 256 | `ceil(N/256)` |
| point transform | `transformPointsKernel` | 256 | `ceil(N/256)` |

统一的 256 线程/block 保持 CUDA occupancy 最优化。transpose 使用 2D block 处理行/列索引映射。

### 3.3 逐元素运算 (`dense/dense_ops.h`、`dense/elementwise.h`)

CPU 路径对小矩阵使用串行循环，超过内部阈值后用 `#pragma omp parallel for` 对平坦数组做循环并行化：
```cpp
for (Index i = 0; i < n; ++i)
    C.data()[i] = A.data()[i] + B.data()[i];
```

GPU 路径使用自定义 kernel（见上表），256 线程/block，CEIL(n, 256) 个 block。

`add/sub/hadamardMultiply/hadamardDivide` 先检查两个输入的 `rows/cols` 完全相同；输出复用
重载也检查同形，内部没有 broadcast plan。标量 multiply/add/divide、abs、sqrt 和 clamp
同样遍历列优先平坦存储。除 `scalarDivide` 拒绝零除数、clamp 拒绝逆区间外，CPU 和 CUDA
都保留 IEEE 浮点行为：Hadamard 零除产生 infinity/NaN，负数 sqrt 产生 NaN，输入 NaN 不被
静默替换。同步 GPU 包装器在返回前同步 stream，异步包装器只负责 launch 和即时 CUDA 错误。

### 3.4 归约 (`ops/reduction*.{h,cpp,cu}`)

调度层先把 `ReductionAxis` 转换成 lane plan：

| Axis | lane 数 | lane 长度 | 输出 |
|------|---------|-----------|------|
| `All` | 1 | `rows*cols` | `1 x 1` |
| `Rows` | `rows` | `cols` | `rows x 1` |
| `Columns` | `cols` | `rows` | `1 x cols` |

`All` 的 source offset 是列优先线性 offset；Rows/Columns 的 indexed reduction 返回被归约
维度内的 offset。extreme reducer 将 NaN 视为传播值，并在相等值或多个 NaN 间保留最低
offset。sum 允许空 lane 并写零；其他归约只在输出 lane 本身存在且长度为零时拒绝。

CPU `float` sum/mean 使用 `double` accumulator。mean 先扫描 NaN/infinity 和最大有限绝对值；
普通累加存在溢出或舍入风险时，切换到按 scale 归一化的补偿求和，再做除法和最终类型转换。
CUDA `All` 使用 CUB reduction 加自定义 summary/store kernel，Rows/Columns 使用每 lane kernel，
保留相同的特殊值和稳定 mean 规则。

`ReductionWorkspace` 保存 CUB temporary storage，是 move-only、grow-only、非线程安全对象。
普通 allocation 可在同步点通过 `reserveBytes()` 增长或解除 stream 绑定；异步 allocation 只能
在创建 stream 上增长。不同 stream 复用必须先同步，再 reset 普通 allocation 或 close async
allocation。归约没有设备业务 status，因此没有 `checkStatus()`；同步包装器只需同步 stream，
异步调用方负责输出和 workspace 生命周期。

allocating `*Async` reduction 输出使用 `DenseMatrix::uninitializedAsync`。value reduction 为一个
stream-ordered owner，arg reduction 为 values/indices 两个 owner；它们必须在所属 stream
销毁前分别关闭。workspace move assignment 为 `noexcept`，会释放目标 storage 并转移源的
storage 和 stream provenance。

### 3.5 索引与紧缩 (`ops/indexing*.{h,cpp,cu}`)

CPU scan 直接沿平坦列优先数组累计，并在每次加法前检查正负 `Index` 溢出。GPU 使用 CUB
exclusive sum，然后用 validation kernel 检查每个 source offset 处的加法；错误记录最低
offset。gather 保持索引顺序和重复项。scatter 先验证完整 index vector，再决定 owner：CPU
对 `(destination, source)` 排序，GPU 对每个 destination 原子选择最低 source row，因此结果
确定且非法索引不会造成部分写入。

stable compact 把非零 byte mask 交给 source-row counting iterator 和 CUB `DeviceSelect::Flagged`。
异步 capacity 形式写 `R x C` values、`R x 1` source indices 和 `1 x 1` count，只有 count 指定
的前缀有效。同步 exact wrapper 等待 count，再分配精确尺寸并用 device-to-device copy/2D copy
移除 capacity pitch；这也是没有 allocating async exact wrapper 的原因。

`IndexingWorkspace` 头部保存 `{overflow, out_of_range}` status batch，后面是对齐的 CUB
temporary storage 和 scatter owner 数组。连续同-stream async 调用共享一个未消费 batch，后续
成功调用不会清掉先前错误；每类记录最低 source offset，同时存在时 overflow 优先。
`checkStatus()` 要求 stream 已完成，复制并消费 batch，然后返回或抛异常。未消费 batch 禁止
同步 reset 和 async close。同步 API 内部完成 synchronize/check；异步 API 由调用方执行。

workspace 的 storage/stream/status 都由 move 构造转移。move assignment 是 `noexcept`，会释放
目标 storage（并丢弃目标未消费 status）后接管源状态；因此覆盖目标前必须显式消费目标状态。

### 3.6 批量对称 3x3 特征分解 (`ops/small_matrix*.{h,cpp,cu}`)

输入每行为 `[xx,xy,xz,yy,yz,zz]`，输出为升序 `N x 3` eigenvalues 和按
`[v0x,v0y,v0z,v1x,...,v2z]` 打包的 `N x 9` eigenvectors。CPU 在分配输出前检查完整输入
为 `N x 6` 且全有限；GPU 在 launch 前检查形状，非有限值由设备端按行检测。

CPU/GPU 都使用固定 8 sweep cyclic Jacobi，每 sweep 依次旋转 `(0,1)`、`(0,2)`、`(1,2)`。
旋转角先按 `max(abs(app),abs(aqq),abs(apq))` 缩放，降低极值溢出风险。特征值 stable sort 后，
相差不超过 `256*epsilon*max(1,abs(a),abs(b))` 的组视为重复特征空间；实现把固定坐标轴投影
到该空间，执行两遍正交化，构造确定性基。最终向量归一化并规范符号：最大绝对分量非负，
并列选择最低 component index。

GPU kernel 对 nonfinite 或 basis failure 行写零，并分别记录最低失败 row。workspace status 中
nonfinite 优先；同步包装器等待并检查，异步包装器要求调用方同步后 `checkStatus()`。
`SymmetricEigh3x3Workspace` 的增长、stream 绑定和未消费 status 规则与 indexing workspace
一致，但 move assignment 有意允许抛异常：若目标仍有未消费 status，则保持源和目标不变；
消费后才释放目标并转移源 storage/status/stream provenance。

同一头文件还提供栈上 `solveSmallLinearSystem<Scalar, N>`。求解前按行均衡，
再执行部分主元高斯消元，适合 BA 等频繁求解且旋转/平移量纲差异明显的
3x3/6x6 法方程；奇异、非有限输入或非有限解返回 `false`。

### 3.7 稳健统计 (`ops/statistics.h`)

`finiteMedian` 忽略 NaN 和正负无穷，使用 `nth_element` 原地选择中位数。
偶数样本采用防溢出的均值计算；没有有限样本时返回 `std::nullopt`。

---

## 4. 矩阵乘法 (GEMM)

### 4.1 CPU 实现 (`gemm_cpu.cpp`)

CPU 始终使用项目内原生内核。内核把 B 按连续
4 列 panel 打包，每个输出 tile 只写自己的 C 区域：

内层微内核保持 A、C 的列内访问连续：
```cpp
packFourColumns(B, panel);
#pragma omp parallel for schedule(static)
for (row_tile = 0; row_tile < row_tile_count; ++row_tile)
    microkernel(A + row_begin, panel, C + row_begin, rows, k);
```

- B panel 的四列连续存储，避免通用 k-major packing 在大矩阵上的跨列 stride
- 标量微内核同时累计四个 C 列；x86 AVX2/FMA 内核对 float 一次处理 8 行、double 一次处理 4 行
- 启动时运行时检测 AVX2、FMA 和 OS 保存 YMM 状态；不支持时保留完全相同 API 的标量路径
- 线程数按输出工作量和 tile 数自动限制，静态分片避免竞争写入和线程过量
- 列优先索引：`A[i + p*m]`，`B[p + j*k]`，`C[i + j*m]`

### 4.2 GPU 实现 (`gemm.cu`)

使用 cuBLAS，关键调用参数：
```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,            // 1.0
    A.data(), lda=m,   // A 的 leading dimension = m (行数)
    B.data(), ldb=k,   // B 的 leading dimension = k
    &beta,             // 0.0
    C.data(), ldc=m);
```

**为什么都是 `CUBLAS_OP_N`**：cuBLAS 假定列优先存储。当我们的矩阵也是列优先时，无需转置。`lda=m` 是因为列优先下 leading dimension 等于行数。

**Handle 管理**：函数级 `static cublasHandle_t`，懒初始化（C++11 保证线程安全的 static 初始化）。整个进程生命周期内复用一个 handle。支持可选 `cudaStream_t` 参数，调用 `cublasSetStream` 设定，并在返回前同步该 stream，避免调用方立即传回 CPU 时读到未完成结果。

---

## 5. 矩阵分解

### 5.1 SVD

#### CPU：原生 one-sided Jacobi (`decomposition_cpu.cpp`)

CPU 返回完整 `U(m,m)`、紧凑 `S(min(m,n),1)` 和完整 `Vt(n,n)`：

**算法**：one-sided Jacobi（在 A 的列上直接工作，不显式构造 `A^T A`）：

1. 初始化：`U = A`，`Vt = I`，`S = 0`
2. Jacobi 扫描循环（最多 100 轮）：
   - 对每对列 `(j1, j2)` 计算 `a = ||col_j1||²`，`b = ||col_j2||²`，`c = col_j1 · col_j2`
   - 收敛判断：归一化列相关性 `|c| / sqrt(a*b)` 不超过 `8 * epsilon(Scalar)` 时跳过
   - 计算 Givens 旋转参数：`tau = (a-b)/(2c)`，`t = sign(tau)/(|tau|+sqrt(1+tau²))`，`cs = 1/sqrt(1+t²)`，`sn = cs*t`
   - 同时旋转工作矩阵的两列和 Vt 的两行
3. 提取奇异值：`S[j] = ||U_col_j||`
4. 归一化 U 的列
5. 按奇异值降序排列

64 列以上使用 round-robin 调度：一轮中的列对互不相交，可静态并行；轮次顺序、列对顺序和每个点积顺序固定。
完整 U 的零空间基使用两遍 modified Gram-Schmidt，避免高瘦矩阵的正交性损失。3×3 热路径另有
allocation-free `svd3x3()`，不进入动态矩阵分解。

#### GPU：cuSOLVER gesvd (`decomposition.cu`)

```cpp
cusolverDnSgesvd(handle, 'A', 'A', m, n, A_copy, lda, S, U, ldu, Vt, ldvt,
                 d_work, lwork, d_rwork, d_dev_info);
```

经典的四步 cuSOLVER 模式：
1. `gesvd_bufferSize` 查询工作空间大小
2. `cudaMalloc` 分配 `d_work`、`d_rwork`、`d_dev_info`
3. 调用 `gesvd`
4. 从 `d_dev_info` 读取错误码 → 释放资源 → 检查错误

注意：gesvd 会覆写输入矩阵 A，所以先 `cudaMemcpyDeviceToDevice` 复制一份。

### 5.2 QR 分解

#### CPU：Householder 反射 (`decomposition_cpu.cpp`)

对每列 k：
1. 提取子列 `x = R(k:m, k)`，计算 `norm_x`
2. 构造 Householder 向量：`alpha = -sign(x0) * norm_x`，`v0 = x0 - alpha`，`tau[k] = (alpha - x0)/alpha`
3. 存储反射向量于 R 对角线下
4. 对每个后续列 j，计算 `dot = v^T · R(k:m, j)`，更新 `R(k:m, j) -= tau * dot * v`

范数使用 `hypot` 缩放累积，阈值随矩阵尺度变化。大矩阵的尾随列变换和从 R 恢复 Q 时的行变换
按互不重叠的输出区域静态并行。

**时间复杂度**：O(m n min(m,n)) = O(n³) 对于方阵。

#### GPU：cuSOLVER geqrf + orgqr (`decomposition.cu`)

- **geqrf**：计算 QR 分解（同 CPU Householder 原理，GPU 优化版本）
- **orgqr**：从反射向量显式构建正交矩阵 Q
- **限制**：当 `m > n`（矩形矩阵，行 > 列），orgqr 以 `k = min(m,n)` 调用，只有前 k 列形成标准正交基。剩余 `m-k` 列保持为单位向量，不是正交完备。这是 cuSOLVER 的行为限制。

### 5.3 对称特征值 (Eigh)

#### CPU：Householder 三对角化 + 隐式 QL (`decomposition_cpu.cpp`)

默认先用尺度安全的 Householder rank-2 对称更新把矩阵约化成三对角，再对对角/次对角数组执行隐式
QL 迭代，最终按降序排序。约化只保留下三角语义并显式维持对称性；QL 使用相对机器精度作为分裂阈值。
固定 3×3 PCA 热路径使用 `symmetricEigh3x3()`，不分配动态工作矩阵。

#### GPU：cuSOLVER syevd (`decomposition.cu`)

```cpp
cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                 n, A_work, lda, eigvals, d_work, lwork, d_dev_info);
```

- `NOVECTOR`：只求特征值，不求特征向量（点云协方差 PCA 场景只需特征值）
- `LOWER`：只引用下三角（矩阵是对称的）
- syevd 使用分治算法 (divide-and-conquer)，比 QR iteration 更快
- **注意**：cuSOLVER 返回特征值 **升序**。代码在 CPU 侧做选择排序翻转为降序，便于 CPU/GPU 结果对照

---

## 6. 线性求解器

### 6.1 CPU：列主元 LU 分解 (`solver_cpu.cpp`)

标准列主元高斯消元（Gaussian elimination with partial pivoting）：

**前向消元** (对每列 k)：
1. 寻找主元：`pivot_row = argmax_{i>=k} |A(i,k)|`
2. 奇异判断：`max_val < 1e-15` → 抛 "matrix is singular"
3. 行交换：交换 LU 和右端项 X 的第 k 行和 pivot_row 行（LU 只交换 k 列及之后，左侧已是零）
4. 消元：`factor = A(i,k)/A(k,k)`，存储 L 因子在 A 下三角，消除尾随子矩阵和 RHS

**回代** (从 n-1 到 0)：
```
X(i,j) = (X(i,j) - SUM_{k=i+1}^{n-1} LU(i,k) * X(k,j)) / LU(i,i)
```

**复杂度**：O(n³) 消元 + O(n² × nrhs) 求解。

### 6.2 GPU：cuSOLVER getrf + getrs (`solver.cu`)

```cpp
cusolverDnSgetrf(handle, n, n, A_work, lda, d_work, d_pivot, d_dev_info);
cusolverDnSgetrs(handle, CUBLAS_OP_N, n, nrhs, A_work, lda, d_pivot, B_work, ldb, d_dev_info);
```

- `getrf`：LU 分解，`d_pivot` 存储主元排列，`d_dev_info > 0` 表示 U 奇异
- `getrs`：利用 LU 因子和 pivot 求解 `A*X = B`

输入矩阵 A 和 B 都先复制到工作矩阵（同 SVD 模式：cuSOLVER 会覆写输入）。

---

## 7. 点云运算

### 7.0 小向量数学 (`vector.h`)

`Vec3<Scalar>` 是与 CPU/GPU 后端无关的 header-only 三维向量类型，负责单个坐标、方向和法线的轻量运算。
它支持与 `std::array<Scalar, 3>` 显式转换、分量算术、点积、叉积、平方范数、范数、带阈值归一化和有限性检查。
批量点云仍使用列优先 `DenseMatrix<Scalar, Device>`；`Vec3` 不承担动态存储或设备内存管理。

### 7.1 Rodrigues 旋转矩阵 (`point_cloud_cpu.cpp`)

```cpp
R = I*cos(θ) + K*sin(θ) + (1-cos(θ))*(axis·axis^T)
```
其中 K 是旋转轴的叉乘矩阵。归一化轴后按列优先逐元素填入 3×3 矩阵。CPU 计算再传 GPU（3×3 太小不值得写 kernel）。

### 7.2 刚体变换 (`point_cloud_cpu.cpp`)

复制 3×3 旋转矩阵到 4×4 左上角，设置平移向量在第 4 列（索引 12, 13, 14），`T(3,3)=1`，其余为零（矩阵构造时已零初始化）。

### 7.3 批量点变换

CPU：逐点展开的 3×3 矩阵向量乘 `p' = R*p + t`。

GPU：`transformPointsKernel`，256 block 大小，索引 `idx = blockIdx.x * blockDim.x + threadIdx.x` 处理点 `points[idx]`，同样的展开矩阵向量乘。

### 7.4 协方差矩阵

CPU：
1. 遍历所有点计算质心 `(cx, cy, cz)`
2. 累加外积：`C[i*3+j] += (1/N) * (pi - ci) * (pj - cj)`
3. 返回 3×3 半正定矩阵

GPU：使用两遍 CUDA reduction。第一遍按 block 归约坐标和并计算均值；第二遍按 block 归约中心化乘积并写出 3×3 协方差。`float` 输入使用 `double` 累积，避免大坐标小方差场景下的 raw-moment 抵消误差。

---

## 8. 稀疏矩阵

### 8.1 COO → CSR

CPU 与 CUDA 转换都按 `(row, column)` 排序并合并重复坐标；重复值按输入顺序求和。
CPU 重载接受 `std::vector`，CUDA 同步重载接受 device 列向量。完全异步的
`cooToCsrAsync()` 要求调用方预分配精确合并后 nnz 的输出，完成 stream 后调用
`SparseOpsWorkspace::checkStatus()`；检查成功同时把输出标记为可信 CSR。

### 8.2 CSR 存储与传输

`CSRMatrix` 独立拥有 values、column indices 和 row offsets。同步 `toCpu()/toGpu()`
完成后可立即使用；异步 copy 记录 producer stream，未同步时禁止跨 stream 消费或
覆盖。可变数组指针一旦逸出，固定迭代异步求解所需的结构可信标记会失效，自适应
CG/PCG 会重新验证结构。

### 8.3 稀疏乘法

CPU `spmv/spmm` 使用排序 CSR；CUDA 路径通过 cuSPARSE 执行，并由
`SparseOpsWorkspace` 复用 descriptor 和临时存储。异步调用绑定一个 stream，必须在
同步并 `closeAsyncAllocation()` 后才能切换 stream 或销毁非默认 stream。

### 8.4 CG 与 PCG

`cg()` 和 `pcg()` 接受 SPD CSR、列向量 RHS、调用方持有的初始解和
`IterativeSolverOptions`。CPU 与 CUDA 均报告初始/最终残差、迭代数和收敛状态；
Jacobi-PCG 会拒绝缺失、非正或过小的对角元。CUDA 自适应接口执行主机收敛检查，
`cgFixedIterationsAsync()/pcgFixedIterationsAsync()` 则提交固定轮数并通过显式 finalize
读取报告。

CUDA/OpenCL `blockPcg()` 使用调用方提供的 row-major 逆对角块，适合块法方程和 Schur
约化系统；普通 `pcg()` 的标量 Jacobi 行为保持不变。

`DenseCpu` Schur 后端直接装配 row-major 下三角，不创建 CSR 或完整对称副本。每个 block slot
由唯一线程按固定 term 顺序累计，常见 3×3 eliminated / 9×3 cross 使用固定尺寸内核和 workspace
连续缓冲。小系统走串行左看 Cholesky；128 阶以上使用自动选择的 32/48/64 列 block 和一个持久
OpenMP 并行区，依次执行对角 POTRF、panel TRSM 和 4×4 tiled 下三角 SYRK/GEMM 更新。算法选择
不依赖线程数，因而一线程和多线程保持相同数值顺序。

---

## 9. Handle 生命周期策略

cuBLAS 和 cuSOLVER handle 使用**懒初始化 static 局部变量**模式：

```cpp
cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();
    return handle;
}
```

- C++11 保证 `static` 局部变量初始化线程安全
- 整个进程生命周期复用同一个 handle
- 不做显式 destroy（进程退出时自动清理）
- 三个独立 handle：`gemm.cu`（cuBLAS）、`decomposition.cu`（cuSOLVER）、`solver.cu`（cuSOLVER）

---

## 10. 模板实例化策略

| 文件 | 策略 | 实例化类型 |
|------|------|-----------|
| `gemm_cpu.cpp` | 显式实例化 | `gemm<float/double, CPU>` |
| `gemm.cu` | 显式实例化 | `gemm<float/double, GPU, stream>` |
| `decomposition_cpu.cpp` | 显式实例化 | `svd/qr/eigh<float/double, CPU>` |
| `decomposition.cu` | 显式特化 | `svd/qr/eigh<float/double, GPU>` |
| `elementwise_cpu.cpp/.cu` | 显式实例化 | elementwise × `{float,double}` × CPU/GPU |
| `reduction_cpu.cpp/.cu` | 显式实例化 | value/indexed reduction × `{float,double}` |
| `indexing_cpu.cpp/.cu` | 显式实例化 | row indexing/compact × `{float,double}`；scan 使用 `Index` |
| `small_matrix_cpu.cpp/.cu` | 显式实例化 | batched symmetric 3x3 × `{float,double}` |
| `solver_cpu.cpp` | 显式实例化 | `solve<float/double, CPU>` |
| `solver.cu` | 显式特化 | `solve<float/double, GPU>` |
| `csr_matrix.cpp` | 显式实例化 | `CSRMatrix/COOMatrix` × `{float,double}` × `{CPU,GPU}` |

GPU 版本使用**显式特化**（`template<>`）而非实例化，因为需要覆盖泛型模板中的 `if constexpr (Dev == GPU)` 分支。

---

## 11. 无 CUDA 时的行为

当 `-DPLAMATRIX_WITH_CUDA=OFF`：

- `CMakeLists.txt` 不启用 CUDA 语言，不链接 CUDA 库
- `no_cuda_stubs.h` 提供 CUDA 类型和存储/传输 API 的桩定义
- `Device::GPU` 矩阵存储可编译，但 `.cu` 中的 GPU 算法不参与编译
- 普通 GPU allocation/transfer 桩仍用于公共类型测试；stream-ordered
  `DenseMatrix::uninitializedAsync` 和 allocator async API 明确抛出不可用错误
- cuBLAS/cuSOLVER 调用不参与编译（`.cu` 文件不编译）
- elementwise、reduction、indexing 和 batched 3x3 的 GPU 重载保留 inline runtime stubs，
  错误会标明操作名和 `PLAMATRIX_WITH_CUDA=ON`
- 三类 workspace 的 reserve/checkStatus 在 CPU-only 构建中抛出明确错误；
  `closeAsyncAllocation()` 对空 stub workspace 为 no-op。`ReductionWorkspace` 本身没有
  `checkStatus()`
- GPU benchmark 函数通过 `#ifdef PLAMATRIX_WITH_CUDA` 保护；CPU-only 构建会拒绝 `--mode cuda`
- 业务代码在 CPU-only 构建下应使用 `Device::CPU` 运算路径
- CPU-only 测试仍会运行核心矩阵、分解、稀疏和 no-CUDA 行为回归；GPU 专属用例按构建配置跳过

---

## 12. 基准测试程序

`plamatrix_benchmark` 的结构：

- `main.cpp`：CLI 参数解析（`--mode`, `--size`, `--case`, `--output`, `--list`）
- `benchmark_cases.cpp`：CPU 基准用例（每个运算的 serial + OMP 版本）
- `benchmark_cases.cu`：GPU 基准用例（含传输时间单独测量）
- `report_writer.cpp`：Markdown 报告生成 + 环境信息采集

**计时方法**：`measure(fn, warmup, trials)`：
1. warmup 次热身（不计时）
2. trials 次计时；CUDA 构建在每次计时前后同步设备，CPU-only 构建省略 CUDA 同步
3. 取中位数返回

**尺寸限制**：SVD/QR/Eigh 的 CPU 路径在 N > 256 时自动跳过，防止宽档位 benchmark 被慢速分解主导；可用 `--size tiny --case svd,qr,eigh` 做快速专项回归。

**CUDA 用时**不含数据传输时间（传输单独计时存入 `time_transfer_ms`）。
