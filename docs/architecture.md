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
│   └── dense_ops.h      逐元素加减、标量运算
├── sparse/       稀疏矩阵
│   ├── coo_matrix.h     COO 格式 + toCsr()
│   └── csr_matrix.h     CSR 格式 (三数组)
└── ops/          运算层
    ├── gemm.h            矩阵乘法 (CPU OpenMP / cuBLAS)
    ├── decomposition.h   SVD / QR / Eigh
    ├── solver.h          线性求解
    └── point_cloud.h     旋转矩阵, 刚体变换, 协方差

src/                            实现文件
├── dense/*.cu                  GPU kernel (fill, transpose, add, sub)
├── ops/*.cu + *_cpu.cpp        GPU + CPU 运算实现
└── sparse/csr_matrix.cpp       稀疏矩阵模板实例化
```

**核心设计原则**：
- `DeviceMatrix<Scalar, Device>` 作为 RAII 基类，编译期 `if constexpr` 分发 CPU/GPU 代码
- 列优先 (column-major) 存储，兼容 cuBLAS Fortran 序
- 显式设备管理：`toCpu()` / `toGpu()` 触发 `cudaMemcpy`
- GPU 加速通过 cuBLAS / cuSOLVER 库 + 少量自定义 kernel
- CPU 并行通过 OpenMP

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

`Device::GPU` 矩阵在无 CUDA 时使用 CPU 内存，功能完全可用，只是无 GPU 加速。这是一个关键的可用性决策：让用户代码不加修改即可在无 GPU 环境编译运行。

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
};
```

CPU 分配器使用 32 字节对齐（`posix_memalign`），适配 AVX-256 向量化。GPU 分配器包装 `cudaMalloc`/`cudaFree`，无 CUDA 时通过桩回退到 `malloc`/`free`。

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
- **默认构造删除**：必须指定维度，避免未初始化状态
- **禁止拷贝**：大矩阵拷贝昂贵且隐式，强制用户显式操作
- **支持移动**：完整 move 构造/赋值，源矩阵置空
- **RAII**：析构自动释放，`release()` 置 nullptr 防止二次释放
- **`if constexpr` 编译期分发**：CPU/GPU 代码路径在编译期确定，零运行时开销

---

## 3. 密集矩阵模块

### 3.1 DenseMatrix (`dense/dense_matrix.h`)

继承 `DeviceMatrix`，增加：

**构造和初始化**：
- 参数化构造后立即零初始化：CPU 用 `memset`，GPU 用 `cudaMemset`
- `fill()`：CPU 用 `fill_n`；GPU 零值用 `cudaMemset`，非零值启动自定义 kernel

**元素访问**：
- `operator()(row, col)`：CPU only，`static_assert` 阻止 GPU 调用。列优先索引公式 `data[row + col * rows]`
- `setValue(row, col, val)` / `getValue(row, col)`：双设备通用。GPU 路径通过单元素 `cudaMemcpy` 实现

**设备传输**：
- `toGpu()`：`cudaMemcpyHostToDevice`，`static_assert(Dev == CPU)`
- `toCpu()`：`cudaMemcpyDeviceToHost`，`static_assert(Dev == GPU)`
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

### 3.3 逐元素运算 (`dense/dense_ops.h` + `dense_ops.cu`)

CPU 路径使用 `#pragma omp parallel for` 对平坦数组做循环并行化：
```cpp
for (Index i = 0; i < n; ++i)
    C.data()[i] = A.data()[i] + B.data()[i];
```

GPU 路径使用自定义 kernel（见上表），256 线程/block，CEIL(n, 256) 个 block。

---

## 4. 矩阵乘法 (GEMM)

### 4.1 CPU 实现 (`gemm_cpu.cpp`)

三重循环，列优先访问模式：
```cpp
#pragma omp parallel for collapse(2)
for (j = 0; j < n; ++j)           // C 的列
    for (i = 0; i < m; ++i)       // C 的行
        for (p = 0; p < k; ++p)   // 归约维度
            C[i + j*m] += A[i + p*m] * B[p + j*k];
```

- `collapse(2)` 融合外层两个循环，所有 `(i,j)` 对被并行分配到线程
- 每个线程维护局部累加器 `sum`，避免对 `C[i,j]` 的竞争写入
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

**Handle 管理**：函数级 `static cublasHandle_t`，懒初始化（C++11 保证线程安全的 static 初始化）。整个进程生命周期内复用一个 handle。支持可选 `cudaStream_t` 参数，调用 `cublasSetStream` 设定。

---

## 5. 矩阵分解

### 5.1 SVD

#### CPU：双边 Jacobi (`decomposition_cpu.cpp`)

**算法**：隐式双边 Jacobi（在 A 的列上直接工作，不显式构造 `A^T A`）：

1. 初始化：`U = A`，`Vt = I`，`S = 0`
2. Jacobi 扫描循环（最多 100 轮）：
   - 对每对列 `(j1, j2)` 计算 `a = ||col_j1||²`，`b = ||col_j2||²`，`c = col_j1 · col_j2`
   - 收敛判断：`|c| / sqrt(a*b) < 1e-12` 则跳过
   - 计算 Givens 旋转参数：`tau = (a-b)/(2c)`，`t = sign(tau)/(|tau|+sqrt(1+tau²))`，`cs = 1/sqrt(1+t²)`，`sn = cs*t`
   - 双边应用旋转到 U 的列和 Vt 的行
3. 提取奇异值：`S[j] = ||U_col_j||`
4. 归一化 U 的列
5. 按奇异值降序排列

**时间复杂度**：O(n⁴) 每轮扫描。基准测试中 N > 256 自动跳过 CPU SVD。

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

然后从 R 中恢复 Q：初始化 `Q = I`，从右反向应用所有 Householder 变换。

**时间复杂度**：O(m n min(m,n)) = O(n³) 对于方阵。

#### GPU：cuSOLVER geqrf + orgqr (`decomposition.cu`)

- **geqrf**：计算 QR 分解（同 CPU Householder 原理，GPU 优化版本）
- **orgqr**：从反射向量显式构建正交矩阵 Q
- **限制**：当 `m > n`（矩形矩阵，行 > 列），orgqr 以 `k = min(m,n)` 调用，只有前 k 列形成标准正交基。剩余 `m-k` 列保持为单位向量，不是正交完备。这是 cuSOLVER 的行为限制。

### 5.3 对称特征值 (Eigh)

#### CPU：经典 Jacobi (`decomposition_cpu.cpp`)

直接在对称矩阵 A 上对角化：
1. 扫描上三角元素对 `(p, q)`，跳过 `|a_pq| < 1e-12`
2. 计算 Jacobi 旋转：`tau = (a_qq - a_pp)/(2*a_pq)` → `t, c, s`
3. 更新 `A(p,p)`, `A(q,q)`，清零 `A(p,q)` 和 `A(q,p)`
4. 更新 p 行和 q 列的其他元素（跳过 p, q 位置），保持对称性
5. 迭代直到 `max_off_diag < 1e-12` 或 100 轮

提取对角线作为特征值，选择排序降序排列。

**时间复杂度**：O(n⁴) 每轮。同样在 N > 256 时跳过 CPU 基准测试。

#### GPU：cuSOLVER syevd (`decomposition.cu`)

```cpp
cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                 n, A_work, lda, eigvals, d_work, lwork, d_dev_info);
```

- `NOVECTOR`：只求特征值，不求特征向量（点云协方差 PCA 场景只需特征值）
- `LOWER`：只引用下三角（矩阵是对称的）
- syevd 使用分治算法 (divide-and-conquer)，比 QR iteration 更快
- **注意**：cuSOLVER 返回特征值 **升序**。代码在 CPU 侧做选择排序翻转为降序，便于与 CPU Jacobi 结果对照

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

GPU：将 N×3 点云传回 CPU 计算协方差，再将 3×3 传回 GPU。协方差归约需要 reduction kernel，CPU 代码更简单且对 3×3 输出来说足够快。

---

## 8. 稀疏矩阵

### 8.1 COOMatrix (`sparse/coo_matrix.h`)

内部存储三个 `std::vector`：`_row_indices`、`_col_indices`、`_values`。`add()` 简单 push_back —— CPU only，不做去重或排序。

### 8.2 COO → CSR 转换 (`toCsr()`)

1. **排序三元组**：创建索引排列 `0..nnz-1`，按 `(row, col)` 排序
2. **计数非零元**：每行非零元个数 → `row_counts`
3. **前缀和**：`offsets[0]=0`，`offsets[r+1]=offsets[r]+row_counts[r]` → `row_offsets`
4. **散射**：按排列顺序写 `values` 和 `col_indices`，用行游标 `pos[row]`
5. **GPU 传输**：散射在 CPU 完成，结果通过 `cudaMemcpy` 传至 GPU

### 8.3 CSRMatrix (`sparse/csr_matrix.h`)

独立管理三个数组（不从 DeviceMatrix 继承数据存储）：
- `_values`：nnz 个 Scalar
- `_col_indices`：nnz 个 Index
- `_row_offsets`：rows+1 个 Index

每个数组独立分配和释放。析构函数调用 `releaseArrays()` 逐数组检查 nullptr 后释放。Move 语义转移三个指针的所有权。

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
| `solver_cpu.cpp` | 显式实例化 | `solve<float/double, CPU>` |
| `solver.cu` | 显式特化 | `solve<float/double, GPU>` |
| `csr_matrix.cpp` | 显式实例化 | `CSRMatrix/COOMatrix` × `{float,double}` × `{CPU,GPU}` |

GPU 版本使用**显式特化**（`template<>`）而非实例化，因为需要覆盖泛型模板中的 `if constexpr (Dev == GPU)` 分支。

---

## 11. 无 CUDA 时的行为

当 `-DPLAMATRIX_WITH_CUDA=OFF`：

- `CMakeLists.txt` 不启用 CUDA 语言，不链接 CUDA 库
- `no_cuda_stubs.h` 提供所有 CUDA 类型的桩定义
- `Device::GPU` 仍然可用，但底层用 CPU 内存（`malloc`/`free`）
- cuBLAS/cuSOLVER 调用不参与编译（`.cu` 文件不编译）
- GPU benchmark 函数通过 `#ifdef PLAMATRIX_WITH_CUDA` 保护
- 运行时 `Device::GPU` 矩阵行为与 CPU 完全相同（无加速效果）
- 所有 53 个单元测试在 CPU-only 构建中仍然通过

---

## 12. 基准测试程序

`plamatrix_benchmark` 的结构：

- `main.cpp`：CLI 参数解析（`--mode`, `--size`, `--output`, `--list`）
- `benchmark_cases.cpp`：CPU 基准用例（每个运算的 serial + OMP 版本）
- `benchmark_cases.cu`：GPU 基准用例（含传输时间单独测量）
- `report_writer.cpp`：Markdown 报告生成 + 环境信息采集

**计时方法**：`measure(fn, warmup, trials)`：
1. warmup 次热身（不计时）
2. trials 次计时，每次前后 `cudaDeviceSynchronize()`
3. 取中位数返回

**尺寸限制**：SVD/QR/Eigh 的 CPU 路径在 N > 256 时自动跳过（Jacobi O(n⁴) 太慢）。

**CUDA 用时**不含数据传输时间（传输单独计时存入 `time_transfer_ms`）。
