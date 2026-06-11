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

- **CPU**：可用时使用 BLAS；否则使用项目内三重循环，较大工作量走 OpenMP
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
```

- CPU 侧小矩阵保持串行，超过内部阈值后使用 OpenMP 并行
- 同步 GPU 接口使用自定义 CUDA kernel (256 threads/block)，支持可选 CUDA stream；
  返回前会同步该 stream
- GPU 输出复用重载要求 `C_gpu` 与输入矩阵同尺寸；`addAsync/subAsync` 不主动同步
- **标量运算**：CPU 矩阵支持 `2.0f * A`，矩阵加法使用 `add(A, B)`；
  标量乘加写作 `auto C = add(2.0f * A, B);`

## SVD 分解

```cpp
/// A = U * diag(S) * Vt
/// 返回 (U, S_vector, Vt)，奇异值降序排列
template <typename Scalar, Device Dev>
std::tuple<DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>> svd(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：可用时使用 LAPACK `gesvd`；否则使用项目内 Jacobi fallback
- **GPU**：`cusolverDnSgesvd` (float) / `cusolverDnDgesvd` (double)，legacy cuSOLVER 路径要求 `rows >= cols`
- 返回形状为 `U(m,m)`, `S(min(m,n),1)`, `Vt(n,n)`。

## QR 分解

```cpp
/// A = Q * R (Q 正交, R 上三角)
template <typename Scalar, Device Dev>
std::tuple<DenseMatrix<Scalar, Dev>, DenseMatrix<Scalar, Dev>>
qr(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：Householder 反射变换
- **GPU**：`cusolverDnSgeqrf` + `cusolverDnSorgqr`

## 对称特征值

```cpp
/// 对称矩阵特征值分解，返回特征值向量 (降序)
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> eigh(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：可用时使用 LAPACK `syev`；否则使用项目内 Jacobi fallback
- **GPU**：`cusolverDnSsyevd` (分治算法)

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
| gemm | 依 BLAS/CUDA 和矩阵尺寸而定 | CPU BLAS 在小矩阵上可能优于 CUDA |
| add/sub | ~4096 | 内存带宽受限，GPU 优势出现较晚 |
| svd | 依 LAPACK/CUDA 和矩阵尺寸而定 | CPU LAPACK 在小矩阵上可能优于 CUDA |
| solve | ~256 | GPU LU 分解远超 CPU 高斯消元 |

*注：临界尺寸因硬件而异，请使用 `plamatrix_benchmark` 工具实际测量。*

CUDA benchmark 会在 CUDA run 边界启用 GPU memory pool。
GEMM/add/sub 使用 CUDA event 计时并复用输出矩阵，报告里的 CUDA 时间主要反映设备端计算，
不包含每轮输出分配成本；传输时间使用 pinned host 输入矩阵，单独记录在 `transfer_ms`。
