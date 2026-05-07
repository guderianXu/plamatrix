# 线性代数 API

## 矩阵乘法 (gemm)

```cpp
/// C = A * B  (column-major layout)
/// A: m×k, B: k×n → C: m×n
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> gemm(
    const DenseMatrix<Scalar, Dev>& A,
    const DenseMatrix<Scalar, Dev>& B,
    cudaStream_t stream = nullptr  // GPU only
);
```

- **CPU**：三重循环 + `#pragma omp parallel for collapse(2)` 并行
- **GPU**：`cublasSgemm` / `cublasDgemm`，支持指定 CUDA stream
- **示例**：`auto C = gemm(A, B);`

## 逐元素运算

```cpp
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> add(A, B);  // C[i] = A[i] + B[i]

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> sub(A, B);  // C[i] = A[i] - B[i]
```

- CPU 侧使用 OpenMP 并行
- GPU 侧使用自定义 CUDA kernel (256 threads/block)
- **标量运算**：`auto C = 2.0f * A + B;` (标量乘 + 矩阵加)

## SVD 分解

```cpp
/// A = U * diag(S) * Vt
/// 返回 (U, S_vector, Vt)，奇异值降序排列
template <typename Scalar, Device Dev>
std::tuple<DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>,
           DenseMatrix<Scalar, Dev>> svd(const DenseMatrix<Scalar, Dev>& A);
```

- **CPU**：单边 Jacobi 算法 (迭代 Givens 旋转)
- **GPU**：`cusolverDnSgesvd` (float) / `cusolverDnDgesvd` (double)

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

- **CPU**：Jacobi 特征值算法
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
| gemm | ~512 | 超过此尺寸 GPU 显著优于 CPU |
| add/sub | ~4096 | 内存带宽受限，GPU 优势出现较晚 |
| svd | ~64 | GPU 加速非常显著 (O(n³) vs Jacobi) |
| solve | ~256 | GPU LU 分解远超 CPU 高斯消元 |

*注：临界尺寸因硬件而异，请使用 `plamatrix_benchmark` 工具实际测量。*
