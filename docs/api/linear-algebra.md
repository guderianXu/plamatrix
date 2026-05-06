# 线性代数 API

PlaMatrix 提供完整的线性代数运算，覆盖基础运算、矩阵分解和线性求解。所有运算均提供 CPU（OpenMP）和 GPU（cuBLAS / cuSOLVER）两种后端。

## 头文件

```cpp
#include <plamatrix/plamatrix.h>
```

## 基础运算

### add — 矩阵加法

```cpp
// CPU: 逐元素加法，OpenMP 并行
DenseMatrix<Scalar, Device::CPU> C = add(A, B);  // C = A + B
```

```cpp
// GPU: kernel 并行加法
DenseMatrix<Scalar, Device::GPU> C = add(A_gpu, B_gpu);
```

| 参数 | 说明 |
|------|------|
| `A, B` | 同维度矩阵 |
| 返回值 | 新矩阵，维度与 `A`、`B` 相同 |
| 异常 | `A`、`B` 维度不匹配时行为未定义 |

> **性能**：CPU 端使用 `#pragma omp parallel for` 多线程加速。GPU 端每个线程处理一个元素。

### sub — 矩阵减法

```cpp
auto C = sub(A, B);  // C = A - B
```

接口与 `add` 完全一致，执行逐元素减法。

### 标量乘法

```cpp
// C = alpha * A
auto C1 = 2.0f * A;    // 每个元素乘以 2.0
auto C2 = A * 2.0f;    // 等价形式

// 仅支持 CPU 矩阵
DenseMatrix<float, Device::CPU> A(3, 3);
auto C = 0.5f * A;  // 每个元素乘以 0.5
```

### 标量加法

```cpp
// C = alpha + A
auto C = 1.0f + A;  // 每个元素加 1.0

// 仅支持 CPU 矩阵
```

> 标量运算目前仅提供了 CPU 重载。GPU 上可以通过 `add(GPU_matrix, fill_with_scalar)` 或自定义 kernel 实现。

## 矩阵乘法 (GEMM)

### 通用矩阵乘法：C = A * B

```cpp
// CPU GEMM
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> gemm(
    const DenseMatrix<Scalar, Device::CPU>& A,
    const DenseMatrix<Scalar, Device::CPU>& B
);

// GPU GEMM (cuBLAS)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> gemm(
    const DenseMatrix<Scalar, Device::GPU>& A,
    const DenseMatrix<Scalar, Device::GPU>& B,
    cudaStream_t stream = nullptr
);
```

| 参数 | 说明 |
|------|------|
| `A` | 左矩阵，维度 m x k |
| `B` | 右矩阵，维度 k x n |
| `stream` | CUDA 流（仅 GPU, 可选） |
| 返回值 | 新矩阵 C，维度 m x n |

**示例**：

```cpp
// CPU GEMM
DenseMatrix<float, Device::CPU> A(100, 50);  // 100x50
DenseMatrix<float, Device::CPU> B(50, 20);   // 50x20
auto C = gemm(A, B);  // C: 100x20

// GPU GEMM（通过 cuBLAS）
auto A_gpu = A.toGpu();
auto B_gpu = B.toGpu();
auto C_gpu = gemm(A_gpu, B_gpu);
auto C_result = C_gpu.toCpu();
```

> **性能说明**：
> - GPU GEMM 通过 cuBLAS 实现，对于大矩阵（n > 1000）可提供 10-100x 加速
> - 列优先 (column-major) 存储，与 cuBLAS 的原生布局一致
> - GPU 版本支持通过 `stream` 参数进行异步计算

## 矩阵分解

### SVD — 奇异值分解

将矩阵 A 分解为 `A = U * diag(S) * Vt`，其中 U 和 Vt 为正交矩阵，S 为奇异值向量（降序排列）。

```cpp
// CPU SVD
template <typename Scalar>
std::tuple<
    DenseMatrix<Scalar, Device::CPU>,   // U: m x m
    DenseMatrix<Scalar, Device::CPU>,   // S: min(m,n) x 1 列向量
    DenseMatrix<Scalar, Device::CPU>    // Vt: n x n
> svd(const DenseMatrix<Scalar, Device::CPU>& A);

// GPU SVD (cuSOLVER gesvd)
template <typename Scalar>
std::tuple<
    DenseMatrix<Scalar, Device::GPU>,
    DenseMatrix<Scalar, Device::GPU>,
    DenseMatrix<Scalar, Device::GPU>
> svd(const DenseMatrix<Scalar, Device::GPU>& A);
```

**示例**：

```cpp
DenseMatrix<float, Device::CPU> A(4, 3);
// ... 填充 A ...
auto [U, S, Vt] = svd(A);

// U 维度: 4x4 (m x m)
// S 维度: 3x1 (min(m,n) x 1)，奇异值降序
// Vt 维度: 3x3 (n x n)

// 验证: A ≈ U * diag(S) * Vt
// 提取第一个奇异值
float sigma_0 = S(0, 0);
```

### QR — QR 分解

将矩阵 A 分解为 `A = Q * R`，其中 Q 为正交矩阵 (m x m)，R 为上三角矩阵 (m x n)。

```cpp
// CPU QR (Householder 反射)
template <typename Scalar>
std::tuple<
    DenseMatrix<Scalar, Device::CPU>,   // Q: m x m
    DenseMatrix<Scalar, Device::CPU>    // R: m x n
> qr(const DenseMatrix<Scalar, Device::CPU>& A);

// GPU QR (cuSOLVER geqrf + orgqr)
template <typename Scalar>
std::tuple<
    DenseMatrix<Scalar, Device::GPU>,
    DenseMatrix<Scalar, Device::GPU>
> qr(const DenseMatrix<Scalar, Device::GPU>& A);
```

**示例**：

```cpp
DenseMatrix<double, Device::CPU> A(6, 4);
// ... 填充 A ...
auto [Q, R] = qr(A);

// Q: 6x6 (正交矩阵)
// R: 6x4 (上三角矩阵)
// 验证: A ≈ Q * R
```

### eigh — 对称矩阵特征值分解

计算对称矩阵的全体特征值，按降序排列。

```cpp
// CPU eigh (Jacobi 算法)
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> eigh(
    const DenseMatrix<Scalar, Device::CPU>& A
);

// GPU eigh (cuSOLVER syevd)
template <typename Scalar>
DenseMatrix<Scalar, Device::GPU> eigh(
    const DenseMatrix<Scalar, Device::GPU>& A
);
```

| 参数 | 说明 |
|------|------|
| `A` | 对称矩阵，n x n |
| 返回值 | 列向量，n x 1，特征值降序 |

> **注意**：输入矩阵必须是**对称**的（A = A^T）。当前实现**不检查**对称性，调用者需确保输入满足条件。

**示例**：

```cpp
DenseMatrix<float, Device::CPU> A(3, 3);
// 构造对称矩阵
A.setValue(0, 0, 4.0f); A.setValue(0, 1, 1.0f); A.setValue(0, 2, 0.0f);
A.setValue(1, 0, 1.0f); A.setValue(1, 1, 3.0f); A.setValue(1, 2, 1.0f);
A.setValue(2, 0, 0.0f); A.setValue(2, 1, 1.0f); A.setValue(2, 2, 2.0f);

auto eigvals = eigh(A);

// eigvals 维度: 3x1
float lambda_0 = eigvals(0, 0);  // 最大特征值
float lambda_1 = eigvals(1, 0);
float lambda_2 = eigvals(2, 0);  // 最小特征值
```

## 线性方程组求解

### solve — 求解 A * X = B

使用 LU 分解与部分选主元 (partial pivoting) 求解线性方程组。

```cpp
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> solve(
    const DenseMatrix<Scalar, Dev>& A,   // n x n 方阵
    const DenseMatrix<Scalar, Dev>& B    // n x nrhs 右端项
);
```

| 参数 | 说明 |
|------|------|
| `A` | 系数矩阵，必须为方阵 (n x n) |
| `B` | 右端项，n 行，任意列数 (nrhs) |
| 返回值 | 解矩阵 X，维度 n x nrhs |
| 异常 | `A` 非方阵、维度不匹配、矩阵奇异时抛出 `std::runtime_error` |

**示例**：

```cpp
// 构造方程: 3x + y = 9, x + 2y = 8
DenseMatrix<float, Device::CPU> A(2, 2);
A(0, 0) = 3.0f; A(0, 1) = 1.0f;
A(1, 0) = 1.0f; A(1, 1) = 2.0f;

DenseMatrix<float, Device::CPU> B(2, 1);
B(0, 0) = 9.0f;
B(1, 0) = 8.0f;

auto X = solve(A, B);
// X[0] = 2.0 (x = 2)
// X[1] = 3.0 (y = 3)

// 多右端项
DenseMatrix<float, Device::CPU> B2(2, 2);
B2(0, 0) = 9.0f; B2(0, 1) = 5.0f;
B2(1, 0) = 8.0f; B2(1, 1) = 4.0f;

auto X2 = solve(A, B2);  // 同时求解两个方程组
```

## 设备间切换典型模式

```cpp
// 1. CPU 准备数据
DenseMatrix<double, Device::CPU> A_cpu(1000, 800);
// ... 填充数据 ...

// 2. 转移到 GPU
auto A_gpu = A_cpu.toGpu();

// 3. GPU 加速运算
auto [U_gpu, S_gpu, Vt_gpu] = svd(A_gpu);

// 4. 结果取回 CPU
auto U = U_gpu.toCpu();
auto S = S_gpu.toCpu();
auto Vt = Vt_gpu.toCpu();
```

## 数值精度说明

| 运算 | CPU float 误差 | CPU double 误差 | GPU float 误差 | GPU double 误差 |
|------|---------------|-----------------|---------------|-----------------|
| GEMM | < 1e-4 | < 1e-12 | < 1e-4 | < 1e-12 |
| SVD | < 1e-3 | < 1e-10 | < 1e-3 | < 1e-10 |
| QR | < 1e-4 | < 1e-12 | < 1e-4 | < 1e-12 |
| Eigh | < 1e-3 | < 1e-10 | < 1e-3 | < 1e-10 |
| Solve | < 1e-4 | < 1e-12 | < 1e-4 | < 1e-12 |

> 以上为典型数量级。GPU 和 CPU 的结果可能因浮点运算顺序不同而产生微小差异。对精度要求极高的场景，推荐使用 `double`。
