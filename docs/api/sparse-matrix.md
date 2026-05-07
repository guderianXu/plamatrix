# 稀疏矩阵 API

PlaMatrix 支持 COO (坐标格式) 和 CSR (压缩稀疏行) 两种稀疏矩阵格式，适用于点云 BA 优化和图优化场景。

## COOMatrix

三元组格式，适合逐步构建稀疏矩阵。

```cpp
template <typename Scalar, Device Dev>
class COOMatrix : public DeviceMatrix<Scalar, Dev>;
```

### 基本操作

```cpp
COOMatrix<float, Device::CPU> coo(4, 4);  // 4×4 稀疏矩阵
coo.add(0, 0, 3.14f);   // 行0 列0 值3.14
coo.add(1, 2, 2.71f);   // 行1 列2 值2.71
coo.add(3, 3, 1.0f);    // 行3 列3 值1.0

Index n = coo.nnz();     // 非零元个数 → 3
```

- `add(row, col, value)` — 添加一个三元组 (CPU only, push_back 到内部 vector)
- `nnz()` — 返回非零元个数

### COO → CSR 转换

```cpp
auto csr = coo.toCsr();  // CPU: 按(row,col)排序后构建CSR
                         // GPU: cusparseXcoo2csr
```

## CSRMatrix

压缩稀疏行格式，适合高效的矩阵向量乘和稀疏求解。

```cpp
template <typename Scalar, Device Dev>
class CSRMatrix : public DeviceMatrix<Scalar, Dev>;
```

### 内部结构

```
row_offsets:  [0, 2, 2, 3, 4]     // 每行起始偏移 (size = rows+1)
col_indices:  [0, 2, 1, 3]        // 列索引 (size = nnz)
values:       [1.0, 2.0, 3.0, 4.0] // 非零值 (size = nnz)
```

### 访问器

```cpp
Scalar* values();              // 非零值数组
Index* colIndices();           // 列索引数组
Index* rowOffsets();           // 行偏移数组 (长度 rows+1)
Index nnz() const;             // 非零元个数
```

## 稀疏求解器

```cpp
auto A = COOMatrix<float, Device::CPU>(n, n);
// ... 填充三元组 ...
auto csr = A.toCsr();
auto x = solve(csr, b);  // 稀疏线性求解 Ax = b
```

## 完整示例：三对角矩阵

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int n = 100;
COOMatrix<float, Device::CPU> coo(n, n);

// 构建三对角矩阵 (对角占优)
for (int i = 0; i < n; ++i)
{
    coo.add(i, i, 2.0f);          // 主对角
    if (i > 0)     coo.add(i, i-1, -1.0f);  // 下对角
    if (i < n-1)   coo.add(i, i+1, -1.0f);  // 上对角
}

// 转换为 CSR
auto csr = coo.toCsr();

// 构建右端项
DenseMatrix<float, Device::CPU> b(n, 1);
b.fill(1.0f);

// 求解 Ax = b
auto x = solve(csr, b);

// GPU 加速
auto csr_gpu = coo.toCsr();       // 返回 GPU CSR
auto b_gpu = b.toGpu();
auto x_gpu = solve(csr_gpu, b_gpu);
auto x_result = x_gpu.toCpu();
```
