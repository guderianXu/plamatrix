# SparseMatrix API

PlaMatrix 提供两种稀疏矩阵格式：

- **COOMatrix**：坐标格式 (Coordinate)，用于**矩阵构建**阶段。支持 `add()` 方法逐个添加非零元素。
- **CSRMatrix**：压缩稀疏行格式，用于**矩阵运算**阶段。通过 `COOMatrix::toCsr()` 转换得到。

两种格式均支持 `Device::CPU` 和 `Device::GPU`。

## 头文件

```cpp
#include <plamatrix/plamatrix.h>
```

## COOMatrix

`COOMatrix<Scalar, Dev>` 以三元组 `(row, col, value)` 的形式存储非零元素。元素按插入顺序存储，在 `toCsr()` 时按 `(row, col)` 排序。

### 构造

```cpp
// 创建 10x10 的 COO 稀疏矩阵（初始为空）
COOMatrix<float, Device::CPU> coo(10, 10);
```

参数说明：
- `rows`：矩阵行数
- `cols`：矩阵列数

> 构造时仅保存维度信息，不预分配元素存储空间。

### add — 添加非零元素

```cpp
// 逐个添加非零元素
coo.add(0, 0, 3.14f);   // A(0,0) = 3.14
coo.add(0, 1, 2.71f);   // A(0,1) = 2.71
coo.add(1, 1, 1.41f);   // A(1,1) = 1.41
coo.add(2, 0, 0.57f);   // A(2,0) = 0.57
```

参数说明：
- `row`：行索引（0-based）
- `col`：列索引（0-based）
- `value`：非零值

> `add()` 不检查重复元素。如果需要累加同一位置的多个值，请在构建时自行处理后再添加。添加顺序不影响最终 CSR 格式的正确性。

### nnz — 非零元素数量

```cpp
Index count = coo.nnz();  // 返回已添加的非零元素数量
```

### toCsr — 转换为 CSR 格式

```cpp
auto csr = coo.toCsr();
// 类型: CSRMatrix<float, Device::CPU>
// csr.rows() == 10, csr.cols() == 10, csr.nnz() == 4
```

CPU 路径执行流程：
1. 按 `(row, col)` 对三元组排序
2. 统计每行非零元素数量
3. 构建行偏移数组 (`row_offsets`)
4. 按排序后的顺序填充 `values` 和 `col_indices` 数组

GPU 路径：在 host 端构建 CSR 数组，然后通过 `cudaMemcpy` 拷贝到 device 端。

## CSRMatrix

`CSRMatrix<Scalar, Dev>` 以 CSR (Compressed Sparse Row) 格式存储稀疏矩阵。由三个数组表示：

- `values`：长度 `nnz`，按行依次存储非零值
- `col_indices`：长度 `nnz`，对应每个值的列索引
- `row_offsets`：长度 `rows + 1`，第 `i` 行元素在 `values` 中的起始位置

### 构造

```cpp
// 通常从 COOMatrix 转换得到
auto csr = coo.toCsr();
```

如需手动构造：

```cpp
CSRMatrix<float, Device::CPU> csr(5, 5, 3);  // 5x5 矩阵，3 个非零元素
// row_offsets 已初始化为零
// values 和 col_indices 已分配但未初始化
// 需要手动填充 values、col_indices 和 row_offsets
```

参数说明：
- `rows`：矩阵行数
- `cols`：矩阵列数
- `nnz`：非零元素数量

> `CSRMatrix` 同样不可拷贝，仅支持移动语义。

### 访问器

```cpp
Index n = csr.nnz();          // 非零元素数量
Scalar* vals = csr.values();  // values 数组指针
Index* cols = csr.colIndices(); // 列索引数组指针
Index* offs = csr.rowOffsets(); // 行偏移数组指针

// const 版本
const Scalar* c_vals = csr.values();
const Index* c_cols = csr.colIndices();
const Index* c_offs = csr.rowOffsets();
```

### CSR 格式说明

对于如下矩阵：

```
[1.0  0.0  2.0]
[0.0  3.0  0.0]
[4.0  0.0  0.0]
```

CSR 数组为：

```cpp
values      = [1.0, 2.0, 3.0, 4.0]
col_indices = [0,   2,   1,   0  ]
row_offsets = [0,   2,   3,   4  ]
              // row0从index 0开始，row1从index 2开始，row2从index 3开始
```

第 `r` 行的元素索引范围为 `[row_offsets[r], row_offsets[r+1])`。

## 完整示例

```cpp
#include <plamatrix/plamatrix.h>
#include <iostream>
#include <cassert>

using namespace plamatrix;

int main()
{
    // 1. 构建稀疏矩阵（COO 格式）
    // 3x3 三对角矩阵:
    //   [2 -1  0]
    //   [-1 2 -1]
    //   [0 -1  2]
    COOMatrix<double, Device::CPU> coo(3, 3);
    coo.add(0, 0,  2.0);
    coo.add(0, 1, -1.0);
    coo.add(1, 0, -1.0);
    coo.add(1, 1,  2.0);
    coo.add(1, 2, -1.0);
    coo.add(2, 1, -1.0);
    coo.add(2, 2,  2.0);

    assert(coo.nnz() == 7);

    // 2. 转换为 CSR 格式
    auto csr = coo.toCsr();
    assert(csr.nnz() == 7);
    assert(csr.rows() == 3);
    assert(csr.cols() == 3);

    // 3. 验证 CSR 数据
    const double* vals = csr.values();
    const Index*  cidx = csr.colIndices();
    const Index*  roff = csr.rowOffsets();

    std::cout << "Row offsets: ";
    for (Index i = 0; i <= csr.rows(); ++i)
        std::cout << roff[i] << " ";
    std::cout << std::endl;

    std::cout << "Non-zero elements:" << std::endl;
    for (Index r = 0; r < csr.rows(); ++r)
    {
        for (Index k = roff[r]; k < roff[r + 1]; ++k)
        {
            std::cout << "  A(" << r << ", " << cidx[k] << ") = " << vals[k] << std::endl;
        }
    }

    return 0;
}
```

## 性能建议

1. **COO 用于构建**：在矩阵构建阶段，使用 COO 格式逐个添加元素最为方便
2. **CSR 用于计算**：构建完成后转换为 CSR，CSR 格式更适合矩阵向量乘法等运算
3. **GPU 使用**：将 COO 构建为 `Device::GPU`，`toCsr()` 会自动在 host 端排序并拷贝到 device
4. **大批量构建**：对于超大规模稀疏矩阵，建议预分配 `std::vector` 并批量写入，避免逐个 `add()` 的 vector 动态扩容开销
