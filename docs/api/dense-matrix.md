# DenseMatrix API

`DenseMatrix<Scalar, Dev>` 是 PlaMatrix 的核心密集矩阵类型。模板参数 `Scalar` 为元素类型（`float` 或 `double`），`Dev` 为设备类型（`Device::CPU` 或 `Device::GPU`）。

## 头文件

```cpp
#include <plamatrix/plamatrix.h>
```

## 类型定义

```cpp
namespace plamatrix {

template <typename Scalar, Device Dev>
class DenseMatrix : public DeviceMatrix<Scalar, Dev>;

}
```

### 模板参数

| 参数 | 说明 |
|------|------|
| `Scalar` | 元素类型，通常为 `float` 或 `double` |
| `Dev` | 设备类型。`Device::CPU` 表示存入 CPU 内存，`Device::GPU` 表示存入 GPU 显存 |

### 常用别名

```cpp
using MatrixXf = DenseMatrix<float,  Device::CPU>;  // float CPU 矩阵（任意大小）
using MatrixXd = DenseMatrix<double, Device::CPU>;  // double CPU 矩阵（任意大小）
```

## 构造与析构

### 默认构造

```cpp
DenseMatrix<float, Device::CPU> A;  // 0x0 空矩阵
```

### 指定维度的构造

```cpp
// 创建 m 行 n 列的矩阵，初始化为全零
DenseMatrix<float, Device::CPU> A(3, 4);  // 3 行 4 列，全零
DenseMatrix<double, Device::GPU> B(100, 100);  // GPU 端 100x100，全零
```

> 构造时自动分配内存并零初始化。CPU 使用 `std::memset`，GPU 使用 `cudaMemset`。

### 移动语义

`DenseMatrix` 不可拷贝（`non-copyable`），但支持移动构造和移动赋值：

```cpp
DenseMatrix<float, Device::CPU> A(3, 3);
A.setValue(0, 0, 5.0f);

// 移动构造：转移所有权，原矩阵变为空 (0x0)
DenseMatrix<float, Device::CPU> B = std::move(A);
// A 不再可用（A.rows() == 0, A.cols() == 0）
```

## 元素访问

### operator() — 仅 CPU

```cpp
// 返回引用，可读写
DenseMatrix<float, Device::CPU> A(3, 3);
A(0, 1) = 3.14f;          // 写入
float val = A(0, 1);      // 读取
```

> `operator()` 采用**列优先**布局：`data[row + col * rows]`。仅在 CPU 设备可用。

### setValue / getValue — CPU 和 GPU 均可用

```cpp
// GPU 矩阵也可使用
DenseMatrix<float, Device::GPU> A_gpu(3, 3);
A_gpu.setValue(0, 0, 1.0f);   // 设置 A(0,0) = 1.0（host→device 传输）
float v = A_gpu.getValue(1, 2); // 从 GPU 读取，返回 host 值

// CPU 矩阵同样可用
DenseMatrix<float, Device::CPU> A_cpu(3, 3);
A_cpu.setValue(2, 1, 5.0f);    // 直接写入
float u = A_cpu.getValue(2, 1); // 直接读取，返回 5.0
```

> `setValue`/`getValue` 是设备间通用的单元素读写接口。对于 CPU 矩阵等同于直接内存访问；对 GPU 矩阵会触发单次 `cudaMemcpy`，适合少量元素操作。

## 批量操作

### fill — 填充统一值

```cpp
DenseMatrix<float, Device::CPU> A(100, 100);
A.fill(1.0f);  // 所有元素设置为 1.0

DenseMatrix<float, Device::GPU> B(100, 100);
B.fill(2.0f);  // GPU 端全填充 2.0（零值用 cudaMemset 优化）

B.fill(0.0f);  // 零值填充在 GPU 上通过 cudaMemset 高效实现
```

> CPU 端使用 `std::fill_n` 搭配 OpenMP。GPU 端零值填充用 `cudaMemset`，非零值填充用自定义 kernel。

### transpose — 转置

```cpp
DenseMatrix<float, Device::CPU> A(2, 3);
A.setValue(0, 0, 1.0f); A.setValue(0, 1, 2.0f); A.setValue(0, 2, 3.0f);
A.setValue(1, 0, 4.0f); A.setValue(1, 1, 5.0f); A.setValue(1, 2, 6.0f);
// A = [1 2 3]
//     [4 5 6]

auto AT = A.transpose();
// AT = [1 4]
//      [2 5]
//      [3 6]
// AT.rows() == 3, AT.cols() == 2
```

> 返回新矩阵。CPU 端使用嵌套循环，GPU 端使用 2D 转置 kernel。

## 设备间传输

### toGpu — CPU → GPU

```cpp
DenseMatrix<float, Device::CPU> A_cpu(100, 200);
// ... 在 CPU 上填充数据 ...
auto A_gpu = A_cpu.toGpu();  // 将数据拷贝到 GPU
// A_gpu 类型: DenseMatrix<float, Device::GPU>
```

### toCpu — GPU → CPU

```cpp
DenseMatrix<float, Device::GPU> B_gpu(100, 200);
// ... 在 GPU 上计算 ...
auto B_cpu = B_gpu.toCpu();  // 将数据拷贝回 CPU
// B_cpu 类型: DenseMatrix<float, Device::CPU>
```

> 这些方法触发完整的设备间内存拷贝 (`cudaMemcpy`)。仅在编译期目标设备静态检查通过时才可用。

## 基类访问器

继承自 `DeviceMatrix<Scalar, Dev>`：

```cpp
DenseMatrix<float, Device::CPU> A(12, 34);

Index r = A.rows();     // 12 — 行数
Index c = A.cols();     // 34 — 列数
Index n = A.size();     // 408 — 总元素数 (rows * cols)
float* p = A.data();    // 底层数据指针（CPU 端可直接使用）
auto dev = A.device();  // Device::CPU — 编译期设备常量
```

> `data()` 在 **CPU** 上返回可用的 host 指针，可结合 OpenMP 进行高效遍历。在 GPU 上返回 device 指针，host 侧不可直接解引用。

## 完整示例

```cpp
#include <plamatrix/plamatrix.h>
#include <iostream>

using namespace plamatrix;

int main()
{
    // 1. 创建 CPU 矩阵并填充
    DenseMatrix<float, Device::CPU> A(3, 3);
    for (Index j = 0; j < 3; ++j)
        for (Index i = 0; i < 3; ++i)
            A(i, j) = static_cast<float>(i + j * 3);

    std::cout << "A(0,0) = " << A(0, 0) << std::endl;
    std::cout << "A rows = " << A.rows() << ", cols = " << A.cols() << std::endl;

    // 2. 转置
    auto AT = A.transpose();
    std::cout << "AT(0,1) = " << AT(0, 1) << " (was A(1,0) = " << A(1, 0) << ")" << std::endl;

    // 3. 转移到 GPU，进行操作，再取回
    auto A_gpu = A.toGpu();
    A_gpu.fill(7.0f);          // GPU 端全填 7
    auto B = A_gpu.toCpu();    // 取回 CPU
    std::cout << "B(2,2) = " << B(2, 2) << " (应为 7.0)" << std::endl;

    // 4. 移动语义
    DenseMatrix<float, Device::CPU> C(2, 2);
    C.setValue(0, 0, 99.0f);
    DenseMatrix<float, Device::CPU> D = std::move(C);
    // C 不再可用，D 持有数据

    return 0;
}
```
