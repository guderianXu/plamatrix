# DenseMatrix API

DenseMatrix 是 PlaMatrix 的核心容器类型，表示一个列优先存储的密集矩阵。

## 类型定义

```cpp
template <typename Scalar, Device Dev>
class DenseMatrix : public DeviceMatrix<Scalar, Dev>;
```

- `Scalar`: 元素类型 (`float` 或 `double`)
- `Dev`: 设备类型 (`Device::CPU` 或 `Device::GPU`)

## 构造函数

```cpp
DenseMatrix();                        // 空矩阵 (0×0)
DenseMatrix(Index rows, Index cols);  // 创建 rows×cols 矩阵，零初始化
```

```cpp
DenseMatrix<float, Device::CPU> A(100, 200);  // 100行 200列，全零
DenseMatrix<double, Device::GPU> B(50, 50);   // GPU 50×50 double矩阵
```

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

## 设备传输

```cpp
auto A_gpu = A.toGpu();  // CPU → GPU (cudaMemcpy HostToDevice)
auto A_cpu = A_gpu.toCpu();  // GPU → CPU (cudaMemcpy DeviceToHost)
```

传输是一次性的、显式的、昂贵的。库不会隐式搬移数据。

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

DenseMatrix 禁止拷贝，支持移动：

```cpp
DenseMatrix<float, Device::CPU> A(100, 100);
DenseMatrix<float, Device::CPU> B(std::move(A));  // A 变为空
// A.data() == nullptr   (移动后源矩阵置空)
```

## 完整示例

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

// 创建 CPU 矩阵
DenseMatrix<float, Device::CPU> A(3, 4);
A(0, 0) = 1.0f;  A(1, 0) = 2.0f;  A(2, 0) = 3.0f;
A(0, 1) = 4.0f;  A(1, 1) = 5.0f;  A(2, 1) = 6.0f;
// ... 填充剩余元素 ...

// 转移到 GPU 计算
auto A_gpu = A.toGpu();
auto C_gpu = gemm(A_gpu, A_gpu.transpose());

// 取回结果
auto C = C_gpu.toCpu();
```
