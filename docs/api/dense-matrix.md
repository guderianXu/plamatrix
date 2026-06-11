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

传输是一次性的、显式的、昂贵的。库不会隐式搬移数据。

普通 CPU 矩阵使用 pageable host memory。需要让 `cudaMemcpyAsync` 满足真正异步传输条件时，
使用 pinned/page-locked CPU 矩阵：

```cpp
auto pinned = DenseMatrix<float, Device::CPU>::pinned(rows, cols);
bool is_pinned = pinned.isPinnedHost(); // true
```

异步传输返回后不会主动同步；调用方必须在读取 CPU 结果或跨 stream 使用 GPU 结果前
等待对应 stream。
热循环里可以复用输出矩阵：

```cpp
auto host = DenseMatrix<float, Device::CPU>::pinned(A.rows(), A.cols());
DenseMatrix<float, Device::GPU> gpu(A.rows(), A.cols());
auto back = DenseMatrix<float, Device::CPU>::pinned(A.rows(), A.cols());

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
    DenseMatrix<float, Device::GPU> tmp(rows, cols);
} // tmp 的 block 进入 pool，后续同字节数分配可复用

GpuAllocator<float>::releaseMemoryPool();
GpuAllocator<float>::setMemoryPoolEnabled(false);
```

内存池按字节数缓存空闲 block，`releaseMemoryPool()` 会释放当前缓存。
异步 kernel 或异步传输仍要求调用方保证相关矩阵在 stream 完成前保持有效。

## GPU 异步计算和输出复用

同步 GPU 运算会在返回前同步传入的 CUDA stream，适合直接取结果：

```cpp
auto C_gpu = gemm(A_gpu, B_gpu);
auto D_gpu = add(A_gpu, B_gpu);
```

需要在循环或 benchmark 中减少分配和同步开销时，可复用输出矩阵并使用异步接口：

```cpp
DenseMatrix<float, Device::GPU> C_gpu(A_gpu.rows(), B_gpu.cols());
DenseMatrix<float, Device::GPU> D_gpu(A_gpu.rows(), A_gpu.cols());

gemmAsync(A_gpu, B_gpu, C_gpu, stream);
addAsync(A_gpu, A_gpu, D_gpu, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

`gemmAsync`、`addAsync`、`subAsync` 只提交 GPU 工作，不主动同步；
调用方负责保证输入/输出矩阵在对应 stream 完成前保持有效。

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
