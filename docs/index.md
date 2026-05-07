# PlaMatrix 文档

## 概述

PlaMatrix 是一个面向点云处理的高性能矩阵运算库。它在 CPU 侧使用 OpenMP 多线程并行，GPU 侧通过 CUDA / cuBLAS / cuSOLVER 加速，为大规模点云数据的矩阵运算提供统一接口。

### 核心优势

- **双设备统一 API**：同一套模板代码编译出 CPU 和 GPU 两个版本，接口一致
- **显式设备管理**：`toCpu()` / `toGpu()` 明确标记数据传输，不隐藏开销
- **点云场景定制**：内置旋转矩阵、刚体变换、协方差矩阵等点云常用运算
- **性能可量化**：内置三层基准测试工具，自动生成性能报告

## 文档导航

| 文档 | 内容 |
|------|------|
| [编译指南](build.md) | 依赖安装、CMake 选项、各平台编译步骤 |
| [DenseMatrix API](api/dense-matrix.md) | 密集矩阵：构造、访问、填充、转置、设备传输 |
| [稀疏矩阵 API](api/sparse-matrix.md) | COO / CSR 格式、构建与转换 |
| [线性代数 API](api/linear-algebra.md) | 矩阵乘法 (gemm)、SVD、QR、特征值、线性求解 |
| [点云运算 API](api/point-cloud.md) | 旋转矩阵、刚体变换、点变换、协方差 |
| [贡献指南](contributing.md) | 编码规范、TDD 流程、PR 清单 |

## 5 分钟快速上手

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    // 1. 创建 CPU 矩阵并填充
    DenseMatrix<float, Device::CPU> A(1000, 1000);
    DenseMatrix<float, Device::CPU> B(1000, 1000);
    A.fill(1.0f);
    B.fill(2.0f);

    // 2. CPU 逐元素加法 (自动 OpenMP 并行)
    auto C = add(A, B);

    // 3. 转移到 GPU 做矩阵乘法
    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    auto D_gpu = gemm(A_gpu, B_gpu);  // cuBLAS Sgemm

    // 4. 取回结果
    auto D = D_gpu.toCpu();

    return 0;
}
```

### 在你的项目中使用

```cmake
# 方式一: 安装后调用
find_package(plamatrix REQUIRED)
target_link_libraries(my_project plamatrix::plamatrix)

# 方式二: 源码集成
add_subdirectory(plamatrix)
target_link_libraries(my_project plamatrix::plamatrix)
```
