# PlaMatrix 文档

## 概述

PlaMatrix 是一个面向点云处理的高性能矩阵运算库。它在 CPU 侧使用 OpenMP 多线程并行，GPU 侧通过 CUDA / cuBLAS / cuSOLVER 加速，为大规模点云数据的矩阵运算提供统一接口。

### 核心优势

- **直观的矩阵 API**：`MatrixXd`、`Vector3d`、工厂函数和算术运算符表达常见线性代数计算
- **自动选择设备**：日常密集运算在 CPU/CUDA 间选择，应用无需指定设备
- **底层设备控制**：后端维护者通过 `plamatrix::internal` 管理设备、stream 和工作区
- **点云场景定制**：内置旋转矩阵、刚体变换、协方差矩阵等点云常用运算
- **性能可量化**：内置三层基准测试工具，自动生成性能报告

## 文档导航

| 文档 | 内容 |
|------|------|
| [编译指南](build.md) | 依赖安装、CMake 选项、各平台编译步骤 |
| [密集矩阵 API](api/dense-matrix.md) | `Matrix`、运算符、自动求值与表达式 |
| [稀疏矩阵 API](api/sparse-matrix.md) | Triplet、SparseMatrix、迭代与直接求解器 |
| [线性代数 API](api/linear-algebra.md) | 矩阵乘法 (gemm)、SVD、QR、特征值、线性求解 |
| [Geometry API](api/geometry.md) | Quaternion、AngleAxis、Transform、欧拉角与 Umeyama |
| [非线性优化 API](api/optimization.md) | 鲁棒损失、二分块法方程、Schur-PCG、LM 阻尼策略 |
| [Vulkan cooperative matrix API](api/vulkan-cooperative-matrix.md) | Tensor Core 类硬件探测、16×16 批量乘法、设备常驻 Schur 与 FP32 回退 |
| [三维向量与点云运算 API](api/point-cloud.md) | 内部 PackedVector3 算术、旋转矩阵、刚体变换、点变换、协方差 |
| [后端实现接口](api/backend-internals.md) | 内部存储、执行上下文和原语；不属于公开契约 |
| [代码架构](architecture.md) | 内部实现详解：算法、数据流、内存管理、CUDA kernel |
| [贡献指南](contributing.md) | 编码规范、TDD 流程、PR 清单 |

## 5 分钟快速上手

```cpp
#include <plamatrix/plamatrix.h>

int main()
{
    plamatrix::MatrixXf a = plamatrix::MatrixXf::Ones(512, 512);
    plamatrix::MatrixXf b = plamatrix::MatrixXf::Identity(512, 512);
    auto result = a * b;
    return result.rows() == 512 ? 0 : 1;
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
