# PlaMatrix

高性能点云处理矩阵运算库 — 支持 CPU (OpenMP) 与 CUDA GPU 加速

## 特性

- **密集矩阵运算**：加法、减法、标量乘、转置、矩阵乘法 (GEMM)
- **矩阵分解**：SVD 奇异值分解、QR 分解、对称特征值分解 (Eigh)
- **线性求解器**：基于 LU 分解的稠密线性方程组求解（多右端项）
- **稀疏矩阵**：COO 格式构建、CSR 格式存储
- **点云运算**：Rodrigues 旋转矩阵、4x4 刚体变换、批量坐标变换、协方差矩阵
- **统一接口**：模板化设计，编译期选择 `float32` / `float64` 精度
- **基准测试**：内置 CPU vs GPU 性能对比工具

## 快速开始

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
```

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

DenseMatrix<float, Device::CPU> A(3, 3);
A.fill(1.0f);
auto A_gpu = A.toGpu();
auto C_gpu = gemm(A_gpu, A_gpu);
auto C = C_gpu.toCpu();
```

## 环境要求

| 依赖 | 版本 |
|------|------|
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| GCC | 9.0+ |
| OpenMP | 4.5+ |

## 文档

- [快速入门](docs/index.md)
- [编译指南](docs/build.md)
- [DenseMatrix API](docs/api/dense-matrix.md)
- [SparseMatrix API](docs/api/sparse-matrix.md)
- [线性代数 API](docs/api/linear-algebra.md)
- [点云 API](docs/api/point-cloud.md)
- [示例代码](docs/examples/)
- [贡献指南](docs/contributing.md)

## 集成

```cmake
find_package(plamatrix REQUIRED)
target_link_libraries(my_project PRIVATE plamatrix::plamatrix)
```

## 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_USE_FLOAT` | ON | float32 实例 |
| `PLAMATRIX_USE_DOUBLE` | ON | float64 实例 |
| `PLAMATRIX_CUDA_ARCH` | native | CUDA 架构目标 |
| `BUILD_TESTS` | OFF | 构建测试 |
| `BUILD_BENCHMARKS` | OFF | 构建基准测试 |

## 许可证

MIT License — 详见 LICENSE 文件（如已包含）。
