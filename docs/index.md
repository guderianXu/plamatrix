# PlaMatrix 快速入门

## 简介

PlaMatrix 是一个面向点云处理的高性能矩阵运算库，支持 CPU 多线程 (OpenMP) 和 CUDA GPU 加速。所有矩阵运算提供 CPU 和 GPU 两种后端，用户可通过统一的模板接口在设备间切换。

### 核心特性

- **密集矩阵运算**：基础线性代数（加法、减法、标量乘）、矩阵乘法 (GEMM)、转置
- **矩阵分解**：SVD 奇异值分解、QR 分解、对称矩阵特征值分解 (Eigh)
- **线性求解器**：基于 LU 分解的稠密线性方程组求解
- **稀疏矩阵**：COO 格式构建、CSR 格式存储
- **点云专用**：Rodrigues 旋转矩阵、刚体变换、点云坐标变换、协方差矩阵
- **统一基准测试**：CPU vs GPU 性能对比工具
- **双精度支持**：模板化设计，编译期选择 float32 / float64

## 5 分钟上手

### 环境要求

| 依赖 | 最低版本 |
|------|----------|
| CUDA Toolkit | 11.0+ |
| CMake | 3.18+ |
| GCC | 9.0+ |
| OpenMP | 4.5+ |
| Google Test | 1.10+ (仅测试) |

### 编译

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
```

### 可选编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_USE_FLOAT` | ON | 预编译 float32 实例 |
| `PLAMATRIX_USE_DOUBLE` | ON | 预编译 float64 实例 |
| `PLAMATRIX_CUDA_ARCH` | native | CUDA 架构目标（如 `70;75;80`） |
| `BUILD_TESTS` | OFF | 构建单元测试 |
| `BUILD_BENCHMARKS` | OFF | 构建基准测试 |
| `BUILD_DOCS` | OFF | 构建文档 |

### 第一个程序

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    // 创建 CPU 端 3x3 矩阵
    DenseMatrix<float, Device::CPU> A(3, 3);
    A.setValue(0, 0, 1.0f);
    A.setValue(1, 1, 2.0f);
    A.setValue(2, 2, 3.0f);

    // 转移到 GPU 进行计算
    auto A_gpu = A.toGpu();
    auto B_gpu = A_gpu;  // 假设 B = A
    auto C_gpu = gemm(A_gpu, B_gpu);

    // 取回结果
    auto C = C_gpu.toCpu();

    // 输出 C(0, 0) = A(0,0) * A(0,0) = 1.0
    std::cout << C(0, 0) << std::endl;
}
```

### 集成到 CMake 项目

PlaMatrix 提供了标准的 CMake 包配置文件。安装后，在 `CMakeLists.txt` 中添加：

```cmake
find_package(plamatrix REQUIRED)
target_link_libraries(my_project PRIVATE plamatrix::plamatrix)
```

如果使用本地构建的版本（未安装），可以通过设置 `CMAKE_PREFIX_PATH` 或直接使用 `add_subdirectory`：

```cmake
add_subdirectory(path/to/plamatrix)
target_link_libraries(my_project PRIVATE plamatrix::plamatrix)
```

### 运行测试

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
ctest --output-on-failure
```

### 运行基准测试

```bash
cd build && cmake .. -DBUILD_BENCHMARKS=ON && cmake --build . -j$(nproc)
./benchmark/plamatrix_benchmark
```

## 文档导航

| 文档 | 说明 |
|------|------|
| [编译指南](build.md) | 详细编译配置和平台说明 |
| [DenseMatrix API](api/dense-matrix.md) | 密集矩阵构造与操作 |
| [SparseMatrix API](api/sparse-matrix.md) | COO/CSR 稀疏矩阵 |
| [线性代数 API](api/linear-algebra.md) | GEMM、分解、线性求解 |
| [点云 API](api/point-cloud.md) | 旋转矩阵、刚体变换、协方差 |
| [示例代码](examples/) | 完整可编译示例 |
| [贡献指南](contributing.md) | 编码规范和开发流程 |
