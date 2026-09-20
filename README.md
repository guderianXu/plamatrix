# PlaMatrix

PlaMatrix 是一个面向摄影测量、点云和稀疏优化工作负载的 C++17 矩阵库。它提供统一的 CPU/CUDA
矩阵接口，并为 OpenCL 和 Vulkan Compute 提供专用的 CSR Jacobi-PCG 路径。项目最初服务于
[PlaScan](https://github.com/guderianXu/plascan)，也可以作为独立 CMake 库集成。

> **项目状态：** 当前版本为 `0.1.0`，处于活跃开发阶段。CPU/CUDA 矩阵 API 已覆盖主要功能；
> OpenCL/Vulkan 目前聚焦稀疏迭代求解。`0.x` 阶段公开 API 仍可能随性能和下游需求调整。

## 目录

- [适用场景](#适用场景)
- [后端能力](#后端能力)
- [五分钟构建](#五分钟构建)
- [最小示例](#最小示例)
- [集成到其他项目](#集成到其他项目)
- [构建选项](#构建选项)
- [测试与基准](#测试与基准)
- [开发者指南](#开发者指南)
- [当前边界](#当前边界)
- [贡献与许可证](#贡献与许可证)

## 适用场景

PlaMatrix 适合需要以下能力的 C++ 项目：

- 列优先的 `DenseMatrix`，以及显式的 CPU/CUDA 数据传输；
- COO/CSR 稀疏矩阵、确定性 COO→CSR、SpMV、SpMM 和 CG/Jacobi-PCG；
- GEMM、LU 求解、SVD、QR、对称特征分解和批量 3×3 特征分解；
- `Vec3<T>`、Rodrigues 旋转、刚体变换、批量点变换和协方差；
- 鲁棒损失、LM 阻尼、块法方程、Schur 补和块 Jacobi-PCG；
- 可复用 CUDA workspace、异步 stream API 和 CPU OpenMP 并行；
- 在同一 CSR、初值和收敛条件下对比 OpenCL 与 Vulkan Compute。

核心设计有三个原则：

1. **数据位置显式可见。** `Device::CPU` / `Device::GPU` 是编译期类型，`toCpu()` / `toGpu()`
   明确表示传输，不隐藏昂贵拷贝。
2. **存储与数值语义稳定。** 密集矩阵采用列优先布局；稀疏结构会验证边界、排序和有限性。
3. **优化必须可复现。** 测试覆盖 CPU/GPU 一致性，基准分别报告冷启动、热路径和设备执行时间。

## 后端能力

| 后端 | 当前范围 | 主要依赖 |
|------|----------|----------|
| CPU | 密集/稀疏运算、分解、求解、点云运算、块 Schur/LM | OpenMP |
| CUDA | CPU 功能的主要 GPU 路径、cuBLAS/cuSOLVER/cuSPARSE、异步 workspace | CUDA Toolkit |
| OpenCL | GPU 枚举与运行时、CPU-owned CSR 的 float/double Jacobi-PCG、Schur 数值装配 | OpenCL 1.2 |
| Vulkan | Vulkan 1.1 Compute 运行时、CPU-owned CSR 的 float32 Jacobi-PCG | Vulkan 1.1、`glslangValidator` |

OpenCL/Vulkan 并不是通用矩阵容器后端。`DenseMatrix` / `CSRMatrix` 的持久设备类型目前仍是
CPU/CUDA；OpenCL/Vulkan 求解器在单次调用内上传 CPU-owned CSR 和向量。

## 五分钟构建

基础要求：

- CMake 3.18 或更高版本；
- 支持 C++17 的编译器；
- OpenMP；
- Google Test（仅在构建测试时需要）。

第一次构建建议从确定性的 CPU 配置开始：

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix

cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLAMATRIX_WITH_CUDA=OFF \
  -DPLAMATRIX_WITH_OPENCL=OFF \
  -DPLAMATRIX_WITH_VULKAN=OFF \
  -DPLAMATRIX_BUILD_TESTS=ON \
  -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
```

启用 CUDA：

```bash
cmake -S . -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLAMATRIX_WITH_CUDA=ON \
  -DPLAMATRIX_CUDA_ARCHITECTURES=89 \
  -DPLAMATRIX_BUILD_TESTS=ON \
  -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build build/cuda --parallel
ctest --test-dir build/cuda --output-on-failure
```

`89` 只是 Ada GPU 的示例。请按目标设备设置架构，或省略该选项使用项目默认值
`75;86;89`。完整依赖安装和平台说明见[编译指南](docs/build.md)。

## 最小示例

下面的程序只使用 CPU 路径，因此在所有构建组合下都可以运行：

```cpp
#include <iostream>

#include <plamatrix/plamatrix.h>

int main()
{
    using namespace plamatrix;

    DenseMatrix<double, Device::CPU> matrix(2, 2);
    DenseMatrix<double, Device::CPU> rhs(2, 1);
    matrix(0, 0) = 4.0;
    matrix(0, 1) = 1.0;
    matrix(1, 0) = 1.0;
    matrix(1, 1) = 3.0;
    rhs(0, 0) = 1.0;
    rhs(1, 0) = 2.0;

    const auto solution = solve(matrix, rhs);
    std::cout << solution(0, 0) << ' ' << solution(1, 0) << '\n';
}
```

启用 CUDA 后，显式传输和运算保持相同风格：

```cpp
auto matrix_gpu = matrix_cpu.toGpu();
auto product_gpu = gemm(matrix_gpu, matrix_gpu);
auto product_cpu = product_gpu.toCpu();
```

更多完整程序位于 [`docs/examples/`](docs/examples/)。

## 集成到其他项目

### 安装后使用

```bash
cmake --install build/cpu --prefix "$PWD/build/install"
```

```cmake
find_package(plamatrix 0.1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE plamatrix::plamatrix)
```

配置使用方项目时，将 `build/install` 加入 `CMAKE_PREFIX_PATH`。

### 作为源码子目录使用

```cmake
set(PLAMATRIX_BUILD_TESTS OFF CACHE BOOL "")
set(PLAMATRIX_BUILD_BENCHMARKS OFF CACHE BOOL "")
add_subdirectory(external/plamatrix)
target_link_libraries(my_target PRIVATE plamatrix::plamatrix)
```

PlaMatrix 的公开头文件位于 [`include/plamatrix/`](include/plamatrix/)，聚合入口是
`<plamatrix/plamatrix.h>`。

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_WITH_CUDA` | 检测到 Toolkit 时 `ON` | CUDA、cuBLAS、cuSOLVER 和 cuSPARSE 路径 |
| `PLAMATRIX_CUDA_ARCHITECTURES` | `75;86;89` | CUDA 目标架构列表 |
| `PLAMATRIX_WITH_OPENCL` | `ON` | OpenCL 1.2；未显式指定且依赖缺失时自动关闭 |
| `PLAMATRIX_WITH_VULKAN` | `OFF` | Vulkan 1.1 Compute 和 SPIR-V shader |
| `PLAMATRIX_USE_FLOAT` | `ON` | 构建 float32 显式实例化 |
| `PLAMATRIX_USE_DOUBLE` | `ON` | 构建 float64 显式实例化 |
| `PLAMATRIX_BUILD_TESTS` | `OFF` | 构建 Google Test 测试 |
| `PLAMATRIX_BUILD_BENCHMARKS` | `OFF` | 构建性能和后端对比工具 |

独立顶层构建兼容 `BUILD_TESTS` / `BUILD_BENCHMARKS` 短名。作为子项目时应使用带
`PLAMATRIX_` 前缀的选项，避免污染父项目配置。

运行时设备选择：

| 环境变量 | 作用 |
|----------|------|
| `PLAMATRIX_OPENCL_DEVICE_INDEX` | 选择枚举得到的 OpenCL GPU；`-1` 表示自动选择 |
| `PLAMATRIX_OPENCL_DEVICE` | 按厂商和设备名的子串选择 OpenCL GPU |
| `PLAMATRIX_VULKAN_DEVICE_INDEX` | 选择 Vulkan 物理设备，默认 `0` |
| `PLAMATRIX_VULKAN_SPMV` | `auto`、`scalar` 或 `subgroup`，用于覆盖 Vulkan SpMV 策略 |

## 测试与基准

### 测试

```bash
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
```

测试按模块拆分为密集运算、稀疏运算、分解、索引、归约、优化、OpenCL/Vulkan 运行时和集成测试。
GPU 专属测试只应在相应后端和可用设备上报告通过；跳过测试不代表该设备路径已验证。

### 通用算子基准

```bash
./build/cpu/benchmark/plamatrix_benchmark --list
./build/cpu/benchmark/plamatrix_benchmark \
  --mode cpu --size small --case gemm,spmv,pcg \
  --output build/cpu/benchmark-report.md
```

### OpenCL/Vulkan CSR-PCG 对比

```bash
cmake -S . -B build/vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLAMATRIX_WITH_CUDA=OFF \
  -DPLAMATRIX_WITH_OPENCL=ON \
  -DPLAMATRIX_WITH_VULKAN=ON \
  -DPLAMATRIX_BUILD_TESTS=ON \
  -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build build/vulkan --parallel

./build/vulkan/benchmark/plamatrix_backend_compare --suite --convergence-interval 8
./build/vulkan/benchmark/plamatrix_backend_compare \
  --matrix-market /path/to/damped-positive-definite.mtx \
  --convergence-interval 8
```

内置套件包含一维三对角、二维/三维 stencil、BA Schur 相机块图和 MVS visibility 图。
MatrixMarket 输入必须是非空方阵，并且应为正定矩阵或已完成 LM damping。

性能结论应同时记录硬件、驱动、构建类型、矩阵维度、`nnz` 和完整命令。`gpu_ms` 是 Vulkan
timestamp 得到的设备执行时间，`warm_median_ms` 是包含提交、同步和传输的热路径端到端时间；
两者不能互相替代。测量时应关闭其他 CPU/GPU 重负载，并交替运行待比较版本。

Vulkan PCG 当前会批量提交迭代，将标量递推和收敛状态保留在 GPU；完整迭代批次的 command buffer
会在后续批次和同尺寸热启动中复用，最终一级点积归约直接更新 `rho/alpha/beta` 或收敛状态。长行 CSR
可自动选择 subgroup-per-row SpMV。小解向量会在状态批次中同步回读，大解向量使用独立最终回读，
以避免多批次重复传输。对比工具中的 `command_buffer_recordings` 可用于确认录制复用是否生效。
实现和统计字段见[架构文档](docs/architecture.md)与
[编译指南](docs/build.md)。

## 开发者指南

开始修改前建议按以下顺序阅读：

1. [`docs/architecture.md`](docs/architecture.md)：模块边界、存储布局、设备路径和内存语义；
2. 对应的 [`docs/api/`](docs/api/) 文档和公开头文件；
3. 同模块测试，确认边界条件和异常语义；
4. [`docs/contributing.md`](docs/contributing.md)：代码风格、验证矩阵和 PR 清单。

项目目录：

```text
plamatrix/
├── include/plamatrix/   公开 API、模板和容器
├── src/                 CPU、CUDA、OpenCL、Vulkan 实现
├── test/                单元测试、集成测试和参考数据
├── benchmark/           通用算子与后端对比工具
├── docs/api/            按模块组织的 API 文档
├── docs/examples/       可编译示例
└── cmake/               安装包配置
```

提交改动时：

- 保持 CPU-only 构建可用，并为新增 CUDA 类型/API 提供明确的无 CUDA 行为；
- 修改公开接口、设备语义或 CMake 选项时同步更新文档；
- 数值测试使用与误差来源匹配的容差，不通过放宽容差隐藏问题；
- 性能改动同时提供正确性测试、复现命令和测量环境；
- 只格式化本次修改的代码，避免把无关重排混入功能提交。

## 当前边界

- OpenCL/Vulkan 目前只提供专用求解路径，没有通用 `DenseMatrix`、GEMM、SVD 或 QR 后端；
- Vulkan PCG 目前只支持 float32；
- CPU/OpenCL/Vulkan 与 CUDA 的设备所有权模型不同，跨后端调用会产生显式上传和下载；
- GPU 单元素 `getValue()` / `setValue()` 适合测试和调试，不应放在热循环中；
- 异步 API 要求输入、输出、workspace 和 stream 在操作完成前保持有效；
- 当前开发和性能验证以 Linux/GCC 为主，其他平台的结果应按实际环境单独报告。

## 文档

- [文档首页](docs/index.md)
- [编译指南](docs/build.md)
- [DenseMatrix API](docs/api/dense-matrix.md)
- [稀疏矩阵与迭代求解](docs/api/sparse-matrix.md)
- [线性代数 API](docs/api/linear-algebra.md)
- [非线性优化 API](docs/api/optimization.md)
- [三维向量与点云运算](docs/api/point-cloud.md)
- [性能记录](docs/performance-bottlenecks.md)

## 贡献与许可证

欢迎提交可复现的 bug 报告、测试、文档和性能改进。较大的接口或后端改动建议先创建 issue，说明使用
场景、目标设备、兼容性影响和验证计划。Pull request 应写明行为变化、测试命令，以及尚未验证的平台。
详细流程见[贡献指南](docs/contributing.md)。

PlaMatrix 使用 [MIT License](LICENSE)。
