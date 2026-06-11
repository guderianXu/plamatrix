# PlaMatrix

面向点云处理的高性能矩阵运算库，支持 CPU 多线程 (OpenMP) 和 CUDA GPU 加速。

## 特性

- **密集矩阵**：矩阵乘法、逐元素加减、转置、CPU 标量运算
- **矩阵分解**：SVD、QR、对称特征值 (SVD/eigh 可选 LAPACK，QR CPU fallback，GPU cuSOLVER)
- **线性求解**：稠密求解器 (LU 分解 + cuSOLVER getrf/getrs)
- **稀疏矩阵**：COO / CSR 格式，COO→CSR 转换
- **点云专用**：Rodrigues 旋转矩阵、4×4 刚体变换、批量点变换、协方差矩阵
- **双精度**：模板化 `float` / `double`，编译期设备绑定 `Device::CPU` / `Device::GPU`
- **统一基准测试**：一键运行三层测试 (串行 / OpenMP / CUDA)，自动生成 Markdown 性能报告

## 快速开始

**新电脑从零搭建？** 先看 [编译指南](docs/build.md)，包含 CUDA 驱动安装、CMake 升级、Google Test 安装、CPU-only 构建等完整步骤。

### 编译

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
mkdir build && cd build
cmake .. -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
```

**无 NVIDIA GPU？** 加 `-DPLAMATRIX_WITH_CUDA=OFF` 即可 CPU-only 编译。

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_WITH_CUDA` | 自动检测 | 启用 CUDA GPU 加速 |
| `PLAMATRIX_CUDA_ARCHITECTURES` | `75;86;89` | CUDA 计算能力目标 |
| `PLAMATRIX_WITH_SYSTEM_LINALG` | `ON` | 通过 CMake 检测并使用系统 BLAS/LAPACK |
| `PLAMATRIX_USE_FLOAT` | `ON` | 启用 float32 支持 |
| `PLAMATRIX_USE_DOUBLE` | `ON` | 启用 float64 支持 |
| `PLAMATRIX_BUILD_TESTS` | `OFF` | 构建单元测试 |
| `PLAMATRIX_BUILD_BENCHMARKS` | `OFF` | 构建性能基准测试 |

独立顶层构建时仍兼容 `BUILD_TESTS` / `BUILD_BENCHMARKS` 短名；作为子项目集成时优先使用 `PLAMATRIX_BUILD_*` 选项。

### 第一个程序

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

int main()
{
    // 创建 1000×1000 的 CPU 矩阵
    DenseMatrix<float, Device::CPU> A(1000, 1000);
    A.fill(1.0f);

    // 转移到 GPU，执行矩阵乘法
    auto A_gpu = A.toGpu();
    auto C_gpu = gemm(A_gpu, A_gpu);

    // 取回 CPU
    auto C = C_gpu.toCpu();
    return 0;
}
```

### 集成到你的项目

**方式一：安装后 find_package**
```bash
cd build && cmake --install . --prefix /your/install/path
```
```cmake
find_package(plamatrix REQUIRED)
target_link_libraries(my_project plamatrix::plamatrix)
```

**方式二：直接 add_subdirectory**
```cmake
add_subdirectory(plamatrix)
target_link_libraries(my_project plamatrix::plamatrix)
```

## 性能基准

```bash
# CPU 对比 (串行 vs 多线程)
./benchmark/plamatrix_benchmark --mode cpu --size medium

# 完整对比 (CPU + GPU)
./benchmark/plamatrix_benchmark --mode all --size large --output report.md

# 快速 smoke 或只跑指定 case
./benchmark/plamatrix_benchmark --mode cpu --size smoke --case gemm,covariance
```

CUDA 模式下 GEMM/add/sub 的计算时间使用 CUDA event 计时，并复用输出矩阵；
输入传输时间使用 pinned host buffer，仍单独记录。

| 档位 | 矩阵尺寸 |
|------|----------|
| smoke/tiny | 16, 32 |
| small | 256, 512, 1024, 2048 |
| medium | 1024, 2048, 4096, 8192 |
| large | 4096, 8192, 12288, 16384 |

## API 概览

### 矩阵类型
```cpp
DenseMatrix<float, Device::CPU>  A(rows, cols);   // CPU 密集矩阵
DenseMatrix<float, Device::GPU>  B(rows, cols);   // GPU 密集矩阵
COOMatrix<float, Device::CPU>    coo(rows, cols); // COO 稀疏矩阵
CSRMatrix<float, Device::CPU>    csr(rows, cols, nnz); // CSR 稀疏矩阵
```

### 基本运算
```cpp
auto C = gemm(A, B);     // 矩阵乘法 (cuBLAS / BLAS / CPU fallback)
auto D = add(A, B);      // 逐元素加法
auto E = A.transpose();  // 转置
auto F = add(2.0f * A, B); // CPU 标量乘加

// GPU 热循环可复用输出矩阵并异步提交
cudaStream_t stream = nullptr;
DenseMatrix<float, Device::GPU> C_gpu(A_gpu.rows(), B_gpu.cols());
gemmAsync(A_gpu, B_gpu, C_gpu, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
```

### 高级运算
```cpp
auto [U, S, Vt] = svd(A);           // 奇异值分解
auto [Q, R] = qr(A);                // QR 分解
auto eig = eigh(A);                 // 对称特征值
auto X = solve(A, b);               // 线性求解 Ax = b
```

### 点云运算
```cpp
auto R = rotationMatrix(axis, angle);    // Rodrigues 旋转矩阵
auto T = rigidTransform(R, translation); // 4×4 刚体变换
auto pts_t = transformPoints(T, points); // 批量点变换
auto cov = covarianceMatrix(points);     // 协方差矩阵

// GPU 点云热循环可复用输出矩阵并异步提交
DenseMatrix<float, Device::GPU> pts_out(pts_gpu.rows(), 3);
DenseMatrix<float, Device::GPU> cov_out(3, 3);
GpuCovarianceWorkspace<float> cov_workspace;
transformPointsAsync(T_gpu, pts_gpu, pts_out, stream);
covarianceMatrixAsync(pts_gpu, cov_out, cov_workspace, stream);
```

### 设备传输
```cpp
auto A_gpu = A_cpu.toGpu();  // CPU → GPU (触发 cudaMemcpy)
auto A_cpu = A_gpu.toCpu();  // GPU → CPU (触发 cudaMemcpy)
auto pinned = DenseMatrix<float, Device::CPU>::pinned(A_cpu.rows(), A_cpu.cols());
auto B_gpu = pinned.toGpuAsync(stream); // 异步传输，调用方负责同步 stream
```

高频 GPU 临时矩阵可显式开启内存池，减少同尺寸 `DenseMatrix<Device::GPU>` 反复分配成本：

```cpp
GpuAllocator<float>::setMemoryPoolEnabled(true);
// ... GPU pipeline / benchmark ...
GpuAllocator<float>::releaseMemoryPool();
GpuAllocator<float>::setMemoryPoolEnabled(false);
```

## 文档

- [快速入门](docs/index.md)
- [编译指南](docs/build.md)
- [DenseMatrix API](docs/api/dense-matrix.md)
- [稀疏矩阵 API](docs/api/sparse-matrix.md)
- [线性代数 API](docs/api/linear-algebra.md)
- [点云运算 API](docs/api/point-cloud.md)
- [贡献指南](docs/contributing.md)

## 项目结构

```
plamatrix/
├── include/plamatrix/     # 头文件 (模板 + 声明)
│   ├── core/              # 基础类型、内存分配、错误处理
│   ├── dense/             # DenseMatrix + 基本运算
│   ├── sparse/            # COOMatrix / CSRMatrix
│   └── ops/               # gemm, svd, solve, 点云运算
├── src/                   # 源文件 (.cpp / .cu)
├── test/                  # 单元测试 + 集成测试
├── benchmark/             # 统一基准测试程序
└── docs/                  # 中文文档
```

## 许可证

MIT
