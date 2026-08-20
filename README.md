# PlaMatrix

面向点云处理的高性能矩阵运算库，支持 CPU 多线程 (OpenMP)、CUDA GPU 加速，以及可选的 OpenCL
运行时与通用执行资源。

## 特性

- **密集矩阵**：矩阵乘法、逐元素加减乘除、标量变换、绝对值、平方根、截断和转置
- **归约与索引**：`sum/mean/min/max/argMin/argMax`、exclusive scan、按行 gather/scatter/compact
- **矩阵分解**：原生 CPU SVD、QR、对称特征值，GPU 使用 cuSOLVER
- **批量小矩阵**：CPU/CUDA 对称 3x3 特征分解，稳定的 8-sweep Jacobi 和重复特征空间基
- **线性求解**：稠密 LU/cuSOLVER，以及 CPU/CUDA CSR 和 CPU-owned CSR OpenCL 上的 CG/Jacobi-PCG
- **稀疏矩阵**：确定性 COO→CSR、CPU/CUDA 传输、cuSPARSE SpMV/SpMM 和可复用 workspace
- **小向量数学**：`Vec3<T>` 算术、数组转换、点积、叉积、范数、归一化和有限性检查
- **点云专用**：Rodrigues 旋转矩阵、4×4 刚体变换、批量点变换、协方差矩阵
- **双精度**：模板化 `float` / `double`，编译期设备绑定 `Device::CPU` / `Device::GPU`
- **OpenCL 执行与稀疏求解**：GPU 枚举与选择、共享 context、queue/buffer/kernel RAII、program cache，
  以及一次上传 CPU-owned CSR 系统的 Jacobi-PCG
- **通用块优化**：Huber、二分块法方程、LM 阻尼、可复用 Schur CSR pattern，
  多 primary 残差与直接交叉块，以及 CPU/CUDA/OpenCL 块 Jacobi-PCG；CUDA/OpenCL 在设备端装配
  Schur 数值
- **统一基准测试**：一键运行三层测试 (串行 / OpenMP / CUDA)，自动生成 Markdown 性能报告

> `DenseMatrix` / `CSRMatrix` 的持久设备语义仍是 CPU/CUDA；OpenCL PCG 接受 CPU-owned CSR 和向量，
> 在一次调用内上传并求解。GEMM、SVD 和通用 OpenCL 矩阵容器尚未提供。

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
| `PLAMATRIX_WITH_OPENCL` | `ON` | 启用 OpenCL 执行基础；OpenCL 1.2 SDK/loader 未找到且未显式要求时自动关闭 |
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

# 稀疏转换、乘法和迭代求解专项
./benchmark/plamatrix_benchmark --mode all --size smoke \
  --case coo_to_csr,spmv,spmm,cg,pcg
```

CUDA 算子基准复用输出矩阵和 workspace，并分别记录冷分配、热 workspace、
CUDA event/求解总时间和传输时间。自适应 CG/PCG 包含主机收敛检查，因而
`kernel_only_ms` 对这两行表示完整 GPU 求解时间。

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
auto C = gemm(A, B);     // 矩阵乘法 (原生 CPU / cuBLAS)
auto D = add(A, B);      // 逐元素加法
auto H = hadamardMultiply(A, B); // 逐元素乘法；A/B 必须完全同形，不做广播
auto M = mean(A, ReductionAxis::Columns); // 按行归约，得到 1 x A.cols()
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

// 每行 [xx, xy, xz, yy, yz, zz]，返回 N x 3 特征值和 N x 9 特征向量
auto eig3 = symmetricEigh3x3Batched(compact_symmetric_matrices);
```

### 索引与紧缩
```cpp
auto offsets = exclusiveScan(counts);       // 按列优先线性顺序扫描
auto picked = gatherRows(points, indices);  // 保留 indices 顺序和重复项
scatterRows(values, indices, output);       // 重复目标由最低源行获胜
auto compacted = compactRows(points, mask); // 非零 mask，稳定返回精确行数
```

GPU 同步重载会等待给定 stream 并报告设备端错误。异步 indexing 和批量 3x3 特征分解使用
调用方持有的 workspace；等待 stream 后必须调用对应 `checkStatus()`。归约异步接口没有
`checkStatus()`，但同样要求输入、输出和 workspace 在 stream 完成前保持有效。

### 点云运算
```cpp
Vec3<double> a(std::array<double, 3>{1.0, 2.0, 3.0});
Vec3<double> b{4.0, 5.0, 6.0};
auto unit_normal = normalized(cross(a, b), 1.0e-12);
bool valid = isFinite(unit_normal);

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

会分配返回值的部分异步归约和索引 API 使用 stream-ordered GPU 内存。此类矩阵保留创建
stream 的所有权信息；stream 完成后应在销毁 stream 前调用 `closeAsyncAllocation()`。
移动矩阵会转移该规则，移动后的源对象变为 `0 x 0`。

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
- [非线性优化 API](docs/api/optimization.md)
- [三维向量与点云运算 API](docs/api/point-cloud.md)
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
