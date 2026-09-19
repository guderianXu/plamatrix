# PlaMatrix

面向点云处理的高性能矩阵运算库，支持 CPU 多线程 (OpenMP)、CUDA GPU 加速，以及可选的 OpenCL
和 Vulkan Compute 运行时。

## 特性

- **密集矩阵**：矩阵乘法、逐元素加减乘除、标量变换、绝对值、平方根、截断和转置
- **归约与索引**：`sum/mean/min/max/argMin/argMax`、exclusive scan、按行 gather/scatter/compact
- **矩阵分解**：原生 CPU SVD、QR、对称特征值，GPU 使用 cuSOLVER
- **批量小矩阵**：CPU/CUDA 对称 3x3 特征分解，稳定的 8-sweep Jacobi 和重复特征空间基
- **线性求解**：稠密 LU/cuSOLVER、原生块稀疏 Cholesky，以及 CPU/CUDA CSR 和 CPU-owned CSR
  OpenCL 上的 CG/Jacobi-PCG
- **稀疏矩阵**：确定性 COO→CSR、CPU/CUDA 传输、cuSPARSE SpMV/SpMM 和可复用 workspace
- **小向量数学**：`Vec3<T>` 算术、数组转换、点积、叉积、范数、归一化和有限性检查
- **点云专用**：Rodrigues 旋转矩阵、4×4 刚体变换、批量点变换、协方差矩阵
- **双精度**：模板化 `float` / `double`，编译期设备绑定 `Device::CPU` / `Device::GPU`
- **OpenCL 执行与稀疏求解**：GPU 枚举与选择、共享 context、queue/buffer/kernel RAII、program cache，
  以及一次上传 CPU-owned CSR 系统的 Jacobi-PCG
- **Vulkan Compute 稀疏求解**：可选 Vulkan 1.1 Compute runtime、SPIR-V shader 和 float32 CPU-owned CSR
  Jacobi-PCG，可与 OpenCL 使用相同输入和收敛条件进行对比
- **通用块优化**：Huber、二分块法方程、LM 阻尼、可复用 Schur CSR pattern 和块图最小度符号分析，
  多 primary 残差与直接交叉块，以及 CPU/CUDA/OpenCL 块 Jacobi-PCG；CUDA/OpenCL 在设备端装配
  Schur 数值，CPU 可选择原生稀疏或稠密直接求解
- **统一基准测试**：一键运行三层测试 (串行 / OpenMP / CUDA)，自动生成 Markdown 性能报告

> `DenseMatrix` / `CSRMatrix` 的持久设备语义仍是 CPU/CUDA；OpenCL/Vulkan PCG 接受 CPU-owned CSR
> 和向量，在一次调用内上传并求解。GEMM、SVD 和通用 OpenCL/Vulkan 矩阵容器尚未提供。

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
| `PLAMATRIX_WITH_VULKAN` | `OFF` | 启用 Vulkan 1.1 Compute；需要 Vulkan SDK/loader 和 `glslangValidator` |
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

# OpenCL/Vulkan 相同 CSR-PCG 工作负载对比（参数为矩阵规模）
./benchmark/plamatrix_backend_compare 4096
# 多规模大矩阵套件
./benchmark/plamatrix_backend_compare --suite
# 自定义规模
./benchmark/plamatrix_backend_compare --sizes 4096,16384,65536,262144,1048576
# 二维五点、三维七点 stencil
./benchmark/plamatrix_backend_compare --case stencil2d --sizes 64,128,256
./benchmark/plamatrix_backend_compare --case stencil3d --sizes 16,24,32
# BA Schur 相机块拓扑（参数为相机数量，每个相机 6 个变量）
./benchmark/plamatrix_backend_compare --case ba_schur --sizes 32,64,128,256
# MVS 空间邻域 + 跨视图可见性拓扑（参数为影像边长）
./benchmark/plamatrix_backend_compare --case mvs_visibility --sizes 64,128,256
# 直接比较真实的、已经阻尼为正定的 CSR MatrixMarket 文件
./benchmark/plamatrix_backend_compare --matrix-market /path/to/ba_schur.mtx
```

`--suite` 会依次运行一维三对角、二维/三维 stencil、BA Schur 相机块图和 MVS visibility 图，
输出中的 `dimension` 是展开后的标量维度，`nnz` 是实际 CSR 非零元数量。`ba_schur` 的参数是相机数，
每个相机展开为 6 个标量变量；其它二维/三维场景的参数是网格边长。内置 BA/MVS 场景保持了 PlaScan
中常见的稀疏拓扑和不规则行长度，所有内置场景使用确定性的非均匀右端项，避免全 1 向量形成过于容易的
特征方向；矩阵数值仍是确定性合成值。要测量真实工程数据，把已经完成阻尼、适合
Jacobi-PCG 的 CSR 矩阵导出为 MatrixMarket coordinate real general/symmetric 文件后使用
`--matrix-market`。当前 MatrixMarket 输入必须是非空方阵，且矩阵应为正定或经过 LM damping。
PlaScan 当前 MVS 主链不持久化 CSR 文件，因此 `mvs_visibility` 用的是同样的空间邻域和跨视图连接形态；
真实 MVS 数据应通过 MatrixMarket 入口接入。

该基准还会输出 Vulkan 的 `command_submissions`、`gpu_ms`、`barrier_count` 和
`cold_descriptor_set_allocations`，分别用于观察显式 queue submit/fence wait、GPU 实际执行时间、记录的
buffer 依赖屏障数量以及冷启动时 descriptor set 的创建数量；OpenCL 的该列固定为 `0`，仅表示当前未暴露
同等统计。`cold_ms` 包含首次工作区和 shader 路径，`warm_median_ms` 是工作区复用后的端到端 PCG 时间。
使用 `--convergence-interval N` 可以批量执行 N 次 PCG 迭代后再检查收敛；`gpu_ms` 与 `warm_median_ms`
的差值可用于区分 GPU 执行和主机提交/等待开销。

CUDA 算子基准复用输出矩阵和 workspace，并分别记录冷分配、热 workspace、
CUDA event/求解总时间和传输时间。自适应 CG/PCG 包含可批量执行的主机收敛检查，因而
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
