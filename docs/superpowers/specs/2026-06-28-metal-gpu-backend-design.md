# PlaMatrix macOS Metal GPU 后端设计规格

## 1. 背景与目标

PlaMatrix 当前的 `Device::GPU` 实现绑定 CUDA/cuBLAS/cuSOLVER。这个设计在 Linux/Windows NVIDIA 环境下工作良好，但 macOS 没有 CUDA 支持，导致 Apple Silicon 和其他 Mac 用户无法使用同一套 GPU API。

本设计的目标是在不改变用户代码写法的前提下，让 macOS 上的 GPU 后端使用 Metal / Metal Performance Shaders (MPS)。用户仍然写：

```cpp
DenseMatrix<float, Device::GPU> A(rows, cols);
auto C = gemm(A, A);
```

构建系统根据平台自动选择后端：

- CUDA 可用且被选择时，继续使用现有 CUDA 后端。
- macOS 被选择为 GPU 平台时，使用新的 Metal/MPS 后端。
- GPU 被显式关闭或没有可用后端时，使用 no-GPU stub 行为。

非目标：

- 不新增 `Device::Metal`，避免下游 PlaPoint 和现有 PlaMatrix 用户改 API。
- 不改变 DenseMatrix 的列优先存储约定。
- 不弱化或重写现有 CUDA 路径。
- 不承诺 Apple GPU 上的 double 精度硬件加速；macOS double GPU API 使用 CPU/Accelerate fallback。

## 2. 后端选择语义

新增统一 CMake 选项：

```cmake
PLAMATRIX_GPU_BACKEND=AUTO|CUDA|METAL|NONE
```

默认值为 `AUTO`。

`AUTO` 选择规则：

1. macOS 优先选择 `METAL`。
2. 非 macOS 且找到 CUDA Toolkit 时选择 `CUDA`。
3. 都不可用时选择 `NONE`。

兼容现有选项：

- `PLAMATRIX_WITH_CUDA=ON` 等价于请求 CUDA 后端；找不到 CUDA 时配置期失败。
- `PLAMATRIX_WITH_CUDA=OFF` 表示自动选择时不选择 CUDA，但 macOS 仍可选择 Metal，除非显式设置 `PLAMATRIX_GPU_BACKEND=NONE` 或 `PLAMATRIX_WITH_METAL=OFF`。
- 新增 `PLAMATRIX_WITH_METAL` 作为兼容/调试开关；正常用户优先使用 `PLAMATRIX_GPU_BACKEND`。

编译定义：

| 后端 | 编译定义 |
|------|----------|
| CUDA | `PLAMATRIX_WITH_CUDA=1` |
| Metal | `PLAMATRIX_WITH_METAL=1` |
| None | `PLAMATRIX_NO_GPU=1`，并兼容现有 `PLAMATRIX_NO_CUDA=1` |

公共 API 增加后端查询：

```cpp
namespace plamatrix
{
enum class GpuBackend
{
    None,
    Cuda,
    Metal
};

GpuBackend gpuBackend();
const char* gpuBackendName();
}
```

`Device::GPU` 的语义更新为“当前构建选择的 GPU 后端”，而不是“CUDA GPU”。

## 3. 文件与模块布局

Metal 逻辑只放在 `.mm` 文件和少量 C++ 可见的不透明声明中，避免 Objective-C 类型进入公共头文件。

新增文件：

```text
include/plamatrix/core/gpu_backend.h
include/plamatrix/core/metal_context.h
src/core/gpu_backend.cpp
src/core/metal_context.mm
src/dense/dense_matrix_metal.mm
src/dense/dense_ops_metal.mm
src/ops/gemm_metal.mm
src/ops/solver_metal.mm
src/ops/point_cloud_metal.mm
src/ops/decomposition_metal.mm
test/integration/metal_backend_test.cpp
```

职责：

- `gpu_backend.h/.cpp`：提供后端枚举和名称查询，供测试、benchmark、文档示例使用。
- `metal_context.h/.mm`：管理 `MTLDevice`、`MTLCommandQueue`、运行时编译的 Metal shader library、pipeline cache、错误转换。
- `dense_matrix_metal.mm`：实现 Metal 后端的 GPU 内存分配、释放、同步/异步传输、fill、transpose。
- `dense_ops_metal.mm`：实现 Metal `add/sub` 及 async 版本。
- `gemm_metal.mm`：`float` 使用 `MPSMatrixMultiplication`；`double` 使用 CPU/Accelerate fallback。
- `solver_metal.mm`：`float` 使用 `MPSMatrixDecompositionLU` + `MPSMatrixSolveLU`；`double` 使用 CPU/Accelerate fallback。
- `point_cloud_metal.mm`：`float` 使用自定义 Metal kernels；`double` 使用 CPU fallback。
- `decomposition_metal.mm`：提供 SVD/QR/eigh 的 macOS `Device::GPU` API。首版使用 CPU/Accelerate fallback 保证全 API 可用，后续按独立优化任务实现 float Metal 专项算法。

现有 CUDA 文件保持 `.cu` 后缀和当前职责，不引入 Metal 条件分支。

## 4. 数据布局

PlaMatrix 的矩阵布局继续为列优先：

```cpp
offset = row + col * rows;
```

这条约定不变，因为它影响：

- cuBLAS/cuSOLVER 参数。
- `test/reference/` 参考数据。
- PlaPoint 下游调用。
- 现有文档和 API 示例。

Metal 后端按两类方式适配：

1. 自定义 Metal kernels 直接按 PlaMatrix 列优先布局读写，用于 `fill`、`transpose`、`add`、`sub`、点云变换和协方差。
2. MPS 操作使用内部布局适配。首版允许创建临时 row-major MPS buffer，执行 MPS kernel 后再转回 PlaMatrix column-major buffer。后续可以优化为更少临时转换，但不能改变公共布局。

MPS GEMM / solve 的首版策略优先保证正确性和 API 兼容，再通过 benchmark 评估是否需要减少布局转换。

## 5. 精度与 fallback 规则

### 5.1 float on macOS

`DenseMatrix<float, Device::GPU>` 在 macOS 上使用真实 Metal/MPS 后端：

- `fill/transpose/add/sub`：自定义 Metal kernels。
- `gemm`：MPSMatrixMultiplication。
- `solve`：MPS LU decomposition + MPS LU solve。
- `transformPoints/covarianceMatrix`：自定义 Metal kernels。
- `svd/qr/eigh`：首版 API 可用并与 CPU 结果一致；实现可以使用 CPU/Accelerate fallback，后续按专项优化移植到 Metal。

### 5.2 double on macOS

Apple MPS 当前公开矩阵 API 不提供 `MPSDataTypeFloat64`。因此 `DenseMatrix<double, Device::GPU>` 在 macOS 上遵循兼容 fallback：

1. 将 GPU 表象矩阵复制到 CPU `DenseMatrix<double, Device::CPU>`。
2. 调用现有 CPU/Accelerate 实现。
3. 将结果复制回 `DenseMatrix<double, Device::GPU>`。

这个行为必须在 README、build 文档和 API 文档中说明。它保证源码兼容和数值精度，但不是 Apple GPU double 加速。

### 5.3 容差

测试容差按后端和精度区分：

- CUDA float：保留现有容差。
- Metal float：使用 GPU float 容差，和 CPU 结果比较，不要求 bitwise 一致。
- Metal double fallback：使用 CPU double 容差。
- SVD/QR/eigh：如果首版走 CPU fallback，应与 CPU 路径使用同等容差；后续 Metal float 优化另行更新容差依据。

## 6. 异常与错误处理

配置期错误：

- `PLAMATRIX_GPU_BACKEND=CUDA` 但找不到 CUDA Toolkit：CMake fatal error。
- `PLAMATRIX_GPU_BACKEND=METAL` 但不是 macOS 或找不到 Metal/MPS framework：CMake fatal error。
- 同时强制 CUDA 和 Metal：CMake fatal error。

运行期错误：

- Metal 后端启用但 `MTLCreateSystemDefaultDevice()` 返回空：抛 `std::runtime_error("Metal GPU backend requested but no MTLDevice is available")`。
- Metal shader 编译失败：错误信息包含函数名和 Metal 编译日志。
- MPS kernel 编码或执行失败：转换为 `std::runtime_error`，包含操作名。
- no-GPU 后端调用 GPU 算法：抛明确异常，语义替代当前 no-CUDA stub。

新增 Metal 检查工具：

```cpp
#define PLAMATRIX_CHECK_METAL(call)
```

如果 Objective-C API 以 `NSError**` 返回错误，则 `.mm` 内部 helper 将 `NSError.localizedDescription` 转换为 C++ 异常。公共 C++ 头不暴露 Objective-C 类型。

## 7. OpenMP 与 macOS 构建

当前项目 `find_package(OpenMP REQUIRED)` 在只安装 Command Line Tools 的 macOS 上会失败。为了让 macOS Metal 后端能开箱配置，新增：

```cmake
PLAMATRIX_WITH_OPENMP=AUTO|ON|OFF
```

默认 `AUTO`：

- 找到 OpenMP 时链接 `OpenMP::OpenMP_CXX` 并启用现有 OpenMP 路径。
- 找不到 OpenMP 时继续配置，CPU fallback 使用串行循环。
- `ON` 表示强制要求 OpenMP，找不到则配置失败。
- `OFF` 表示不查找 OpenMP。

CPU 代码中的 OpenMP include 和 pragma 需要受 `PLAMATRIX_WITH_OPENMP` 保护，避免 macOS 无 `libomp` 时编译失败。

## 8. 构建系统变更

顶层 `CMakeLists.txt` 调整：

- 在 CUDA 探测前解析 `PLAMATRIX_GPU_BACKEND`。
- macOS `AUTO` 优先尝试 Metal/MPS。
- CUDA 后端继续 `enable_language(CUDA)`，使用现有 `.cu` 文件。
- Metal 后端启用 Objective-C++ 源文件，链接：

```cmake
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
```

- Metal 后端 target sources 添加 `.mm` 文件。
- `plamatrixConfig.cmake.in` 导出后端依赖：CUDA 后端导出 `find_dependency(CUDAToolkit)`；Metal 后端记录 framework 链接。

## 9. 测试策略

新增和扩展测试：

1. 后端配置测试
   - macOS `AUTO` 返回 `gpuBackendName() == "metal"`。
   - `PLAMATRIX_GPU_BACKEND=NONE` 返回 `"none"`。
   - CUDA 构建返回 `"cuda"`。

2. DenseMatrix GPU 测试
   - 构造、零尺寸矩阵、toGpu/toCpu round-trip。
   - `fill(0)`、`fill(nonzero)`。
   - `transpose()` 与 CPU 对比。
   - async API 在 Metal 后端上可以是同步兼容实现，但必须保持调用语义和结果正确。

3. Dense ops 测试
   - `add/sub/addAsync/subAsync` 与 CPU 对比。
   - 维度不匹配异常。

4. GEMM 测试
   - float Metal MPS 结果与 CPU 对比。
   - double macOS fallback 结果与 CPU 对比。
   - 零尺寸和维度不匹配。

5. Solver 测试
   - float Metal LU solve 与 CPU 对比。
   - double fallback 与 CPU 对比。
   - 奇异矩阵异常。

6. Decomposition 测试
   - macOS GPU SVD/QR/eigh API 可用。
   - 结果与 CPU 路径一致。
   - 文档和测试名称说明首版可能使用 CPU fallback。

7. Point cloud 测试
   - rotationMatrix、rigidTransform、transformPoints、covarianceMatrix 与 CPU 对比。
   - float 使用 Metal kernels。
   - double fallback 与 CPU 对比。

验证命令：

```bash
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=METAL
cmake --build build-metal -j$(sysctl -n hw.ncpu)
ctest --test-dir build-metal --output-on-failure

cmake -S . -B build-none -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=NONE
cmake --build build-none -j$(sysctl -n hw.ncpu)
ctest --test-dir build-none --output-on-failure
```

CUDA 回归命令保持：

```bash
cmake -S . -B build -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_GPU_BACKEND=CUDA
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## 10. 文档更新

必须同步更新：

- `README.md`
- `docs/build.md`
- `docs/architecture.md`
- `docs/api/dense-matrix.md`
- `docs/api/linear-algebra.md`
- `docs/api/point-cloud.md`
- `docs/contributing.md`

文档需要明确：

- `Device::GPU` 是平台 GPU 后端，不再等同于 CUDA。
- macOS 默认使用 Metal/MPS。
- macOS double GPU API 是 CPU/Accelerate fallback。
- macOS SVD/QR/eigh 首版以 API 可用和正确性为目标，具体 GPU 加速状态按文档列出。
- CUDA 用户原有构建方式继续可用。
- macOS 如果需要 OpenMP 并行 CPU fallback，可安装 `libomp`；没有 OpenMP 时仍可构建。

## 11. 迁移与兼容性

用户代码兼容性：

- `DenseMatrix<Scalar, Device::GPU>` 不变。
- `toGpu()` / `toCpu()` 不变。
- `gemm/add/sub/svd/qr/eigh/solve/point_cloud` API 不变。
- CUDA stream 参数在 Metal 后端中保持 API 兼容。首版可将 `cudaStream_t` 在 no-CUDA/Metal 构建中定义为 opaque stub 类型，Metal 后端内部使用自己的 command queue；后续如需要可新增平台中立 stream/queue API。

二进制兼容性：

- 这是源码级兼容设计，不承诺旧构建产物 ABI 兼容。
- 安装包导出必须体现实际后端依赖，避免下游 `find_package(plamatrix)` 后链接失败。

## 12. 分阶段交付

### 里程碑 1：构建与后端选择

- 新增 `PLAMATRIX_GPU_BACKEND`。
- 新增 `gpuBackendName()`。
- macOS 可配置 Metal 后端。
- OpenMP 改为可选。
- no-GPU stub 继续工作。

### 里程碑 2：Metal Dense 基础

- Metal allocator/context。
- `DenseMatrix` GPU 构造、传输、fill、transpose。
- `add/sub`。
- 对应测试通过。

### 里程碑 3：MPS GEMM 与 solver

- float `gemm` 使用 MPSMatrixMultiplication。
- float `solve` 使用 MPS LU。
- double fallback。
- 对应 CPU/GPU 一致性测试通过。

### 里程碑 4：点云 Metal kernels

- float transformPoints 和 covarianceMatrix 使用 Metal kernels。
- double fallback。
- point cloud workflow 测试通过。

### 里程碑 5：分解 API 全量可用

- macOS GPU SVD/QR/eigh API 可用。
- 首版允许 CPU/Accelerate fallback。
- 文档标注 fallback 状态。

### 后续优化

- 评估 GEMM/solve 布局转换成本。
- 为 SVD/QR/eigh 设计独立 float Metal 算法或更细的 MPS/Accelerate 混合路径。
- benchmark 区分 CUDA、Metal、CPU fallback。

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| MPS 矩阵 API 与 PlaMatrix 列优先布局不一致 | GEMM/solve 需要额外转换 | 首版用内部临时 buffer 保证正确性，后续 benchmark 驱动优化 |
| Apple GPU 不支持 double MPS 矩阵 | double 无真实 GPU 加速 | 明确 CPU/Accelerate fallback，测试与文档透明说明 |
| SVD/QR/eigh 缺少直接 MPS kernel | 全量 GPU 加速工程量大 | 首版 API fallback，后续独立优化 |
| macOS 无 OpenMP 导致配置失败 | 用户无法构建 | 新增 optional OpenMP |
| CUDA 路径被新后端影响 | 现有用户回归 | CUDA `.cu` 文件保持独立，新增后端互斥选择和 CUDA 回归测试 |

## 14. 验收标准

首版完成时必须满足：

- macOS 上 `cmake -DPLAMATRIX_GPU_BACKEND=METAL -DPLAMATRIX_BUILD_TESTS=ON` 可配置并构建。
- macOS 上 `DenseMatrix<float, Device::GPU>` 的基础运算、GEMM、solve、点云热路径可运行并通过 CPU/GPU 一致性测试。
- macOS 上 `DenseMatrix<double, Device::GPU>` 的公开 API 可运行，并通过 CPU fallback 一致性测试。
- `PLAMATRIX_GPU_BACKEND=NONE` 仍能构建 no-GPU stub。
- CUDA 构建命令继续可用，CUDA 源文件和 API 行为不被 Metal 改动破坏。
- README 和 docs 明确说明后端选择、double fallback、SVD/QR/eigh 首版状态。
