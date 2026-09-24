# PlaMatrix

PlaMatrix 是一个面向摄影测量、点云和稀疏优化工作负载的 C++17 矩阵库。它提供 Eigen 风格的
`Matrix`/`Vector` 密集矩阵 API，可按运算、标量类型与数据位置自动选择 CPU/CUDA/Vulkan/OpenCL；
提供专用的 CSR Jacobi-PCG 路径。项目最初服务于
[PlaScan](https://github.com/guderianXu/plascan)，也可以作为独立 CMake 库集成。

> **项目状态：** 当前版本为 `0.1.0`，处于活跃开发阶段。CPU/CUDA 矩阵 API 已覆盖主要功能；
> OpenCL/Vulkan 同时覆盖基础稠密算术、通用 QR/SVD/自伴特征分解与稀疏迭代求解。
> `0.x` 阶段公开 API 仍可能随性能和下游需求调整。

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

应用入口提供 `MatrixXd`、`Vector3d`、固定尺寸块、`Map`、延迟算术/数组/轴向表达式，以及 LU、LDLT、QR、SVD 和特征分解。Geometry 入口提供 Eigen 5.0.1 同名的 `Quaternion`、`AngleAxis`、`Transform`、`Translation`、欧拉角和 `umeyama()`。稀疏入口提供 `Triplet`、`SparseMatrix`、`SparseView`、稀疏直接/迭代求解器、预条件器和 AMD/COLAMD/Natural ordering。

日常密集运算自动选择 CPU/CUDA/Vulkan/OpenCL，无需用户管理设备、流或传输。公开核心类型使用 Eigen 5
同形的六参数 `Matrix`、`EigenBase`/`DenseBase`/`MatrixBase`/`ArrayBase`、三参数 `Map`
和 `Ref`。当前数值能力与分解限制见 API 文档；尚未实现的 Eigen 模块仍不属于兼容范围。

执行上下文、驻留存储、GPU 原语及块 Schur 数值实现均位于 `plamatrix::internal`，仅供模块后端集成。BA 业务由 PlaBundle 负责，点云体素归组由 PlaPoint 负责。旧 `DenseMatrix`/`CSRMatrix`/`Vec3` 等公开类型和旧头文件路径已删除，不提供兼容层。

## 后端能力

| 后端 | 当前范围 | 主要依赖 |
|------|----------|----------|
| CPU | 密集/稀疏运算、分解、求解、点云运算、块 Schur/LM | OpenMP |
| CUDA | CPU 功能的主要 GPU 路径、cuBLAS/cuSOLVER/cuSPARSE、异步 workspace | CUDA Toolkit |
| OpenCL | float/double 基础稠密算术、Householder QR、Jacobi/BDC SVD、自伴特征分解、resident CSR-PCG、Schur 数值装配 | OpenCL 1.2 |
| Vulkan | float 基础稠密算术、Householder QR、Jacobi/BDC SVD、自伴特征分解、resident CSR-PCG、可选 cooperative-matrix 与设备常驻 Schur 路径 | Vulkan 1.1、`glslangValidator` |

表中列出的是库内部能力，并非每项都有公开自动调度入口。显式设备执行和生命周期约束见[后端实现接口](docs/api/backend-internals.md)。

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

下面的程序展示默认矩阵 API。小矩阵在 CPU 上计算；较大的矩阵乘法会自动选择可用 GPU：

```cpp
#include <iostream>

#include <plamatrix/plamatrix.h>

int main()
{
    plamatrix::Vector3d point(1.0, 2.0, 3.0);
    plamatrix::Matrix3d rotation;
    rotation << 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0;
    plamatrix::MatrixXd measurements = plamatrix::MatrixXd::Zero(100, 3);
    plamatrix::VectorXd weights = plamatrix::VectorXd::Ones(100);

    plamatrix::Vector3d rotated = rotation * point;
    std::cout << rotated(0) << ' ' << measurements.rows() << ' ' << weights(0) << '\n';
}
```

默认自动策略不要求调用方显式搬移 `Matrix` 数据，并在达到各后端实测工作量门槛后按
CUDA → OpenCL → Vulkan 的顺序选择兼容设备，让连续运算复用已驻留的输入。Vulkan 稠密算术当前支持
`float`，OpenCL 支持 `float` 以及具备 FP64
能力设备上的 `double`。非 const `data()` 或系数写入会使设备缓存失效。

更多完整程序位于 [`docs/examples/`](docs/examples/)。
求解器的适用条件、视图和显式 `noalias()` 用法见
[`docs/api/linear-algebra.md`](docs/api/linear-algebra.md) 与
[`docs/api/dense-matrix.md`](docs/api/dense-matrix.md)。稀疏矩阵装配与求解示例见
[`docs/api/sparse-matrix.md`](docs/api/sparse-matrix.md)，旋转、位姿和相似变换见
[`docs/api/geometry.md`](docs/api/geometry.md)。

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
`<plamatrix/plamatrix.h>`。`internal/` 内头文件仅供后端集成，不属于公开兼容性契约。

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
| `PLAMATRIX_WITH_EIGEN_REFERENCE` | `OFF` | 构建可选的 Eigen 5.0+ 差分测试和同进程性能基准 |
| `PLAMATRIX_EIGEN3_INCLUDE_DIR` | 空 | Eigen 5.0+ 未提供 CMake package 时的头文件目录 |

独立顶层构建兼容 `BUILD_TESTS` / `BUILD_BENCHMARKS` 短名。作为子项目时应使用带
`PLAMATRIX_` 前缀的选项，避免污染父项目配置。

运行时设备选择：

| 环境变量 | 作用 |
|----------|------|
| `PLAMATRIX_OPENCL_DEVICE_INDEX` | 选择枚举得到的 OpenCL GPU；`-1` 表示自动选择 |
| `PLAMATRIX_OPENCL_DEVICE` | 按厂商和设备名的子串选择 OpenCL GPU |
| `PLAMATRIX_VULKAN_DEVICE_INDEX` | 选择 Vulkan 物理设备，默认 `0` |
| `PLAMATRIX_VULKAN_SPMV` | `auto`、`scalar`、`subgroup` 或 `block`，用于覆盖 Vulkan SpMV 策略 |
| `PLAMATRIX_VULKAN_BLOCK_JACOBI` | `auto`、`cpu` 或 `gpu`；控制 9×9 Schur 逆对角块由主机上传或在 GPU 构造 |

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
./build/cpu/benchmark/plamatrix_expression_benchmark --size 512 --iterations 80
```

比较公开 `MatrixXf` 乘法在各后端的首次执行（包含上传）和设备驻留重复执行：

```bash
./build/vulkan/benchmark/plamatrix_dense_backend_benchmark \
  --sizes 64,128,256,512,1024 \
  --backends cpu,cuda,vulkan,opencl \
  --iterations 5
```

输出为 CSV；`first_ms` 在后端初始化预热后测量输入上传、结果分配和一次 GEMM，
`resident_median_ms` 测量相同输入保持设备驻留后的中位时间。

表达式基准在 `CpuOnly` 下比较 `a + b + c + d` 一次遍历求值与使用 `eval()` 的逐次求值，
也比较 `(a + b).array().square() + c.array() * d.array()` 与分步生成中间矩阵的实现；
输入在计时前创建，报告三轮计时的中位数及校验和。

### Eigen 差分验证与性能基线

Eigen 5.0 或更新版本只用于开发验证，不会成为 `plamatrix` 库或安装包的传递依赖。配置阶段会
编译检查版本以及 `reshaped()`、`canonicalEulerAngles()`，避免用较旧头文件得到无效的差分结果：

```bash
cmake -S . -B build/eigen-reference \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLAMATRIX_WITH_CUDA=OFF \
  -DPLAMATRIX_BUILD_TESTS=ON \
  -DPLAMATRIX_BUILD_BENCHMARKS=ON \
  -DPLAMATRIX_WITH_EIGEN_REFERENCE=ON
cmake --build build/eigen-reference --parallel
ctest --test-dir build/eigen-reference --output-on-failure -R EigenReference

./build/eigen-reference/benchmark/plamatrix_eigen_compare \
  --size 256 --warmup 3 --trials 11 \
  --output build/eigen-reference/eigen-comparison.csv
```

基准在同一进程、相同输入和相同 Release 配置下比较 GEMM、逐元素链、部分主元求解和 SpMV，
并同时输出最大绝对误差。它包含两个实现各自的输出分配，不应与只测 kernel 的 GPU 时间混用。

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
可自动选择 subgroup-per-row SpMV；9×9 BA Schur CSR 会识别为稠密块结构，使用一次缓存的 block
row/column 索引执行 BSR-style SpMV，同时直接读取原有 device-local values。小解向量会在状态批次中同步回读，大解向量使用独立最终回读，
以避免多批次重复传输。对比工具中的 `command_buffer_recordings` 可用于确认录制复用是否生效。
实现和统计字段见[架构文档](docs/architecture.md)与
[编译指南](docs/build.md)。

支持 `VK_KHR_cooperative_matrix`、FP16 shader 和 16 位 storage buffer 的设备还可使用
`plamatrix::internal::vulkan::batchedMultiply16x16()`。该接口处理 row-major 16×16 批量乘法，显式开启
`allowMixedPrecision` 后使用 FP16 输入、FP32 累加的 cooperative-matrix 指令；6×6、9×3 或
9×9 的 BA/Schur 小块可补零后批处理。运行时会核对设备扩展、feature、compute stage、subgroup
大小和 16×16×16 矩阵规格。输入超出 FP16 安全范围、驱动不支持或 shader 工具链不支持时回退到
CPU FP32；`requireCooperativeMatrix` 可将回退改为明确错误。详见
[Vulkan cooperative matrix API](docs/api/vulkan-cooperative-matrix.md)。

可复现的 CPU-owned 热调用对比：

```bash
./build/vulkan/benchmark/plamatrix_cooperative_matrix_benchmark 4096 11
```

完整 BA Schur 热路径可用 9 维相机块、3 维点块和可配置观测数比较 CPU、OpenCL 与 Vulkan：

```bash
./build/vulkan/benchmark/plamatrix_schur_backend_benchmark 512 8192 4 7
```

Vulkan 路径通过 `plamatrix::internal::SchurComplementLinearBackend::Vulkan` 选择；`float` 法方程可使用设备装配，
`double` 法方程通过显式混合精度选项在 CPU 装配后交给 Vulkan FP32 PCG，并要求
`useMixedPrecision=true`。紧凑 9×3/3×3 块上传后由 GPU 完成补零、转置和 FP16 转换，两个
cooperative-matrix 阶段以 FP32 累加生成 Schur 项；CSR values 保留在 device-local buffer，并由
block-Jacobi PCG 直接绑定。Schur 装配、PCG 初始化和首批 8 次迭代共用一次提交；固定拓扑的后续 LM
轮次跳过 Schur/PCG 索引转换与上传。numerical-only 首批路径和后续完整迭代批次都会复用已录制
command buffer，稳定热路径不再重新录制。完整 9×9 block CSR 默认在同一命令缓冲内从已装配 values
执行 Cholesky 求逆并生成 block-Jacobi 预条件器，跳过 CPU primary 逆块计算和上传；报告中的
`deviceBlockJacobiUsed` 可确认该路径。

## 开发者指南

开始修改前建议按以下顺序阅读：

1. [`docs/architecture.md`](docs/architecture.md)：模块边界、存储布局、设备路径和内存语义；
2. 对应的 [`docs/api/`](docs/api/) 文档和公开头文件；
3. 同模块测试，确认边界条件和异常语义；
4. [`docs/contributing.md`](docs/contributing.md)：代码风格、验证矩阵和 PR 清单。

项目目录：

```text
plamatrix/
├── include/plamatrix/   公开数值接口；internal/ 为实现细节
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

- RowMajor 拥有型矩阵、`Map` 和 `Ref` 在 CPU 参与现有密集表达式；CUDA 密集核仍要求
  ColMajor，自动策略不会把 RowMajor 缓冲区作为列优先数据提交；
- `block()`/固定块/行列视图、向量分段、对角线和 `transpose()` 使用 Eigen 5 同形的公开
  `Block`/`VectorBlock`/`Diagonal`/`Transpose` 返回类型；`Ref` 可零复制绑定保留父矩阵 stride 的非连续块；
- `asDiagonal()`、`triangularView()`、`selfadjointView()`、`reshaped()`、`replicate()`、`reverse()` 和
  `transposeInPlace()` 已提供 Eigen 同名表达式；这些结构化表达式当前在 CPU 求值；
- 数组标量操作、逐元素除法、轴向归约和广播目前在 CPU 求值；列优先实数矩阵的
  `sum()`、`dot()` 和 `squaredNorm()` 可复用 CUDA/OpenCL/Vulkan 驻留数据；CUDA 表达式链尚未跨节点融合内核；
- OpenCL/Vulkan 已支持列优先 `float` 基础稠密算术、GEMM、全局归约、无选主元 QR、SVD 和自伴特征分解；
  OpenCL 在设备支持时也执行 `double`，Vulkan 当前仅执行 `float`。列选主元 QR、一般/广义特征分解、
  Schur 和 QZ 仍在 CPU；Vulkan 另提供 16×16 cooperative-matrix 批处理入口和 float 混合精度 Schur 后端；
- Vulkan PCG 内核使用 float32；double BA 通过受控混合精度入口调用；
- Vulkan resident PCG 当前在每次标量归约后同步，尚未复用旧求解器的批量命令录制；性能比较仍以注明入口的基准为准；
- Vulkan Schur 目前要求 primary/eliminated 块边长不超过 16，且 FP16 输入量化会带来可测数值误差；
- 内部 `Resident*` 对象绑定 `ExecutionContext`；借用的 context 须存活至对象释放，共享拥有型构造会保留 context；
- GPU 单元素 `getValue()` / `setValue()` 适合测试和调试，不应放在热循环中；
- 异步 API 要求输入、输出、workspace 和 stream 在操作完成前保持有效；
- 当前开发和性能验证以 Linux/GCC 为主，其他平台的结果应按实际环境单独报告。

## 文档

- [文档首页](docs/index.md)
- [编译指南](docs/build.md)
- [密集矩阵 API](docs/api/dense-matrix.md)
- [稀疏矩阵、迭代与直接求解](docs/api/sparse-matrix.md)
- [线性代数 API](docs/api/linear-algebra.md)
- [非线性优化 API](docs/api/optimization.md)
- [Vulkan cooperative matrix API](docs/api/vulkan-cooperative-matrix.md)
- [三维向量与点云运算](docs/api/point-cloud.md)
- [性能记录](docs/performance-bottlenecks.md)

## 贡献与许可证

欢迎提交可复现的 bug 报告、测试、文档和性能改进。较大的接口或后端改动建议先创建 issue，说明使用
场景、目标设备、兼容性影响和验证计划。Pull request 应写明行为变化、测试命令，以及尚未验证的平台。
详细流程见[贡献指南](docs/contributing.md)。

PlaMatrix 使用 [MIT License](LICENSE)。
