# Vulkan cooperative matrix API（内部实现）

本页描述 `plamatrix::internal` 实现接口，需显式包含 `plamatrix/internal/...` 头文件；不属于 Eigen 风格公开契约。BA 业务由 PlaBundle 负责，点云体素归组由 PlaPoint 负责。

PlaMatrix 提供一个面向 BA/Schur 小块的专用批量矩阵乘法入口。它不是通用 Vulkan
`DenseStorage` 后端：输入和输出仍由 CPU `std::vector<float>` 持有，每个矩阵固定为 row-major
16×16。6×6、9×3 和 9×9 块需要补零到 16×16，积的有效区域再由调用方取回。

## 能力探测

```cpp
const auto capabilities =
    plamatrix::internal::vulkan::selectedVulkanCooperativeMatrixCapabilities();

if (capabilities.enabled)
{
    // 当前构建和所选设备可执行混合精度 cooperative-matrix kernel。
}
```

`hardwareSupported` 表示驱动提供所需扩展、feature 和 16×16×16 FP16/FP32 规格；`enabled`
还要求当前 `glslangValidator` 能编译 shader，并且所选设备 subgroup 大小为 32。当前实现使用
`VK_KHR_cooperative_matrix`，不会以函数指针非空代替设备扩展检查。

## 批量乘法

```cpp
std::vector<float> left(batch_count * 16 * 16);
std::vector<float> right(batch_count * 16 * 16);

plamatrix::internal::vulkan::BatchedMatrixMultiplyOptions options;
options.allowMixedPrecision = true;

plamatrix::internal::vulkan::BatchedMatrixMultiplyReport report;
const auto output =
    plamatrix::internal::vulkan::batchedMultiply16x16(left, right, batch_count, options, &report);
```

默认 `allowMixedPrecision=false`，此时执行 CPU FP32。显式开启后，支持的 Vulkan 设备执行：

1. 合并上传两路 FP32 输入；
2. 在 GPU 上转换为 FP16；
3. 用 cooperative matrix 执行 FP16 输入、FP32 累加的 16×16×16 乘加；
4. 回读 FP32 输出。

pipeline、buffer、descriptor set 和已录制 command buffer 会在同尺寸热调用之间复用。报告中的
`gpuMilliseconds` 是上传、GPU 量化、矩阵乘法和回读命令的设备 timestamp，不包含主机端输入检查、
提交和 fence 等待。

## 数值与回退

混合精度路径只接受有限值和可表示为 FP16 的输入，并根据输入绝对值检查 FP32 累加上界。检查失败、
设备不支持或 GPU 输出出现非有限值时，接口自动执行 CPU FP32，并通过
`numericalSafetyFallback` 报告数值保护回退。设置 `requireCooperativeMatrix=true` 后，以上情况会抛出
异常，适合基准和能力门禁。

FP16 量化会改变结果。BA/Schur 调用方应使用 FP32 或 FP64 重算残差，并在收敛变差时切回高精度；
不要仅凭 kernel 时间判断整体求解性能。CPU-owned 接口包含上传和回读，小批量通常不会比 CPU 快。

## 设备常驻 Schur 路径

`SchurComplementLinearBackend::Vulkan` 将 cooperative matrix 接入完整的块 Schur 求解：

```cpp
plamatrix::internal::SchurComplementSolverOptions<float> options;
options.linearBackend = plamatrix::internal::SchurComplementLinearBackend::Vulkan;
options.useMixedPrecision = true;

const auto report = plamatrix::internal::solveDampedSchurComplement(
    equations, 0.1f, options, workspace, &primary_step, &eliminated_step);
```

该路径只接受 `float` 法方程和 1 到 16 的块边长。它上传紧凑的 cross/inverse 块，由 GPU shader
完成 16×16 补零、转置和 FP16 转换，再执行 `cross * inverse` 与
`transformed_cross * cross^T` 两阶段 cooperative-matrix 乘法。FP32 Schur values 在 GPU 归约后保留在
device-local buffer，并由 Vulkan block-Jacobi PCG 直接绑定，不经过设备内复制、主机回读或重新上传。同一
CSR block slot 的 term 在一个 FP32 cooperative accumulator 中直接求和，避免按 term 保存完整 16×16
中间矩阵。

`SchurComplementSolverWorkspace` 会缓存拓扑 buffer、数值 buffer 和 pipeline。Schur 上传与装配会录入
PCG 首次 command buffer，并与 PCG 初始化及首批 8 次迭代一起提交；固定 BA 邻接的后续 LM 轮次只上传
数值，跳过 CSR row/column 的主机转换和上传。numerical-only 首批路径与后续完整迭代批次都复用录制，
第三次及之后的稳定热调用可以保持 `commandBufferRecordings == 0`。报告中的
`cooperativeMatrixUsed`、`schurAssemblyOnDevice`、`commandSubmissions`、`commandBufferRecordings` 和
`gpuMilliseconds` 可用于确认实际路径。9×9 BA Schur 拓扑还会自动压缩为 block row/column 索引；
`blockSpmvUsed` 表示 PCG 已选择 BSR-style SpMV。该内核直接读取现有标量 values 布局，不增加数值重排。
`deviceBlockJacobiUsed` 表示同一命令缓冲已从 9×9 Schur 对角块执行 Cholesky 求逆并生成预条件器，省去
CPU primary 逆块计算、staging buffer 写入和 host-to-device copy。可用
`PLAMATRIX_VULKAN_BLOCK_JACOBI=cpu|gpu|auto` 做 A/B；`gpu` 需要设备常驻 values 与 9×9 block SpMV。

可用随 Vulkan benchmark 构建的工具分别测量 CPU FP32、中位端到端时间和 Vulkan timestamp：

```bash
./build/vulkan/benchmark/plamatrix_cooperative_matrix_benchmark 4096 11
./build/vulkan/benchmark/plamatrix_schur_backend_benchmark 512 8192 4 7
```
