# PlaMatrix Vulkan 计算后端设计

## 目标

为 PlaMatrix 增加 Vulkan Compute 后端，与现有 OpenCL PCG 后端使用相同的 CSR 输入、初值、
收敛条件和计时阶段，形成可复现的性能对比。第一阶段支持 float32，并覆盖 SpMV、点积归约、
AXPY/残差更新和 PCG；后续再扩展 float64、Schur assembly 和更多算子。

## 约束

- Vulkan 为可选后端，默认关闭；未安装 Vulkan SDK 或 GLSLang 时 CPU/OpenCL/CUDA 构建行为不变。
- 使用 Vulkan 1.1 Compute API，不引入窗口系统、图形交换链或 GPU 厂商专用扩展。
- 设备选择通过 `PLAMATRIX_VULKAN_DEVICE_INDEX`，枚举顺序和诊断信息与 OpenCL 保持一致。
- Vulkan shader 使用 GLSL 编写，在构建阶段由 `glslangValidator` 编译为 SPIR-V；运行时只加载
  构建产物。
- 所有 Vulkan 对象和分配采用 RAII；提交、fence、command buffer 和 descriptor 生命周期明确。

## 模块设计

新增 `include/plamatrix/vulkan/` 与 `src/vulkan/`：

- `runtime`：实例、物理设备、逻辑设备、计算队列、command pool、内存类型选择和一次性提交。
- `execution`：Buffer、descriptor、pipeline、shader module 的 move-only 所有权封装。
- `iterative_solver`：CPU-owned CSR 上传、SpMV、归约、预条件、方向更新和最终下载。
- `shaders/`：SpMV、点积归约、向量更新和 Jacobi 预条件 GLSL compute shader。

公开接口与 `plamatrix::opencl::pcg` 对齐，新增 `plamatrix::vulkan::pcg` 及设备枚举/可用性查询。
Schur 线性后端在第一阶段不自动切换到 Vulkan，避免把后端验证和 Schur assembly 绑定；完成
PCG 对比后再增加 `SchurComplementLinearBackend::Vulkan`。

## 基准与正确性

新增 backend comparison benchmark，固定生成同一组 SPD CSR 矩阵，分别记录：初始化、上传、
kernel-only PCG、下载和端到端时间。每个后端使用相同 warmup、trial、最大迭代次数和容差，
同时输出迭代次数、初始残差、最终残差和设备名称。测试验证 Vulkan 与 CPU 参考解的残差及
OpenCL/Vulkan 迭代结果的一致性。

## 构建与发布

- 增加 `PLAMATRIX_WITH_VULKAN` 选项和 `find_package(Vulkan)` 检测。
- `glslangValidator` 缺失时，用户显式开启 Vulkan 则配置失败；未显式开启则自动关闭。
- 安装时复制 SPIR-V shader，运行时通过编译定义定位 shader 目录。
- CPU-only、CUDA-only、OpenCL-only 和 Vulkan-enabled 配置分别保持可配置、可构建。

## 阶段性边界

第一阶段不实现 Vulkan SVD/GEMM，也不声称 Vulkan 在所有算子上优于 OpenCL。比较结果只对
相同 PCG 工作负载和当前设备有效，报告同时保留传输和纯计算时间。
