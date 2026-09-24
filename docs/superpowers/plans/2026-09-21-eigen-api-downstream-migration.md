# Eigen 风格矩阵 API 与下游直接迁移

状态：进行中。此文件记录破坏性迁移的真实边界；完成一个阶段后更新状态，不以兼容别名或转发层代替调用点迁移。

## 最终接口

- 应用层使用 `Matrix<Scalar, Rows, Cols>`、`MatrixXd`、`MatrixXf`、`Matrix3d`、`Vector3d` 等。固定/动态尺寸、`Zero`/`Ones`/`Identity`/`Constant`、系数访问、`resize`、`block`/`row`/`col`、算术运算符、`dot`/`norm` 遵循 Eigen 的常用命名和列优先数据布局。
- 默认 `ExecutionPolicy::Auto` 按运算能力、数据驻留和计算规模选择后端；`executionInfo()` 可观察实际选择和传输。主机访问按需下载，可变访问使 GPU 副本失效。明确要求 GPU 且后端不可用时失败。
- 自定义内核、异步流和稀疏求解使用 `ExecutionContext` 与 `ResidentMatrix`/`ResidentVector`/`ResidentCsrMatrix`；设备驻留和原生指针仅属于这层。密集矩阵的自动运算与原生内核的设备管理是不同职责。
- `DenseMatrix<Scalar, Device>`、`toGpu()`/`toCpu()`、旧的 `gemm`/`add`/`sub` 等公开调用在消费者改完后删除。底层 CPU/CUDA 算法可继续复用实现，但不能保留同名兼容转发 API。

## 当前已验证的能力

- PlaMatrix 新密集入口支持固定/动态矩阵、向量、工厂函数、常用运算符、视图、深拷贝，以及 CPU/CUDA 自动执行和驻留诊断。
- `ResidentMatrix::copyFrom(Matrix, ExecutionContext)` 与 `toHostMatrix()` 已打通高层主机对象和低层设备对象。
- CUDA 原生实现可从 `cuda::NativeAccess::stream(context)` 获得与 resident 工作区关联的 stream；CPU、CUDA、OpenCL、Vulkan 的矩阵往返测试已覆盖该入口的存储语义。
- OpenCL/Vulkan 目前只有专用 CSR/Schur 路径，没有通用密集 GEMM；自动密集运算不能声称覆盖这两个后端。常用延迟表达式、固定尺寸 block、`Map` 与 CPU 3×3 自伴特征分解已实现，但还不是全部 Eigen API。`noalias()` 表达式赋值已接入现有自动 CUDA 选择，跨固定/动态尺寸的结果保持设备驻留；目前 CUDA 仍为结果分配新缓冲区。
- PlaPoint KD-tree 的单点查询已改为 `Matrix<Scalar, 3, 1>`，半径过滤、ICP、Poisson 重建及 PlaScan 的对应查询调用已直接迁移。PlaBundle 和 PlaScan 的标量三维几何计算中已清除 `plamatrix::Vec3` 调用。
- PlaPoint 的 `PointCloud` 公开属性仍使用 `DenseMatrix<Scalar, Device>`，其 CUDA 内核与 PlaMatrix 旧矩阵/点云运算紧耦合，因此本次尚不能删除旧 API；不能将当前状态描述为下游迁移完成。
- PlaPoint CPU/OpenCL 法线估计的局部协方差和最小特征向量计算已直接改用 `Matrix` 与 `SelfAdjointEigenSolver<Matrix<Scalar, 3, 3>>`；OpenCL 路径另使用 `noalias()` 完成协方差乘法。对外点云容器仍待后续直接迁移。
- 已验证：PlaMatrix CPU 408/408、CUDA 713/713、Vulkan 462/462；PlaPoint CPU 377/377；PlaScan 的 PlaBundle、SFM、MVS 和网格相关目标编译通过，相关 CTest 101/101。CPU/Vulkan 配置中的 CUDA 专用测试按配置跳过。
- 2026-09-22 本轮验证：PlaMatrix CPU CTest 437 项无失败（12 项按 CPU-only 配置跳过）；PlaPoint CPU/OpenCL CTest 392 项无失败（3 项按配置跳过），其中 OpenCL 法线与 CPU 法线对照 2/2 通过。PlaScan `sfm_core`、`mvs_backend` 构建通过。上一行数字是先前阶段的历史记录，不作为本轮 CUDA/Vulkan 验证结论。
- ICP 仍调用专用 `svd3x3`：该算法对秩亏矩阵可有约 10⁻⁸ 的伪非零奇异值，当前不能直接提升为通用 `jacobiSvd()` 实现；迁移前需保持退化配准及最小范数求解语义。

## 必须按顺序完成的直接迁移

1. PlaMatrix：将公共运算输入/输出从旧拥有型设备矩阵改为 `Matrix`、`MatrixView` 或 `ResidentMatrix`，补齐 PlaPoint 实际使用的分解、求解、索引、归约、点云和异步内核入口；删除被替代的旧声明与实现。
2. PlaPoint：先改 `PointCloud` 公开属性、CPU 搜索/过滤/IO，再改 CUDA KNN、ICP、法线和网格内核的设备工作区。CUDA 内核所需的 stream、生命周期和原始 device pointer 必须由新 resident 接口明确表达，不能用旧容器包装转发。
3. PlaBundle：现有 BA 主要调用 `BlockNormalEquations`/Schur 求解，没有直接 `DenseMatrix` 调用。待 PlaMatrix 稀疏/优化接口定型后迁移其后端枚举、工作区和求解调用，并保留可观察的实际执行后端。
4. PlaScan：在 PlaPoint 点云入参改变后，迁移核心、CLI、GUI 和测试中的 `DenseMatrix` 构造及 `Device::CPU` 类型。当前多数调用是向 PlaPoint 传点、颜色、法线和网格数据，不能孤立替换类型。
5. 删除旧公开头文件和下游旧引用；分别完成 PlaMatrix CPU/CUDA/OpenCL/Vulkan、PlaPoint CPU/CUDA、PlaBundle、PlaScan 原生构建与相关测试，再运行各仓库要求的全量门禁。

## 依赖审计（2026-09-21）

- PlaPoint `include/` 和 `src/` 约 497 处 `DenseMatrix` 引用，其中 CPU 约 182 处、GPU 约 274 处，另有模板设备参数引用。`PointCloud` 自身把旧容器暴露在公开类型中；`registration/icp.h` 含大量直接 GPU 工作区代码。
- PlaScan `src/` 中的旧矩阵构造主要流向 PlaPoint 的 `PointCloud`，因此排在 PlaPoint 之后。
- PlaBundle 的直接矩阵容器引用为零；它的旧依赖集中在 PlaMatrix Schur 求解配置，而非密集容器。

以上数量仅用于界定迁移范围；删除旧 API 的完成标准是源码搜索无旧调用、四仓库构建与测试通过，而非数量减少。
