# PlaMatrix 后端上下文与异步 API 设计

## 背景

PlaMatrix 当前的矩阵对象、CUDA stream、OpenCL runtime 和 Vulkan runtime 分别演化，导致同一个算法在不同后端拥有不同的数据驻留和生命周期语义。CUDA 求解器使用设备常驻矩阵和 caller-owned workspace，OpenCL/Vulkan 的公开 PCG 接口则接收 CPU 矩阵并在内部完成传输。异步 CUDA API 还要求调用者自行管理 stream、workspace 状态和 stream-ordered allocation 的释放。

这套设计适合 PlaScan 当前的内部调用，但不适合作为长期稳定的多后端公共 API。本次迁移直接替换旧的公共入口，不增加兼容层；PlaScan 调用点、测试和文档在同一变更中完成迁移。

## 目标

1. 用后端无关的执行上下文表达后端、设备选择、默认执行队列和临时内存生命周期。
2. 用统一的完成对象表达同步或异步操作，并让异步错误在一个明确的边界被观察到。
3. 为 PCG/CSR 提供设备常驻的新入口，使 CUDA、Vulkan、OpenCL 的传输开销可以从求解计时中分离。
4. 让 PlaScan 和 PlaMatrix 测试直接使用新入口，避免两套 API 长期并存。
5. 在无 CUDA/OpenCL/Vulkan 构建中不向用户头文件泄露假的全局 CUDA runtime API。

## 非目标

- 第一阶段不保留 `Device::GPU`、裸 `cudaStream_t` 求解器入口、现有 singleton runtime 或旧版求解器入口作为公开 API。
- 第一阶段不实现完整 Eigen 式表达式模板。
- 第一阶段不承诺所有算子同时获得新后端实现；先完成 PCG/CSR 的 API 垂直切片，并同步迁移已有调用点。
- 第一阶段不改变现有数值算法、收敛准则或默认参数。

## v1 API 冻结前必须确定的基础语义

### 标量、索引、形状和布局

公共 API 必须显式声明支持的标量集合、索引宽度、矩阵布局和步长规则。`float`、`double`、`std::int32_t`、`std::int64_t` 是否可用不能继续依赖链接阶段的显式实例化结果；新增 `ScalarTraits`/`isSupportedScalar`，在编译期拒绝未支持类型。矩阵形状使用统一的 `Shape2`/`Shape3` 和 checked size 计算；布局通过 `Layout::ColumnMajor`、`Layout::RowMajor` 或显式 strides 表达，算法不得假定所有输入都是 column-major。

### 所有权、view 和别名

拥有数据的矩阵、非拥有 view、只读 view 和设备驻留对象必须是不同的类型语义。公共 API 增加 `asConst()`、`clone()`、`copyTo()`、`assign()` 和明确的 aliasing 规则；不依赖模板推导去完成 mutable-to-const 转换。任何可能同步设备的单元素 `getValue()`/`setValue()` 不再作为设备矩阵的便利接口，设备数据读取通过显式 copy 或 reduction 完成。

### 分配器和临时内存

`ExecutionContext` 使用可替换的 `MemoryResource`，至少支持默认分配器、用户分配器、host pinned 和 backend device memory。算子临时空间由 context 的 scratch arena 管理，不能再要求每个模块公开一套 `Workspace` 生命周期和 `closeAsyncAllocation()`。

### 异步完成语义

`Event` 的析构不得隐式阻塞；它只释放一个完成状态引用，context 在事件完成后回收关联资源。需要确定结果或观察错误时必须调用 `wait()`。输出对象和临时资源要么由 event 持有，要么由 context 延迟回收，不能出现悬空设备指针。所有异步 API 使用同一参数顺序：输入、输出、options、context，并返回 `Event`。

### 能力发现和后端降级

`ExecutionContext` 提供能力查询，例如数据类型、稀疏格式、cooperative matrix、FP64、异步分配和 command buffer。算子提交前可以通过 capability 查询选择路径；能力缺失时返回 `BackendUnavailable` 或 `UnsupportedOperation`，不进行隐式 CPU 回退。CPU 回退必须由调用方显式创建 CPU context。

### 算子签名和扩展方式

所有新算子统一使用 `input, output/options, context` 的参数顺序。同步便利入口返回拥有结果；高性能入口由调用方提供输出并接收 `Event`。算子不从全局 singleton 读取设备，也不在参数中隐式创建 stream。通用 `OperationOptions` 为后续增加确定性、累加精度、诊断和 cancellation 保留扩展点；暂不实现完整表达式模板，但保留 `assign/eval/reduce` 命名空间，固定参数数量的 `linearCombination3` 一类入口不再继续扩展。

### 稀疏矩阵语义

CSR/COO/BSR 的 index type、index base、row offsets 单调性、column sortedness、duplicate policy 和 value/structure 可变性必须写入类型或构造 options。结构和数值分离时，修改 structure 必须使相关分析缓存失效；仅修改 values 可以复用结构分析。PCG 的 matrix、preconditioner 和 workspace 必须绑定同一 context、scalar、index type 和 shape。

### 数值与确定性策略

`SolveOptions`/`ReductionOptions` 统一表达容差、最大迭代、累加精度、NaN/Inf 策略、确定性级别和是否记录 residual history。后端 report 需要同时记录实际采用的精度、算法路径和是否满足确定性要求，避免只记录一个最终残差。

### ABI、版本和扩展边界

公共符号使用 `PLAMATRIX_API` 可见性宏，稳定 API 与 backend-native/experimental API 分层。增加 `plamatrix::v1` 命名空间、编译期版本宏、deprecated 宏和安装后的 consumer contract test。C++ 模板类型可以保持 header 定义，但 runtime/context/error 的非模板 ABI 必须有稳定导出边界。新算子优先通过 options/capabilities 扩展，避免继续增加带固定参数数量的函数重载。

### 线程、取消和观测

文档明确 context、matrix、event 的线程安全等级；默认 context 不在多个线程并发提交。异步长迭代求解预留 cancellation token 和 progress callback，但第一阶段只定义稳定的空接口，不把 GUI 进度回调耦合进核心。诊断数据通过 `ExecutionTrace`/`BackendDiagnostics` 统一承载，避免在通用 report 中继续增加后端专用布尔字段。

## 设计

### 后端与设备

增加稳定的后端枚举和设备标识：

```cpp
enum class Backend
{
    Cpu,
    Cuda,
    OpenCl,
    Vulkan
};

struct DeviceId
{
    Backend backend = Backend::Cpu;
    std::size_t index = 0;
};
```

`DeviceId` 只描述执行设备。旧的 `Device::CPU/GPU` 模板模型在迁移中移除；新 API 使用明确的 `MemorySpace`/后端驻留类型，不再把 `GPU` 当作所有 GPU 后端的统一内存类型。

### ExecutionContext

`ExecutionContext` 是不可移动、不可复制的后端资源所有者。不可移动保证 context 绑定对象中的身份引用在整个生命周期内稳定；需要跨作用域传递时传递引用或指针。它至少提供：

```cpp
class ExecutionContext
{
public:
    static ExecutionContext create(DeviceId device = {});
    Backend backend() const noexcept;
    DeviceId device() const noexcept;
    void synchronize();
};
```

上下文负责默认执行队列/stream、后端资源引用和可复用 scratch 生命周期。CPU context 可以直接完成操作；CUDA/OpenCL/Vulkan 的具体 queue/stream 保持在 backend implementation 中，不进入稳定核心头文件。原生句柄只通过明确命名的 backend native header 暴露。

### Event

异步入口返回 move-only `Event`：

```cpp
class Event
{
public:
    Event() noexcept;
    bool valid() const noexcept;
    void wait();
    bool ready() const;
};
```

`wait()` 是唯一要求调用者观察异步执行错误的公共边界。CPU 实现可直接返回已完成事件；后端实现可以包装 CUDA event、OpenCL event 或 Vulkan fence。事件持有完成所需的延迟释放状态，调用者不再管理 stream-ordered allocation，也不再调用 `closeAsyncAllocation()`。

### PCG 新入口

第一阶段将 CSR/PCG 的旧入口替换为 context 版本的设备常驻入口：

```cpp
template <typename Scalar>
IterativeSolverReport pcg(const CSRMatrix<Scalar, Device::CPU>& matrix,
                          const DenseMatrix<Scalar, Device::CPU>& rhs,
                          DenseMatrix<Scalar, Device::CPU>& solution,
                          const IterativeSolverOptions& options,
                          ExecutionContext& context);

template <typename Scalar>
Event pcgAsync(const ResidentCsrMatrix<Scalar>& matrix,
               const ResidentVector<Scalar>& rhs,
               ResidentVector<Scalar>& solution,
               const IterativeSolverOptions& options,
               ExecutionContext& context,
               IterativeSolverReport* report = nullptr);
```

`ResidentCsrMatrix`、`ResidentVector` 是新 API 的后端驻留类型。它们拥有 context 绑定关系，不能被不同 backend 的 context 直接混用。同步便利入口可以从 CPU 输入创建临时设备对象，但必须在文档和 report 中明确传输边界；异步/重复求解使用设备驻留入口。旧的 CPU-owned Vulkan/OpenCL PCG 入口和 CUDA `cudaStream_t` 重载直接删除，调用方必须显式创建 context 和设备对象。`Resident*` 命名用于避开当前仍在迁移的 `DeviceMatrix<Scalar, Device>` 基类；最终公共 API 冻结时只保留一个矩阵所有权命名体系。

`IterativeSolverReport` 保留 `converged`、`iterations`、初始/最终残差等通用字段。命令录制、descriptor、barrier、subgroup kernel 等字段移动到可选的 backend diagnostics 结构，不再污染所有后端的通用报告。

### 错误模型

第一阶段增加统一 `Error` 类型，保留标准异常作为抛出机制；所有迁移后的公共入口都使用该类型：

```cpp
enum class ErrorCode
{
    InvalidArgument,
    UnsupportedBackend,
    BackendUnavailable,
    InvalidState,
    BackendFailure,
    NumericalFailure
};

class Error : public std::runtime_error
{
public:
    Error(ErrorCode code, std::string message,
          Backend backend = Backend::Cpu, int nativeCode = 0);
    ErrorCode code() const noexcept;
    Backend backend() const noexcept;
    int nativeCode() const noexcept;
};
```

新 API 的同步验证错误和 `Event::wait()` 的异步错误都使用 `Error`。旧的异常分支和后端错误宏在迁移完成后从公共实现中删除。

### 头文件分层

- `plamatrix/plamatrix.h` 只保留稳定容器、view、基础算子和 context/event。
- `plamatrix/cuda/native.h`、`opencl/native.h`、`vulkan/native.h` 承载原生句柄和后端专用能力。
- 旧版 backend runtime 头文件不再安装为公共入口；后端原生能力只从明确的 backend native header 使用。
- `no_cuda_stubs.h` 删除；无后端构建通过 feature discovery 和 `ErrorCode::BackendUnavailable` 表达能力，不伪造 CUDA runtime API。

## 生命周期与线程规则

- context 必须比其创建的 device matrix、event 和 workspace 存活更久。
- 一个 event 可以移动到另一个作用域，但不能复制；销毁 event 不阻塞，关联资源由 context 延迟回收。
- context 和 workspace 默认不保证并发复用；不同线程应使用不同 context，或由后续版本显式提供线程安全队列。
- 所有调用方通过 context 获取执行顺序；不再有公开的裸 stream 生命周期规则。

## 迁移策略

1. 先实现 core 的 `Backend`、`DeviceId`、`Error`、`ExecutionContext`、`Event`，并补充 CPU 单元测试。
2. 实现设备驻留 CSR/vector 的最小容器和 context 绑定检查。
3. 将现有 CUDA PCG 直接改写为新入口，迁移 PlaScan 和所有 PlaMatrix 测试，删除旧重载。
4. 增加 Vulkan/OpenCL 的 context 实现；如果设备后端不可用，返回结构化 `BackendUnavailable`。
5. 增加 API 契约测试，覆盖 CPU-only 编译、跨 context 使用、异步 wait、重复求解和传输计时边界。
6. 迁移其他算子，清理旧 workspace、singleton 和裸 stream 公共声明，更新所有文档和安装清单。

## 验收标准

- CPU-only 配置可以包含稳定 umbrella header，并使用 CPU context 编译、运行。
- 可用的 CUDA 构建中，新 PCG 入口在相同输入上的解、迭代次数和收敛状态通过既有数值基准与参考实现验证。
- 设备驻留矩阵重复调用 PCG 时不发生隐式的每次上传/下载。
- Vulkan/OpenCL 不可用时，新入口返回 `ErrorCode::BackendUnavailable`，而不是空指针、静默回退或未定义行为。
- 异步执行只需保存并等待 `Event`，不需要调用者手动管理临时 allocation 的关闭顺序。
- 迁移后的 PlaScan 和 PlaMatrix 测试全部使用新 API 并保持通过；仓库中不再存在旧公共入口的调用。

## 风险与取舍

- 设备驻留类型会增加模板和后端适配代码，但这是区分传输成本与计算成本所必需的。
- event 析构等待可能在错误使用时产生阻塞；文档和 debug 诊断应明确其行为，后续可增加显式 `detach`，第一阶段不引入。
- 这是一次破坏性迁移，调用方需要同步更新；通过版本升级说明、迁移文档和编译期错误尽早暴露未迁移代码。
