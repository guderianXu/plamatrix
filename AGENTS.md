# AGENTS.md

本文件给后续 agent 使用，优先级低于用户当前指令和系统/开发者指令。目标是在 PlaMatrix 项目里改代码、跑测试、维护 benchmark 和同步文档时保持一致做法。

## 项目定位

PlaMatrix 是面向点云处理的高性能矩阵运算库，使用 C++17，支持 CPU OpenMP 和可选 CUDA GPU 加速。它是 PlaPoint 的底层数学后端。主线包含：

- `include/plamatrix/core/`：`Device`、`Index`、RAII 内存管理、CUDA/cuBLAS/cuSOLVER 错误检查、no-CUDA stubs。
- `include/plamatrix/dense/`：列优先 `DenseMatrix` 和逐元素运算。
- `include/plamatrix/sparse/`：COO、CSR 稀疏矩阵。
- `include/plamatrix/ops/`：GEMM、SVD/QR/Eigh、线性求解、点云刚体变换和协方差。
- `src/`：CPU OpenMP 实现、CUDA kernel、cuBLAS/cuSOLVER 调用和模板实例化。
- `test/`：Google Test 单元测试、CPU/GPU 一致性测试、NumPy/SciPy 参考数据测试。
- `benchmark/`：串行 CPU、OpenMP、CUDA benchmark 和 Markdown 报告生成。
- `docs/`：架构、构建、API 和示例文档。

先读现有实现、测试、`docs/architecture.md` 和相关 API 文档，再改动。优先延续当前模块边界和命名风格，避免无关重构。

## 工作区约束

- 本目录既是独立 Git 仓库，也是上层 `plascan/3rdparty` 下的子模块。远端仓库是 `https://github.com/guderianXu/plamatrix.git`，当前主分支为 `main`。
- 开始改动前检查：
  ```bash
  git status --short
  git branch --show-current
  ```
- 如果在上层项目中工作，只有在用户明确要求时才更新上层仓库的子模块指针。
- 不要删除或批量重写 `test/reference/` 的 `.bin` 参考数据。只有数学定义、容差策略或参考生成脚本确实变化时，才重新生成并说明命令和原因。
- benchmark 输出通常是实验产物。除非用户要求留档，不要把临时报告、机器特定性能结果或大文件加入版本控制。
- PlaPoint 依赖 PlaMatrix 的公开 API。修改 `include/plamatrix/` 下的接口、布局、异常行为或设备语义时，要考虑下游兼容性。
- 不要把 GitHub token、私有路径、临时 PID 或本机凭据写进文档、脚本或 remote URL。

## 代码规范

### C++

- 使用 C++17，4 空格缩进，不使用 tab，行宽尽量不超过 120 字符。
- 新增或大幅重写的代码使用 Allman brace style，左大括号单独成行。编辑旧代码时优先保持邻近风格，不做纯格式化 churn。
- 类和结构体使用 PascalCase，例如 `DenseMatrix`、`CudaAllocator`。
- 函数和方法使用 camelCase，例如 `toGpu()`、`solveLinear()`。
- 局部变量和普通成员变量使用 snake_case，例如 `row_count`、`data_ptr`。
- 私有成员变量前加下划线，例如 `_data`、`_rows`。
- 常量和宏使用大写下划线，例如 `PLAMATRIX_WITH_CUDA`、`PLAMATRIX_NO_CUDA`。
- 模板参数使用 PascalCase 或单个大写字母，例如 `Scalar`、`Dev`、`N`。
- 命名空间使用 `plamatrix` 及必要的内部子命名空间。
- 头文件使用 `.h` 和 `#pragma once`；CUDA 源文件使用 `.cu`；纯 C++ 源文件使用 `.cpp`。
- 公共声明需要简短文档注释，说明参数、返回值和重要异常条件。
- 新文件 include 顺序优先为标准库、CUDA/cuBLAS/cuSOLVER、第三方库、项目头文件；编辑旧文件时不要只为 include 排序制造无关 diff。
- CPU 侧参数非法、维度不匹配、内存分配失败和数值求解失败优先抛出明确异常。
- GPU 侧必须使用 `PLAMATRIX_CHECK_CUDA`、`PLAMATRIX_CHECK_CUBLAS`、`PLAMATRIX_CHECK_CUSOLVER` 或现有封装，不留下未检查的 runtime/library 调用。
- 一个函数只承担清晰职责。文件超过 400 行或嵌套超过 4 层时优先拆小，但不要为了机械满足限制做无意义拆分。

### 数值和设备约定

- `DenseMatrix` 使用列优先存储，索引公式是 `row + col * rows`。这与 cuBLAS/cuSOLVER 调用参数强相关，不能随意改为行优先。
- `Device` 是编译期模板参数。CPU/GPU 路径优先通过 `if constexpr` 分发，避免不必要的运行时分支。
- `DeviceMatrix` 负责 RAII 内存管理，矩阵对象禁止隐式昂贵拷贝，优先 move 和显式 `toCpu()` / `toGpu()`。
- `operator()(row, col)` 只用于 CPU 矩阵；跨设备路径使用 `getValue()` / `setValue()`，并注意 GPU 单元素拷贝成本。
- `Index` 使用 64 位有符号整数。新增循环和 CUDA kernel 入参时要显式处理 `Index` 到 `int` 的边界，不要静默窄化大矩阵尺寸。
- `PLAMATRIX_WITH_CUDA=OFF` 时 no-CUDA stubs 需要保持可编译、可运行。新增 CUDA include、类型或 API 时必须有无 CUDA 分支。
- `PLAMATRIX_USE_FLOAT` 和 `PLAMATRIX_USE_DOUBLE` 控制显式实例化。新增运算时要同时覆盖启用的标量类型。
- 数值测试容差必须和算法误差来源匹配。不要为了让测试通过随意放宽 tolerance；需要放宽时写清楚 CPU/GPU、float/double 的依据。

## 构建与测试

常用开发验证：

```bash
cmake -S . -B build -DPLAMATRIX_BUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/test/plamatrix_tests
ctest --test-dir build --output-on-failure
```

CPU-only 验证：

```bash
cmake -S . -B build-cpu -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_WITH_CUDA=OFF
cmake --build build-cpu -j$(nproc)
ctest --test-dir build-cpu --output-on-failure
```

benchmark 构建和短跑：

```bash
cmake -S . -B build-bench -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build build-bench -j$(nproc)
./build-bench/benchmark/plamatrix_benchmark --mode cpu --size small --output benchmark_report.md
```

改动选择验证范围：

- 改 core/allocator/device_matrix：跑全量测试、CPU-only 构建和 CPU/GPU 一致性测试。
- 改 dense ops 或 GEMM：跑对应 unit test、NumPy reference test、CPU-only 构建和 CUDA 构建。
- 改 SVD/QR/Eigh/solver：跑 decomposition/solver 单测、reference 测试，并检查 float/double 容差。
- 改 sparse：跑 COO/CSR 单测，并验证空矩阵、非法尺寸和边界索引。
- 改 point_cloud ops：跑 point_cloud 单测、workflow integration test，并考虑 PlaPoint 下游影响。
- 改 benchmark：跑 `--mode cpu --size small`。只有具备 CUDA 环境时再跑 `--mode all` 或 `--mode cuda`。
- 改 CMake 选项：至少验证默认构建、`PLAMATRIX_WITH_CUDA=OFF`、`PLAMATRIX_BUILD_TESTS=ON`。

重新生成参考数据前先确认 Python 依赖，并只在预期数值变化时执行：

```bash
python3 test/reference/generate_reference.py
```

## 文档与 benchmark

- 公开 API、设备语义、存储布局、构建选项或 no-CUDA 行为变化时，同步更新 `README.md`、`docs/build.md`、`docs/architecture.md` 和对应 `docs/api/*.md`。
- 新增主要功能时，在 `docs/examples/` 中补充最小可运行示例，避免只写 README 片段。
- benchmark 报告应记录命令、模式、size preset、CPU/GPU、CUDA 版本和输出路径。面向人工阅读的结果优先 Markdown。
- 性能结论要给出具体命令和硬件环境，不要只写“更快”或“优化明显”。

## Git 与同步

- 如用户要求提交，先检查 `git status --short`，保留用户未授权的改动。
- 提交作者建议使用 GitHub 关联身份：
  ```bash
  git config user.name "guderianXu"
  git config user.email "guderian_xu@henu.edu.cn"
  ```
- 不要在未经用户要求时 commit、push、改 remote 或更新上层子模块指针。
- 若用户要求发布到 GitHub，使用不含凭据的 remote URL：
  ```bash
  git remote set-url origin https://github.com/guderianXu/plamatrix.git
  ```

## 文档与沟通

- 对用户用中文简洁说明做了什么、验证了什么、还有什么风险。
- 涉及矩阵尺寸、性能、显存、CUDA 架构和测试结果的结论要给出具体命令、硬件条件或测试路径。
- 不要把临时 PID、一次性时间估计、sudo 密码或访问令牌写进项目文档。
- 面向人工留档的报告优先写 Markdown；机器消费的数据才写 JSON/CSV/bin。
