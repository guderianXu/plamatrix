# 贡献指南

感谢你改进 PlaMatrix。本文说明提交 issue、修改代码、验证结果和准备 pull request 时需要遵循的约定。

## 开始之前

- Bug 报告应包含最小复现、完整错误信息、操作系统、编译器、CMake 版本和启用的后端。
- 性能问题应同时记录硬件、驱动、构建类型、矩阵规模、`nnz`、运行命令和多次测量结果。
- 较大的公开 API、存储布局或新后端改动，建议先创建 issue 讨论兼容性和验证范围。
- 不要在 issue、日志或测试数据中提交访问令牌、私有路径或不能公开的数据集。

## 建立开发环境

先创建一个可复现的 CPU-only 构建：

```bash
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

然后为实际修改到的可选后端创建单独构建目录。不要在同一个 CMake 缓存中反复切换 CUDA、OpenCL
和 Vulkan，以免旧工具链或依赖路径影响结果。

## 代码约定

- 使用 C++17、4 空格缩进和 Allman 花括号风格，行宽尽量不超过 120 字符。
- 类和结构体使用 PascalCase，函数使用 camelCase，局部变量使用 snake_case，私有成员使用 `_` 前缀。
- `DenseMatrix` 使用列优先布局；不要在局部优化中改变公开存储语义。
- `Device` 是编译期模板参数。昂贵的设备传输必须通过 `toCpu()` / `toGpu()` 等接口显式表达。
- CPU 参数和数值错误抛出明确异常；CUDA 调用使用项目现有的错误检查宏。
- 新增 CUDA API 时保持 `PLAMATRIX_WITH_CUDA=OFF` 可编译，并提供明确的 stub 行为。
- 公开函数应有简短文档注释，说明参数、返回值、所有权、同步要求和重要异常。
- 避免无关格式化、机械重命名和大范围重构；让每个 PR 保持单一、可审查的目标。

## 测试策略

修复 bug 时先增加能够复现问题的测试；新增功能至少覆盖正常路径、边界输入和失败路径。数值容差应有
算法依据，不能为了通过测试而任意放宽。

按改动选择验证：

| 改动 | 最低验证 |
|------|----------|
| 文档 | `git diff --check`，人工核对命令、链接和行为描述 |
| CPU 核心/密集运算 | CPU-only 构建、相关测试、全量 CTest |
| CUDA | CPU-only 构建、CUDA 构建、相关 GPU 测试和 CPU/GPU 一致性测试 |
| 稀疏求解 | COO/CSR、CG/PCG 测试，空矩阵、非法结构和非收敛路径 |
| OpenCL/Vulkan | 对应运行时测试、求解器测试、无另一后端的独立构建 |
| 性能 | 正确性测试、修改前后交替测量、完整复现命令 |
| CMake/安装 | 独立构建、`cmake --install` 和最小 `find_package` 使用方 |

运行单个测试可使用 CTest 正则：

```bash
ctest --test-dir build/cpu --output-on-failure -R 'Sparse|Iterative|Optimization'
```

提交前运行适用配置的完整测试：

```bash
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
git diff --check
```

只报告实际运行过的平台和后端。跳过测试、缺少设备或依赖失败时，在 PR 中写清原因和剩余风险。

## 文档和基准

- 新增公开 API 时更新对应 `docs/api/*.md`，并在需要时添加 `docs/examples/` 示例。
- 修改设备语义、构建选项或安装方式时同步更新 `README.md`、`docs/build.md` 和架构文档。
- benchmark 输出是环境相关的实验数据。除非它是经过说明的长期基线，否则不要提交临时报告。
- 性能结论至少包含中位数或其他稳健统计；不要只选最快的一次运行。

## Pull request 清单

- [ ] PR 描述说明了具体问题和最终行为。
- [ ] 改动保持在必要范围内，没有覆盖其他人的工作。
- [ ] 新增或修改的行为有对应测试。
- [ ] 适用构建和全量 CTest 已通过。
- [ ] 公开 API、构建方式和示例文档已同步。
- [ ] 性能结论包含硬件、命令、输入规模和修改前后数据。
- [ ] 未验证的平台、已知限制和兼容性风险已明确列出。
