# 贡献指南

感谢您对 PlaMatrix 的关注！本文档说明如何参与 PlaMatrix 的开发。

## 编码规范

项目遵循统一的编码规范，详见仓库根目录的 `CLAUDE.md`。以下是关键原则的摘要：

### 命名规范

| 分类 | 风格 | 示例 |
|------|------|------|
| 类/结构体 | PascalCase | `DenseMatrix`, `CudaAllocator` |
| 函数/方法 | camelCase | `toCpu()`, `rotationMatrix()` |
| 变量 | snake_case | `row_count`, `data_ptr` |
| 私有成员 | _前缀 snake_case | `_data`, `_rows` |
| 常量/宏 | UPPER_SNAKE_CASE | `MAX_THREADS`, `PLAMATRIX_VERSION` |
| 模板参数 | PascalCase / 单字母 | `Scalar`, `Dev`, `N` |
| 命名空间 | 全小写 | `plamatrix` |

### 格式规范

- 4 空格缩进，不使用 tab
- 行宽限制 120 字符
- 头文件函数声明必须写文档注释（`///` 或 `/** */`），说明参数、返回值、异常条件
- 头文件使用 `#pragma once`

### 错误处理

- **CPU 侧**：使用异常（`throw std::runtime_error`），构造失败或参数非法时抛出
- **GPU 侧**：kernel 内使用错误码；host 侧 CUDA 调用使用宏：
  - `PLAMATRIX_CHECK_CUDA(call)` — CUDA Runtime API
  - `PLAMATRIX_CHECK_CUBLAS(call)` — cuBLAS API
  - `PLAMATRIX_CHECK_CUSOLVER(call)` — cuSOLVER API

### 文件组织

- 一个文件不超过 400 行，超出则拆分
- 嵌套不超过 4 层
- 头文件使用 `.h`，CUDA 源文件使用 `.cu`，纯 C++ 使用 `.cpp`
- 公共头文件统一由 `include/plamatrix/plamatrix.h` 聚合

### include 顺序

1. 标准库头文件
2. CUDA 头文件（`cuda_runtime.h`, `cublas_v2.h`, `cusolverDn.h`）
3. 第三方库头文件
4. 项目头文件（`plamatrix/...`）

每组之间空行分隔，组内按字母序。

## 开发流程 (Feature Branch + TDD)

### 分支策略

- `main` 分支始终稳定可构建，不直接在 main 上开发
- 每个功能/修复在独立 feature 分支上开发：`feat/<描述>` 或 `fix/<描述>`
- 分支上完成 TDD 循环（红 -> 绿 -> 重构）且测试全部通过后，合并回 main

### TDD 铁律

1. **先写测试** -> 运行确认失败（红）
2. **写最小实现** -> 运行确认通过（绿）
3. **重构优化** -> 保持测试通过
4. **每次提交前跑相关测试**，禁止提交破坏测试的代码

### 分支操作流程

```bash
# 1. 从 main 创建功能分支
git checkout -b feat/<功能名>

# 2. TDD 开发
#   - 写测试 -> 提交
#   - 写实现 -> 提交
#   - 重构   -> 提交

# 3. 合并回 main
git checkout main
git merge feat/<功能名>
git push origin main
```

> 注意：`main` 分支可能有未合并的 feature 分支，因此需要先 `git pull origin main` 获取最新代码后再切换分支。

### 编译验证

**每次修改代码后必须在 build 目录编译验证**：

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
ctest --output-on-failure
```

编译失败不能提交；编译警告需评估。

## Pull Request 清单

提交 PR 前请确认：

- [ ] 代码遵循编码规范（命名、格式、include 顺序）
- [ ] 所有新功能有对应的单元测试
- [ ] 所有测试通过：`ctest --output-on-failure`
- [ ] CPU 和 GPU 路径均已覆盖（如果功能涉及 GPU）
- [ ] 编译无错误、无新增警告
- [ ] 新增的公共 API 已添加文档注释
- [ ] 未破坏现有功能（回归测试通过）
- [ ] 如果有性能相关变更，已运行基准测试对比
- [ ] PR 描述清晰说明了变更原因和影响范围
- [ ] 单个 PR 范围聚焦——一个 PR 解决一个问题

## 测试规范

- 测试文件命名：`*_test.cpp`
- 测试用例命名：`ClassName_MethodName_Scenario`
- 测试框架：Google Test
- GPU 测试需要 CUDA 设备可用

### 测试分类

| 目录 | 内容 |
|------|------|
| `test/unit/core/` | 核心组件（allocator, types, error_check） |
| `test/unit/dense/` | DenseMatrix 及其运算 |
| `test/unit/sparse/` | COO/CSR 稀疏矩阵 |
| `test/unit/ops/` | GEMM、SVD、QR、Eigh、solver、point_cloud |
| `test/integration/` | CPU vs GPU 一致性测试 |

## 新增功能开发指南

### 添加新的矩阵运算

1. 在 `include/plamatrix/ops/` 下添加头文件（声明 CPU 和 GPU 重载）
2. 在 `src/ops/` 下添加：
   - `*_cpu.cpp`：CPU 实现（OpenMP 并行）
   - `*.cu`：GPU 实现（CUDA kernel 或 cuBLAS/cuSOLVER 调用）
3. 在 `include/plamatrix/plamatrix.h` 中添加聚合 include
4. 在 `src/CMakeLists.txt` 中添加源文件
5. 在 `test/unit/ops/` 下添加测试文件（含 CPU 和 GPU 测试）
6. 运行完整测试套件确保通过

### 添加新精度类型

如果需要支持 `int32` 等新元素类型：

1. 在 `types.h` 或新文件中添加类型定义
2. 为所有相关模板编写显式实例化
3. 添加对应的测试用例

## Git 配置

提交前请确保 Git 用户配置正确：

```bash
git config user.email "your_github_email@example.com"
git config user.name "Your Name"
```

## 问题反馈

- 使用 GitHub Issues 报告 bug 或提出功能需求
- 描述问题时请包含：环境信息（OS、CUDA 版本、GPU 型号）、复现步骤、期望行为
- 附带最小复现代码或测试用例

## 许可证

PlaMatrix 使用 MIT 许可证。贡献代码即表示同意将代码以 MIT 许可证发布。
