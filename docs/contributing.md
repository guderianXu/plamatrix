# 贡献指南

## 编码规范

详见项目根目录 `CLAUDE.md`。

### 命名

| 元素 | 风格 | 示例 |
|------|------|------|
| 类/结构体 | PascalCase | `DenseMatrix` |
| 函数/方法 | camelCase | `toCpu()` |
| 变量 | snake_case | `row_count` |
| 私有成员 | _前缀 | `_data` |
| 常量/宏 | UPPER_CASE | `MAX_THREADS` |
| 模板参数 | CamelCase | `Scalar`, `Device` |

### 格式

- 4 空格缩进
- Allman 花括号 (左花括号独占一行)
- 行宽 ≤ 120 字符
- `#pragma once`

## 开发流程

### TDD (测试驱动开发)

1. **红** — 先写测试，运行确认失败
2. **绿** — 写最小实现，运行确认通过
3. **重构** — 优化代码，保持测试通过

### 分支管理

```bash
git checkout -b feat/<feature-name>  # 从 main 创建功能分支
# ... TDD 开发 ...
git checkout main && git merge feat/<feature-name>
```

### 提交前检查

- [ ] `cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)` 编译通过
- [ ] `./test/plamatrix_tests` 全部测试通过
- [ ] 无编译警告
- [ ] 单文件不超过 400 行

### PR 清单

- [ ] 遵循 TDD (测试 → 实现 → 重构)
- [ ] 公开函数有 `///` 文档注释
- [ ] 代码在对应设备上验证 (CPU / GPU)
- [ ] 新增 API 有使用示例
- [ ] 无新增编译警告
