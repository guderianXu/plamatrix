# PlaMatrix 设计规格

## 1. 项目概述

PlaMatrix 是一个面向点云处理的高性能矩阵运算库，利用 CUDA/cuBLAS/cuSOLVER 和多线程 CPU 进行并行计算优化。API 参考 Eigen 风格但更精简，聚焦点云处理场景中需求量最大的矩阵运算。

### 核心目标

- 为千万级点云的大规模处理提供高效的线性代数运算
- 通过 GPU 和多线程 CPU 实现显著的加速比
- 提供简洁的 CMake 集成接口，便于在其他点云项目中调用
- 提供完整中文文档和性能基准测试工具

---

## 2. 技术选型

| 维度 | 选择 |
|------|------|
| 运算范围 | 基础线性代数 + 点云专用运算 + 稀疏矩阵支持 |
| 数据规模 | 大规模 (千万点以上) |
| 浮点精度 | float32 + float64，模板化双精度 |
| API 风格 | 操作符重载 + 函数调用混合 |
| 设备管理 | 显式设备绑定 (GPU/CPU) |
| 硬件目标 | 单 GPU 工作站 + 多线程 CPU |
| 外部依赖 | 仅 CUDA Toolkit + C++ 标准库 |
| 文档语言 | 中文 |

---

## 3. 核心架构

### 3.1 类型系统

```
DeviceMatrix<Scalar, Device>        — 编译期绑定设备的矩阵基类
├── DenseMatrix<Scalar, Device>     — 密集矩阵
└── SparseMatrix<Scalar, Device, Format>
    ├── COOMatrix<Scalar, Device>   — COO 格式稀疏矩阵
    └── CSRMatrix<Scalar, Device>   — CSR 格式稀疏矩阵
```

- `Scalar` = `float` | `double`
- `Device` = `enum class Device { CPU, GPU }`，编译期模板参数，避免运行时分支
- 矩阵构造时分配内存，析构时释放，支持 move 语义，禁止拷贝

### 3.2 内存模型

- **GPU 侧**: `cudaMalloc` 分配，列优先存储 (兼容 cuBLAS Fortran 序)
- **CPU 侧**: `std::vector<Scalar>` 或 aligned allocator，列优先存储
- 提供 `rowMajorView()` / `colMajorView()` 零拷贝视图

### 3.3 设备间数据移动

```cpp
auto A_gpu = DenseMatrix<float, Device::GPU>(1000, 1000);
// GPU 上计算...
auto A_cpu = A_gpu.toCpu();  // 显式、同步、昂贵
```

只有 `toCpu()` / `toGpu()` 触发数据传输，让用户清楚感知代价。数据传输时间不计入计算耗时。

### 3.4 多线程策略

CPU 侧对 `N > 阈值` 的运算自动使用 OpenMP 并行。用户可通过 `PLAMATRIX_NUM_THREADS` 环境变量或 `setNumThreads(n)` 函数覆盖。

---

## 4. API 设计

### 4.1 操作符重载 (密集矩阵)

```cpp
auto C = A * B;              // 矩阵乘法 → cuBLAS gemm / CPU gemm
auto D = A + B;              // 逐元素加法
auto E = A.transpose();      // 转置 (零拷贝视图)
auto F = 2.0f * A + B;       // 标量乘 + 加法
```

实现方式: 表达式模板，编译期构建表达式树，`operator=` 时一次性求值。

支持的融合模式:
1. 二元链: `A * B + C` → 融合为 `gemm` + `axpy`
2. 标量广播: `alpha * A + beta` → 直接 `geam`

更复杂的链直接求值为临时矩阵，保持实现复杂度可控。

### 4.2 函数调用式 (分解 / 求解 / 点云专用)

```cpp
// 矩阵分解
auto [U, S, Vt] = svd(A);                   // cuSOLVER gesvd / gesvdj
auto [Q, R] = qr(A);
auto eigenvalues = eigh(A);                 // 对称矩阵特征值

// 线性求解
auto X = solve(A, b);                       // 密集求解器
auto X = solve(sparseA, b);                 // 稀疏求解器

// 点云专用
auto R = rotationMatrix(axis, angle);       // 轴角 → 3x3 旋转矩阵
auto T = rigidTransform(R, t);              // 4x4 齐次变换矩阵
auto pts_t = transformPoints(T, pts);       // 批量点变换 (Nx3)
auto C = covarianceMatrix(pts);             // 点云 3x3 协方差矩阵
```

### 4.3 稀疏矩阵

```cpp
auto coo = COOMatrix<float, Device::GPU>(n, n);
coo.add(0, 0, 3.14f);
coo.add(1, 2, 2.71f);
auto csr = coo.toCsr();                     // COO → CSR 转换
auto x = solve(csr, b);                     // cuSOLVER 稀疏求解
```

### 4.4 CUDA Stream

```cpp
cudaStream_t s1;
cudaStreamCreate(&s1);
auto C = matMul(A, B, s1);                  // 在指定 stream 上执行
cudaStreamSynchronize(s1);
cudaStreamDestroy(&s1);
```

库函数不创建/销毁 stream，只接受用户传入的 stream。

### 4.5 错误处理宏

```cpp
PLAMATRIX_CHECK_CUDA(call)
PLAMATRIX_CHECK_CUBLAS(call)
PLAMATRIX_CHECK_CUSOLVER(call)
```

- CPU 侧: 异常 (`std::runtime_error`)
- GPU 侧: CUDA API 调用失败通过宏包装 → 提取文件名/行号 → 抛异常

---

## 5. 项目结构

```
plamatrix/
├── include/plamatrix/
│   ├── core/
│   │   ├── types.h              # Scalar, Device, 基础类型定义
│   │   ├── allocator.h           # 内存分配器
│   │   └── device_matrix.h       # DeviceMatrix 基类
│   ├── dense/
│   │   ├── dense_matrix.h        # DenseMatrix
│   │   └── dense_ops.h           # 逐元素 / 矩阵乘法
│   ├── sparse/
│   │   ├── coo_matrix.h          # COO 格式
│   │   └── csr_matrix.h          # CSR 格式
│   ├── ops/
│   │   ├── gemm.h                # 通用矩阵乘法
│   │   ├── decomposition.h       # SVD / QR / Eigh
│   │   ├── solver.h              # 线性求解器
│   │   └── point_cloud.h         # 点云专用运算
│   └── plamatrix.h               # 总头文件
├── src/
│   ├── core/
│   ├── dense/
│   ├── sparse/
│   └── ops/
│       └── kernels/              # CUDA kernel (.cu 文件)
├── test/
│   ├── unit/                     # 单元测试
│   │   ├── core/
│   │   ├── dense/
│   │   ├── sparse/
│   │   └── ops/
│   ├── integration/              # CPU/GPU 一致性测试
│   └── fixtures/                 # 共享测试数据
├── benchmark/
│   ├── main.cpp                  # 统一测试入口
│   └── benchmark_cases.cpp       # 各运算的三模测试
├── docs/
│   ├── index.md
│   ├── build.md
│   ├── api/
│   │   ├── dense-matrix.md
│   │   ├── sparse-matrix.md
│   │   ├── linear-algebra.md
│   │   └── point-cloud.md
│   ├── examples/
│   │   ├── basic-operations.cpp
│   │   ├── point-transform.cpp
│   │   └── sparse-solver.cpp
│   └── contributing.md
├── cmake/
│   └── plamatrixConfig.cmake.in
├── CMakeLists.txt
└── README.md
```

---

## 6. CMake 构建系统

### 6.1 核心选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_CUDA_ARCH` | native | CUDA 架构目标 |
| `PLAMATRIX_USE_FLOAT` | ON | 预编译 float32 实例化 |
| `PLAMATRIX_USE_DOUBLE` | ON | 预编译 float64 实例化 |
| `BUILD_TESTS` | OFF | 构建单元测试 |
| `BUILD_BENCHMARKS` | OFF | 构建性能基准测试 |
| `BUILD_DOCS` | OFF | 生成文档 |

### 6.2 用户集成

安装模式:
```cmake
find_package(plamatrix REQUIRED)
target_link_libraries(my_project plamatrix::plamatrix)
```

源码集成:
```cmake
add_subdirectory(plamatrix)
target_link_libraries(my_project plamatrix::plamatrix)
```

或通过 `FetchContent` 拉取。

---

## 7. 测试策略

### 7.1 三层测试

每个运算用例必须在三种环境下测试:

1. **串行 CPU** — `omp_set_num_threads(1)`
2. **多线程 CPU** — `omp_set_num_threads(max)`
3. **CUDA GPU** — 完整 GPU 路径

### 7.2 校验标准

- 三种路径的结果一致性: float 容差 `1e-4`, double 容差 `1e-10`
- 边界测试: 零矩阵、单行/单列、非方阵、超大矩阵

### 7.3 命名规范

- 测试文件: `*_test.cpp`
- 测试用例: `ClassName_MethodName_Scenario`
- 测试框架: Google Test + CUDA 测试宏

---

## 8. 统一基准测试程序

### 8.1 可执行程序: `plamatrix_benchmark`

```
用法:
  plamatrix_benchmark [选项]

选项:
  --mode all|cpu|cuda    测试模式 (默认: all)
  --size small|medium|large  矩阵尺寸档位 (默认: medium)
  --output FILE          报告输出路径 (默认: benchmark_report.md)
  --list                 列出所有测试用例
```

### 8.2 尺寸档位

| 档位 | 小矩阵 | 中矩阵 | 大矩阵 |
|------|--------|--------|--------|
| small | 128, 256 | 512 | 1024 |
| medium | 512, 1024 | 2048 | 4096 |
| large | 2048, 4096 | 8192 | 16384 |

### 8.3 基准测试流程

每个用例:

1. 预热 3 次 (不计时)
2. 正式测试 10 次取中位数
3. CPU→GPU 数据传输时间单独记录 (不计入 GPU 计算时间)
4. 三种模式分别记录耗时

### 8.4 报告输出

自动生成 `benchmark_report.md`，包含:

1. **环境信息**: CPU 型号/核数, GPU 型号/显存/驱动版本, CUDA 版本, OS
2. **每运算的三模耗时表** + 加速比

```
### gemm

| 尺寸  | CPU串行 | CPU多线程 | CUDA   | 多线程加速比 | CUDA加速比 |
|-------|---------|----------|--------|-------------|-----------|
| 256   | 1.2ms   | 0.8ms    | 0.3ms  | 1.5x        | 4.0x      |
| 1024  | 45ms    | 12ms     | 2.1ms  | 3.8x        | 21.4x     |
| 4096  | 1200ms  | 280ms    | 38ms   | 4.3x        | 31.6x     |
```

3. **汇总分析**: GPU 建议起步尺寸, 多线程饱和核数, CUDA vs 多线程 CPU 的交叉点
4. **复现指令**: `./plamatrix_benchmark --mode all --size large --output report.md`

### 8.5 不同硬件用户的使用场景

- 无 GPU 用户: `--mode cpu` 对比串行 vs 多线程
- 有 GPU 用户: `--mode all` 完整对比
- 性能评估: 不同档位跑完得到完整报告，直接对照硬件判断最优路径

---

## 9. 文档结构

```
docs/
├── index.md                      # 总览, 5 分钟快速上手
├── build.md                      # 编译依赖, CMake 选项, 各平台步骤
├── api/
│   ├── dense-matrix.md           # DenseMatrix 构造/赋值/基本操作
│   ├── sparse-matrix.md          # COO/CSR 构建与转换
│   ├── linear-algebra.md         # gemm/svd/qr/solve 等
│   └── point-cloud.md            # 点云专用运算
├── examples/
│   ├── basic-operations.cpp
│   ├── point-transform.cpp
│   └── sparse-solver.cpp
└── contributing.md               # 编码规范, TDD 流程, PR 流程
```

所有头文件公开 API 使用 `///` 文档注释，Doxygen 可生成 API 参考。

---

## 10. 编码规范

### 命名规范

| 元素 | 风格 | 示例 |
|------|------|------|
| 类/结构体 | 大驼峰 PascalCase | `DenseMatrix`, `CudaAllocator` |
| 函数/方法 | 小驼峰 camelCase | `toCpu()`, `solveLinear()` |
| 变量 | 全小写下划线 snake_case | `row_count`, `data_ptr` |
| 私有成员变量 | 下划线前缀 | `_data`, `_rows` |
| 常量/宏 | 全大写+下划线 | `MAX_THREADS`, `PLAMATRIX_VERSION` |
| 模板参数 | 大写驼峰或单大写字母 | `Scalar`, `Device` |
| 命名空间 | 全小写单单词 | `plamatrix` |

### 格式规范

- 4 空格缩进，不使用 tab
- 花括号 Allman 风格 (左花括号独占一行)
- 行宽上限 120 字符
- 头文件使用 `.h` 扩展名, CUDA 源文件 `.cu`, 纯 C++ `.cpp`
- 使用 `#pragma once`

### include 顺序

1. 标准库头文件
2. CUDA 头文件
3. 第三方库头文件
4. 项目头文件

每组之间空行分隔，组内按字母序。

### 文件组织

- 单文件不超过 400 行，超了拆分
- 嵌套不超过 4 层
- 头文件公开函数必须写 `///` 文档注释

### 错误处理

- CPU 侧: 异常 (`std::runtime_error`)
- GPU 侧: kernel 错误码 + `PLAMATRIX_CHECK_CUDA(call)` 宏 → 抛异常

### 测试规范

- 测试文件: `*_test.cpp`
- 测试用例: `ClassName_MethodName_Scenario`
- 测试框架: Google Test

---

## 11. 开发流程

### 分支策略

- `main` 分支始终稳定可构建
- 功能/修复在 `feat/<描述>` 或 `fix/<描述>` 分支上开发
- TDD 循环完成后合并回 main

### TDD 铁律

1. 先写测试 → 确认失败 (红)
2. 写最小实现 → 确认通过 (绿)
3. 重构优化 → 保持通过
4. 提交前跑相关测试

### 编译验证

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
```

编译失败不能提交，警告需评估。
