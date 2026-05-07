# 编译指南

## 系统要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| CMake | 3.18+ | 构建系统 |
| GCC | 9.0+ | 需支持 C++17 |
| CUDA Toolkit | 11.0+ | GPU 加速（可选，仅 CPU 模式不需要） |
| OpenMP | 4.5+ | CPU 多线程（通常随 GCC 安装） |
| Google Test | 1.11+ | 单元测试框架（仅 `BUILD_TESTS=ON` 时需要） |

## CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_CUDA_ARCH` | `native` | CUDA 架构目标，如 `80` (A100), `89` (RTX 4090) |
| `PLAMATRIX_USE_FLOAT` | `ON` | 启用 float32 实例化 |
| `PLAMATRIX_USE_DOUBLE` | `ON` | 启用 float64 实例化 |
| `BUILD_TESTS` | `OFF` | 构建并运行单元测试 |
| `BUILD_BENCHMARKS` | `OFF` | 构建性能基准测试工具 |

## 编译步骤

### 标准构建 (含 GPU 支持)

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
```

### 仅 CPU 模式 (无需 CUDA)

```cmake
cmake .. -DPLAMATRIX_USE_FLOAT=ON -DPLAMATRIX_USE_DOUBLE=ON -DBUILD_TESTS=ON
```

注意：需要移除 `CMakeLists.txt` 中的 `LANGUAGES CUDA` 并通过条件判断跳过 CUDA 依赖。

### 指定 CUDA 架构

```bash
cmake .. -DPLAMATRIX_CUDA_ARCH=89 -DBUILD_TESTS=ON
```

## 运行测试

```bash
cd build
cmake --build . -j$(nproc)
./test/plamatrix_tests
```

## 运行基准测试

```bash
# CPU 对比 (串行 vs 多线程)
./benchmark/plamatrix_benchmark --mode cpu --size medium

# 完整对比 (CPU + GPU)
./benchmark/plamatrix_benchmark --mode all --size large --output report.md
```

## 安装

```bash
cmake --install . --prefix /path/to/install
```

安装后目录结构：
```
/path/to/install/
├── include/plamatrix/    # 头文件
├── lib/libplamatrix.a    # 静态库
└── lib/cmake/plamatrix/  # CMake 配置文件
```
