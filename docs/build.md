# 编译指南

## 环境要求

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| CUDA Toolkit | 11.0 | GPU 计算、cuBLAS、cuSOLVER |
| CMake | 3.18 | 构建系统（CUDA 语言支持） |
| GCC | 9.0 | C++17 编译（CPU 路径） |
| OpenMP | 4.5 | CPU 多线程并行 |
| Google Test | 1.10 | 单元测试（仅 BUILD_TESTS=ON 时需要） |

### 验证环境

```bash
# 检查 CUDA
nvcc --version

# 检查 CMake
cmake --version

# 检查 GCC
gcc --version

# 检查 OpenMP（通常随 GCC 安装）
echo | gcc -fopenmp -x c - -o /dev/null 2>&1 || echo "OpenMP 不可用"
```

## CMake 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `PLAMATRIX_USE_FLOAT` | BOOL | ON | 预编译 float32 模板实例化。禁用可缩短编译时间 |
| `PLAMATRIX_USE_DOUBLE` | BOOL | ON | 预编译 float64 模板实例化。禁用可缩短编译时间 |
| `PLAMATRIX_CUDA_ARCH` | STRING | `native` | CUDA 架构目标。可设置为逗号分隔列表，如 `70;75;80` |
| `BUILD_TESTS` | BOOL | OFF | 构建 Google Test 单元测试 |
| `BUILD_BENCHMARKS` | BOOL | OFF | 构建基准测试可执行文件 |
| `BUILD_DOCS` | BOOL | OFF | （预留）构建 API 文档 |

### CUDA 架构选择

- `native`：自动检测当前 GPU 架构，仅生成当前设备代码。适合开发和测试
- 具体值：如 `70` (V100), `75` (T4), `80` (A100), `86` (RTX 3090), `89` (RTX 4090)
- 逗号分隔列表：`70;75;80` 为多个架构生成代码（二进制文件更大）

```bash
# 指定 A100 架构
cmake .. -DPLAMATRIX_CUDA_ARCH=80

# 支持多种架构
cmake .. -DPLAMATRIX_CUDA_ARCH="70;75;80"
```

## 编译步骤

### 1. 克隆仓库

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
```

### 2. 配置 CMake

```bash
mkdir build && cd build

# 最小构建（仅库）
cmake ..

# 带测试和基准测试
cmake .. -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

# Release 构建
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
```

### 3. 编译

```bash
cmake --build . -j$(nproc)
```

### 4. 运行测试（可选）

```bash
ctest --output-on-failure
```

### 5. 安装

```bash
# 安装到系统默认目录（通常需要 sudo）
sudo cmake --install .

# 安装到指定目录
cmake --install . --prefix /path/to/install
```

安装后的目录结构：

```
<prefix>/
├── include/
│   └── plamatrix/
│       ├── plamatrix.h
│       ├── core/
│       ├── dense/
│       ├── sparse/
│       └── ops/
├── lib/
│   └── libplamatrix.a
└── lib/cmake/plamatrix/
    ├── plamatrixConfig.cmake
    └── plamatrixTargets.cmake
```

## 平台说明

### Linux（主要支持平台）

PlaMatrix 主要面向 Linux 开发环境。所有测试和基准测试在 Linux 下验证通过。

- 推荐 Ubuntu 20.04+ 或等效发行版
- 使用系统包管理器安装 CUDA Toolkit
- OpenMP 通常由 GCC 自带，无需额外安装

### Windows

目前未对 Windows 平台进行完整验证。如需在 Windows 上使用：

- 需要 MSVC 2019+ 或 MinGW-w64
- CUDA 需通过 NVIDIA 官方安装
- 部分 OpenMP 配置可能需要调整

### macOS

macOS 不支持 CUDA。如需在 macOS 上开发：

- 仅 CPU 路径可用（需禁用 GPU 编译）
- 使用 Apple Clang 编译，OpenMP 需通过 Homebrew 安装 `libomp`

## 常见问题

### CUDA 找不到

```
CMake Error: Could not find a package configuration file for "CUDAToolkit"
```

**解决方案**：
- 确认 CUDA Toolkit 已正确安装：`nvcc --version`
- 如果在非标准路径，设置 `CUDAToolkit_ROOT`：
  ```bash
  cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda-11.8
  ```

### OpenMP 找不到

```
CMake Error: Could not find OpenMP
```

**解决方案**：
- Ubuntu/Debian：`sudo apt install libomp-dev`
- 确认 GCC 版本 >= 9.0

### CUDA 架构不匹配

```
CUDA error: no kernel image is available for execution on the device
```

**解决方案**：
- 设置 `PLAMATRIX_CUDA_ARCH` 为你的 GPU 架构或使用 `native`
- 查询 GPU 架构：`nvidia-smi --query-gpu=compute_cap --format=csv`

### 编译时间过长

- 禁用不需要的精度类型：`-DPLAMATRIX_USE_DOUBLE=OFF`
- 仅编译当前 GPU 架构：`-DPLAMATRIX_CUDA_ARCH=native`
- 使用 `-j$(nproc)` 并行编译

### 链接时的模板实例化错误

如果看到 `undefined reference to plamatrix::gemm<float>(...)` 类似的错误，说明模板实例化未找到。确保：

1. 库编译时启用了对应的精度类型（`PLAMATRIX_USE_FLOAT` / `PLAMATRIX_USE_DOUBLE`）
2. 使用 `target_link_libraries` 正确链接 `plamatrix::plamatrix`
