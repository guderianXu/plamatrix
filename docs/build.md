# 编译指南

本文档说明 Linux/CUDA、macOS/Metal 和无 GPU 后端的常用构建方式。

## 1. 系统要求

| 组件 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 20.04+ / Debian 11+ / RHEL 9+ / macOS 12+ | Linux x86_64 或 Apple Silicon/Intel macOS |
| 编译器 | GCC 9+ / AppleClang | 需要 C++17 支持；Metal 后端需要 Objective-C++ |
| CMake | 3.18+ | 构建系统 |
| CUDA Toolkit | 11.0+ | GPU 加速（可选，无 NVIDIA GPU 可跳过） |
| Metal / MPS | 系统框架 | macOS GPU 后端（随 Xcode Command Line Tools 提供） |
| OpenMP | 4.5+ | CPU 多线程（可选；找不到时自动使用串行 fallback） |
| Google Test | 1.11+ | 单元测试（仅 `-DPLAMATRIX_BUILD_TESTS=ON` 时需要） |

---

## 2. 从零搭建 (Ubuntu 24.04)

### 2.1 基础编译工具

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

验证安装：

```bash
gcc --version    # 需要 9.0+
cmake --version  # 需要 3.18+
```

如果 CMake 版本过低，用官方源升级：

```bash
wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo tee /etc/apt/trusted.gpg.d/kitware.asc
echo "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update && sudo apt install -y cmake
```

### 2.2 NVIDIA 驱动 + CUDA Toolkit

**确认是否有 NVIDIA GPU：**

```bash
lspci | grep -i nvidia
```

如果没有输出，Linux 上可跳到 [2.3 无 GPU 后端构建](#23-无-gpu-后端构建)；macOS 见下一节的 Metal 构建。

#### 方式一：通过 apt 安装（推荐）

```bash
# 1. 安装 NVIDIA 驱动 580
sudo apt update
sudo apt install -y nvidia-driver-580
sudo reboot

# 2. 安装 CUDA Toolkit 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# 3. 添加到 PATH（写入 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

验证 CUDA：

```bash
nvidia-smi           # 应显示 GPU 和 Driver Version: 580.xxx
nvcc --version       # 应显示 CUDA 12.8
```

**如果本机有旧版 CUDA，先清理：**

```bash
# 查看已安装的 CUDA 包
dpkg -l | grep cuda

# 卸载旧版
sudo apt purge -y cuda-toolkit-12-0 'cuda-*'
sudo apt autoremove -y

# 删除残留目录
sudo rm -rf /usr/local/cuda /usr/local/cuda-12.0 /usr/local/cuda-12

# 然后按上面步骤重新安装 12.8
```

#### 方式二：通过 runfile 安装（离线或指定版本）

从 [NVIDIA CUDA Toolkit 下载页](https://developer.nvidia.com/cuda-downloads) 选择对应系统的 runfile，下载后：

```bash
# 先停掉图形界面
sudo systemctl isolate multi-user.target

# 安装（跳过驱动如果已安装）
sudo sh cuda_12.8.0_580.xxx_linux.run --toolkit --silent

# 重启
sudo reboot
```

> **版本对应关系**：驱动 580 最高支持 CUDA 13.0。我们使用 CUDA 12.8（成熟稳定版）。`nvidia-smi` 右上角显示的 "CUDA Version" 是驱动支持的最高 CUDA 版本，不是已安装的 toolkit 版本。

### 2.3 macOS Metal 构建

macOS 上 `PLAMATRIX_GPU_BACKEND=AUTO` 默认选择 Metal/MPS。用户代码仍使用 `DenseMatrix<Scalar, Device::GPU>`，不需要改成平台专用类型。

```bash
xcode-select --install
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
cmake -S . -B build-metal -DPLAMATRIX_BUILD_TESTS=ON
cmake --build build-metal -j$(sysctl -n hw.ncpu)
ctest --test-dir build-metal --output-on-failure
```

也可以显式指定：

```bash
cmake -S . -B build-metal -DPLAMATRIX_GPU_BACKEND=METAL -DPLAMATRIX_BUILD_TESTS=ON
```

Metal 后端当前策略：

- `float` 的 dense fill/transpose/add/sub、GEMM、solver 和点变换使用 Metal/MPS。
- `double` GPU API 在 macOS 上使用 CPU/共享内存 fallback，保持 API 可用。
- SVD/QR/eigh 在 Metal 后端首版使用 CPU fallback 后转回 GPU 矩阵。

### 2.4 无 GPU 后端构建

如果不希望启用任何 GPU 后端，PlaMatrix 支持纯 CPU / stub 编译。此时 GPU 存储/传输桩可让头文件和 CPU 测试正常编译，但 GPU 算法入口会抛出明确异常；业务代码应使用 `Device::CPU` 路径。

```bash
# 只需要基础工具（无需 CUDA）
sudo apt install -y build-essential cmake git

# 编译时关闭所有 GPU 后端
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
cmake -S . -B build-none -DPLAMATRIX_GPU_BACKEND=NONE -DPLAMATRIX_BUILD_TESTS=ON
cmake --build build-none -j$(nproc)
```

旧写法 `-DPLAMATRIX_WITH_CUDA=OFF` 仍兼容；新项目建议使用 `PLAMATRIX_GPU_BACKEND=NONE`。

### 2.5 Google Test（运行测试时需要）

```bash
sudo apt install -y libgtest-dev
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp lib/libgtest*.a /usr/lib
```

或者通过源码编译（需要特定版本时）：

```bash
git clone https://github.com/google/googletest.git -b v1.14.0
cd googletest
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

---

## 3. 编译 PlaMatrix

### 3.1 完整构建（含 GPU + 测试 + 基准测试）

```bash
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
mkdir build && cd build
cmake .. -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
```

### 3.2 CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `PLAMATRIX_GPU_BACKEND` | `AUTO` | GPU 后端：`AUTO`、`CUDA`、`METAL`、`NONE` |
| `PLAMATRIX_WITH_CUDA` | 自动检测 | 旧 CUDA 开关；`ON` 等价于请求 CUDA，`OFF` 禁止 AUTO 选择 CUDA |
| `PLAMATRIX_WITH_METAL` | `ON` | macOS AUTO 模式下允许选择 Metal |
| `PLAMATRIX_WITH_OPENMP` | `AUTO` | OpenMP：`AUTO`、`ON`、`OFF` |
| `PLAMATRIX_CUDA_ARCHITECTURES` | `75;86;89` | CUDA 架构目标。可设为具体值如 `80`(A100)、`86`(RTX3090)、`89`(RTX4090) |
| `PLAMATRIX_WITH_SYSTEM_LINALG` | `ON` | 通过 CMake `find_package(BLAS/LAPACK)` 检测并使用系统 BLAS/LAPACK 加速 CPU GEMM/SVD/eigh |
| `PLAMATRIX_USE_FLOAT` | `ON` | 启用 `float` (32-bit) 实例化 |
| `PLAMATRIX_USE_DOUBLE` | `ON` | 启用 `double` (64-bit) 实例化 |
| `PLAMATRIX_BUILD_TESTS` | `OFF` | 构建单元测试 (需要 Google Test) |
| `PLAMATRIX_BUILD_BENCHMARKS` | `OFF` | 构建性能基准测试 |

独立顶层构建时仍兼容 `BUILD_TESTS` / `BUILD_BENCHMARKS` 短名；作为 `add_subdirectory()` 子项目集成时应使用 `PLAMATRIX_BUILD_*` 选项。

### 3.3 常用构建组合

```bash
# 仅核心库（无测试、无 benchmark，最简构建）
cmake .. && cmake --build . -j$(nproc)

# macOS Metal 版本
cmake .. -DPLAMATRIX_GPU_BACKEND=METAL && cmake --build . -j$(sysctl -n hw.ncpu)

# 无 GPU 后端版本
cmake .. -DPLAMATRIX_GPU_BACKEND=NONE && cmake --build . -j$(nproc)

# 不使用系统 BLAS/LAPACK，验证项目内 fallback 数值路径
cmake .. -DPLAMATRIX_GPU_BACKEND=NONE -DPLAMATRIX_WITH_SYSTEM_LINALG=OFF
cmake --build . -j$(nproc)

# 开发模式（测试 + benchmark）
cmake .. -DPLAMATRIX_BUILD_TESTS=ON -DPLAMATRIX_BUILD_BENCHMARKS=ON && cmake --build . -j$(nproc)

# 生产安装
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build . -j$(nproc)
sudo cmake --install .
```

---

## 4. 验证

### 运行测试

```bash
cd build
./test/plamatrix_tests
```

默认构建会按平台选择 GPU 后端：macOS 运行 Metal 与 CPU/GPU 一致性测试，CUDA 构建运行 CUDA 测试；`PLAMATRIX_GPU_BACKEND=NONE` 会跳过 GPU 专属用例并运行 stub 回归。

### 运行基准测试

```bash
# 快速 smoke：适合日常回归
./benchmark/plamatrix_benchmark --mode cpu --size smoke

# CPU 对比（串行 vs 多线程）
./benchmark/plamatrix_benchmark --mode cpu --size small

# 只跑指定 case
./benchmark/plamatrix_benchmark --mode all --size small --case gemm,covariance

# 完整对比（CPU + GPU）
./benchmark/plamatrix_benchmark --mode all --size small --output report.md

# 只跑当前平台 GPU 后端；--mode cuda 仍可作为兼容别名
./benchmark/plamatrix_benchmark --mode gpu --size smoke --case transpose
```

### 安装到系统

```bash
sudo cmake --install build --prefix /usr/local
```

安装后目录结构：

```
/usr/local/
├── include/plamatrix/   # 头文件
├── lib/libplamatrix.a   # 静态库
└── lib/cmake/plamatrix/ # CMake 包配置
```

其他项目通过 `find_package(plamatrix)` 即可引用。

---

## 5. 常见问题

### CMake 找不到 CUDA

```
CMake Error: Could not find a package configuration file for "CUDAToolkit"
```

**解决**：CUDA 未安装或未加入 PATH。

```bash
export PATH=/usr/local/cuda/bin:$PATH
# 或
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Google Test not found

**解决**：按 [2.4 节](#24-google-test运行测试时需要) 安装，或者不构建测试：

```bash
cmake .. -DPLAMATRIX_BUILD_TESTS=OFF
```

### POSIX memalign not supported（macOS）

macOS 使用 `malloc` 而不是 `posix_memalign`。在 `allocator.h` 中将 `posix_memalign` 替换为 `aligned_alloc`，或使用 Linux 环境。

### 编译时 OOM（内存不足）

大规模矩阵实例化时编译消耗内存。减少并行编译线程：

```bash
cmake --build . -j2  # 只用 2 线程
```

### GPU 基准测试不可用

如果是 CUDA 后端，确认驱动已加载：

```bash
nvidia-smi          # 检查驱动状态
ls /dev/nvidia*     # 检查设备节点
```

如果是 CPU-only 构建，`--mode gpu` 会被拒绝；重新配置时使用 `-DPLAMATRIX_GPU_BACKEND=AUTO`、`CUDA` 或 `METAL`。

### 运行时 OpenMP 库冲突

如果看到 `libgomp.so.1 may be hidden` 警告，设置 `LD_LIBRARY_PATH` 排除 anaconda 路径：

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
