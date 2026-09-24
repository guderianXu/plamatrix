# 编译指南

本文档假设你有一台全新的 Linux 电脑，从零开始搭建编译环境。

## 1. 系统要求

| 组件 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 20.04+ / Debian 11+ / RHEL 9+ | Linux x86_64，其他发行版同理 |
| GCC | 9.0+ | 需要 C++17 支持 |
| CMake | 3.18+ | 构建系统 |
| CUDA Toolkit | 11.0+ | GPU 加速（可选，无 NVIDIA GPU 可跳过） |
| OpenMP | 4.5+ | CPU 多线程（随 GCC 安装，无需额外操作） |
| Google Test | 1.11+ | 单元测试（仅 `-DPLAMATRIX_BUILD_TESTS=ON` 时需要） |

CUDA 构建应让 `CMAKE_CXX_COMPILER` 与 `CMAKE_CUDA_HOST_COMPILER` 使用一致的工具链和系统库；
不要混用 Conda 的旧 sysroot 与系统 GCC 的 CUDA host 编译结果。驻留矩阵视图与克隆测试包含在
`plamatrix_tests`，有限值统计视图测试包含在 `plamatrix_statistics_tests`；CPU-only 构建覆盖
不可用后端的明确报错行为。

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

如果没有输出，说明没有 NVIDIA GPU，跳到 [2.3 CPU-only 编译](#23-cpu-only-编译无需-nvidia-gpu)。

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

### 2.3 CPU-only 编译（无需 NVIDIA GPU）

如果没有 NVIDIA GPU，PlaMatrix 支持纯 CPU 编译。此时 CUDA 存储/传输桩可让头文件和 CPU 测试正常编译，但 `.cu` 中的 GPU 算法不会构建；业务代码应使用 `Device::CPU` 路径。

```bash
# 只需要基础工具（无需 CUDA）
sudo apt install -y build-essential cmake git

# 编译时关闭 CUDA 支持
git clone https://github.com/guderianXu/plamatrix.git
cd plamatrix
mkdir build && cd build
cmake .. -DPLAMATRIX_WITH_CUDA=OFF -DPLAMATRIX_BUILD_TESTS=ON
cmake --build . -j$(nproc)
```

### 2.4 Google Test（运行测试时需要）

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
| `PLAMATRIX_WITH_CUDA` | 自动检测 | GPU 加速，无 NVIDIA GPU 设为 `OFF` |
| `PLAMATRIX_CUDA_ARCHITECTURES` | `75;86;89` | CUDA 架构目标。可设为具体值如 `80`(A100)、`86`(RTX3090)、`89`(RTX4090) |
| `PLAMATRIX_WITH_OPENCL` | `ON` | OpenCL 1.2 执行后端，依赖缺失时自动关闭 |
| `PLAMATRIX_WITH_VULKAN` | `OFF` | Vulkan 1.1 Compute 后端；显式开启时需要 Vulkan SDK/loader 与 `glslangValidator` |
| `PLAMATRIX_USE_FLOAT` | `ON` | 启用 `float` (32-bit) 实例化 |
| `PLAMATRIX_USE_DOUBLE` | `ON` | 启用 `double` (64-bit) 实例化 |
| `PLAMATRIX_BUILD_TESTS` | `OFF` | 构建单元测试 (需要 Google Test) |
| `PLAMATRIX_BUILD_BENCHMARKS` | `OFF` | 构建性能基准测试 |

独立顶层构建时仍兼容 `BUILD_TESTS` / `BUILD_BENCHMARKS` 短名；作为 `add_subdirectory()` 子项目集成时应使用 `PLAMATRIX_BUILD_*` 选项。

### 3.4 Vulkan Compute 构建与 OpenCL 对比

Vulkan 后端只使用 Compute API，不需要窗口系统。配置时 GLSL shader 会由
`glslangValidator` 编译成 SPIR-V：

```bash
cmake .. -DPLAMATRIX_WITH_CUDA=OFF -DPLAMATRIX_WITH_OPENCL=ON \
  -DPLAMATRIX_WITH_VULKAN=ON -DPLAMATRIX_BUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
PLAMATRIX_VULKAN_DEVICE_INDEX=0 ./benchmark/plamatrix_backend_compare 4096
# 多规模大矩阵套件
PLAMATRIX_VULKAN_DEVICE_INDEX=0 ./benchmark/plamatrix_backend_compare --suite
```

对比程序输出 OpenCL/Vulkan 的设备名、实际 CSR 维度和非零元数、迭代次数、初始/最终残差、
端到端 PCG 中位时间、Vulkan 的 `command_submissions`、`command_buffer_recordings`、`gpu_ms`、
`barrier_count`、`spmv_kernel`、`cold_descriptor_set_allocations` 和
`cold_command_buffer_recordings`。这些列用于观察显式 queue submit/fence wait、命令录制复用、GPU 实际
执行时间、实际执行的 buffer 依赖屏障数量以及冷启动 descriptor set 创建数量；OpenCL 当前填 `0`，
不代表它没有内部命令提交。Vulkan 第一阶段只实现 float32 PCG，
结果不代表其他算子（例如 GEMM 或 SVD）的后端性能。Vulkan PCG 的标量递推和收敛状态驻留在
GPU 状态缓冲区，按 `convergenceCheckInterval` 批量提交；`command_submissions` 可用于确认批量边界
带来的同步变化，`command_buffer_recordings` 小于提交数时表示完整迭代批次发生了复用。命令上下文按
buffer 的读写方向记录依赖，只在存在写后读/写冲突时插入屏障；`gpu_ms`
与端到端时间的差值可用于判断瓶颈是否仍在主机提交和 fence wait。
支持 subgroup arithmetic 的设备会在 CSR 平均行长达到 `max(16, subgroupSize / 2)` 时自动使用
subgroup-per-row SpMV；短行 CSR 使用 scalar-per-row 内核。上传、初始残差与容差初始化和首批 PCG
迭代共享一次提交。解向量不超过 64 KiB 时，最终解回读也合并进批次提交；更大向量继续单独下载，
避免在多个收敛批次中重复回读完整结果。
`PLAMATRIX_VULKAN_SPMV=auto|scalar|subgroup|block` 可覆盖自动选择，
其中强制 `subgroup` 在设备不支持时会返回明确错误。
设备常驻的完整 9×9 block CSR 默认用 GPU 构造 block-Jacobi 逆块；
`PLAMATRIX_VULKAN_BLOCK_JACOBI=auto|cpu|gpu` 可覆盖该选择，其中 `gpu` 在条件不满足时返回明确错误。

配置阶段还会试编译 KHR cooperative-matrix shader。试编译成功不代表所选 GPU 一定支持该路径；
运行时还会核对 `VK_KHR_cooperative_matrix`、`VK_KHR_shader_float16_int8`、
`VK_KHR_vulkan_memory_model`、16 位 storage buffer、compute stage、subgroup 大小以及
16×16×16 FP16 输入/FP32 累加规格。可通过
`selectedVulkanCooperativeMatrixCapabilities()` 区分硬件支持和当前构建实际启用状态。

完整的 BA Schur 对比使用 9 维相机块和 3 维点块；下面的命令生成 512 个相机、8192 个点、每点 4 次
观测的确定性法方程，并报告总时间、Schur 装配、线性求解、Vulkan GPU timestamp、命令提交/录制次数
及相对 CPU 解的最大绝对误差：

```bash
./benchmark/plamatrix_schur_backend_benchmark 512 8192 4 7
```

Vulkan Schur 要求所选设备满足上述 cooperative-matrix 能力，且当前只支持显式 opt-in 的 float
混合精度；普通 CSR-PCG 对比仍可在没有 cooperative matrix 的 Vulkan 设备上运行。

除了默认三对角系统，还可以运行更接近 PlaScan 数据形态的场景：

```bash
./benchmark/plamatrix_backend_compare --case stencil2d --sizes 64,128,256
./benchmark/plamatrix_backend_compare --case stencil3d --sizes 16,24,32
./benchmark/plamatrix_backend_compare --case ba_schur --sizes 32,64,128,256
./benchmark/plamatrix_backend_compare --case mvs_visibility --sizes 64,128,256
```

其中 `ba_schur` 的参数是相机数量，每个相机展开为 6 个变量；其它二维/三维场景的参数是网格边长。
内置 BA/MVS 场景是保持实际稀疏拓扑、不规则行长度和正定数值性质的确定性合成数据，并使用非均匀右端项
避免全 1 向量形成过于容易的特征方向。对真实工程数据，
将已经完成 LM damping 的 BA Schur 或 MVS CSR 导出为 MatrixMarket coordinate real general/symmetric
文件，然后运行：

```bash
./benchmark/plamatrix_backend_compare --matrix-market /path/to/real_schur.mtx
```

PlaScan 当前 MVS 主链不持久化 CSR 文件，所以 `mvs_visibility` 是可重复的拓扑代理；真实 MVS CSR
应通过 MatrixMarket 入口接入。

`--suite` 会运行上述五类场景的代表性规模；`cold_ms` 包含首次 shader/工作区路径，
`warm_median_ms` 是复用工作区后的端到端 PCG 时间。

### 3.3 常用构建组合

```bash
# 仅核心库（无测试、无 benchmark，最简构建）
cmake .. && cmake --build . -j$(nproc)

# CPU-only 版本
cmake .. -DPLAMATRIX_WITH_CUDA=OFF && cmake --build . -j$(nproc)

# CPU 线性代数始终使用自包含的原生实现
cmake .. -DPLAMATRIX_WITH_CUDA=OFF

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

默认 CUDA 构建会运行 CUDA 与 CPU/GPU 一致性测试；CPU-only 构建会跳过 GPU 专属用例并运行 no-CUDA stub 回归。

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
```

### Eigen 5 对照验证

安装 Eigen 5.0 或更新版本后，可打开仅供测试和 benchmark 使用的对照依赖：

```bash
cmake -S . -B build/eigen-reference \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLAMATRIX_WITH_CUDA=OFF \
  -DPLAMATRIX_WITH_OPENCL=OFF \
  -DPLAMATRIX_WITH_VULKAN=OFF \
  -DPLAMATRIX_BUILD_TESTS=ON \
  -DPLAMATRIX_BUILD_BENCHMARKS=ON \
  -DPLAMATRIX_WITH_EIGEN_REFERENCE=ON
cmake --build build/eigen-reference --parallel
ctest --test-dir build/eigen-reference --output-on-failure -R EigenReference
./build/eigen-reference/benchmark/plamatrix_eigen_compare --size 256 --trials 11
```

若 Eigen 没有安装 CMake package，可额外传入
`-DPLAMATRIX_EIGEN3_INCLUDE_DIR=/path/to/eigen`。CI 或干净环境也可以传入
`-DPLAMATRIX_FETCH_EIGEN_REFERENCE=ON`，由 CMake 下载经过 SHA-256 校验的 Eigen 5.0.1 源码；
该选项会自动启用 `PLAMATRIX_WITH_EIGEN_REFERENCE`，且只用于测试和 benchmark。
配置阶段会检查 Eigen 的语义化主版本以及
`reshaped()`、`canonicalEulerAngles()`；较旧头文件会直接给出错误。该选项不会改变 PlaMatrix 的
公开链接接口或安装包依赖。

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

公开入口为 `<plamatrix/plamatrix.h>`。`include/plamatrix/internal/` 也随模板实现安装，但不是公开兼容性契约。三维向量使用 `Vector3f`/`Vector3d`，旧 `Vec3` 类型和旧头文件路径已删除。启用的 CUDA、OpenCL、Vulkan 链接依赖由包配置自动查找。

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

### GPU 基准测试报 "CUDA not available"

确认驱动已加载：

```bash
nvidia-smi          # 检查驱动状态
ls /dev/nvidia*     # 检查设备节点
```

### 运行时 OpenMP 库冲突

如果看到 `libgomp.so.1 may be hidden` 警告，设置 `LD_LIBRARY_PATH` 排除 anaconda 路径：

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
