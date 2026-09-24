# 点云运算 API（内部实现）

本页描述 `plamatrix::internal` 实现接口，需显式包含 `plamatrix/internal/...` 头文件；不属于 Eigen 风格公开契约。BA 业务由 PlaBundle 负责，点云体素归组由 PlaPoint 负责。

## 数据结构

```cpp
template <typename Scalar>
struct PackedVector3
{
    Scalar x{};
    Scalar y{};
    Scalar z{};

    PackedVector3();
    PackedVector3(Scalar x, Scalar y, Scalar z);
    explicit PackedVector3(const std::array<Scalar, 3>& values);
    std::array<Scalar, 3> toArray() const;
};
```

`PackedVector3` 定义于 `<plamatrix/internal/ops/vector.h>`，仅用于内部紧凑布局。应用向量使用 `plamatrix::Vector3f` 或 `plamatrix::Vector3d`。

## 三维向量运算

```cpp
PackedVector3<double> a(std::array<double, 3>{1.0, 2.0, 3.0});
PackedVector3<double> b{4.0, 5.0, 6.0};

auto sum = a + b;
auto difference = a - b;
auto scaled = 2.0 * a;
double projection = dot(a, b);
auto normal = cross(a, b);
double length = norm(normal);
auto unit = normalized(normal, 1.0e-12);
bool valid = isFinite(unit);
auto values = unit.toArray();
```

- `squaredNorm()` 避免不需要的平方根。
- `normalized(value, epsilon)` 在长度不大于 `epsilon` 时返回原向量，不虚构默认方向。
- `isFinite()` 对三个分量逐一拒绝 NaN 和正负无穷。
- 所有运算都是 header-only 标量运算，不分配 `DenseStorage`，也不触发 CPU/GPU 数据传输。

## 旋转矩阵

```cpp
/// Rodrigues 公式: R = I + sin(θ)·K + (1-cos(θ))·K²
/// K 是归一化旋转轴的叉乘矩阵
template <typename Scalar, Device Dev>
DenseStorage<Scalar, Dev> rotationMatrix(
    const PackedVector3<Scalar>& axis,  // 旋转轴 (自动归一化)
    Scalar angle               // 旋转角度 (弧度)
);
```

返回 3×3 旋转矩阵。

```cpp
// 绕 Z 轴旋转 90°
PackedVector3<float> z_axis{0, 0, 1};
auto R = rotationMatrix<float, Device::CPU>(z_axis, M_PI / 2.0);
```

## 刚体变换

```cpp
/// 构建 4×4 齐次变换矩阵: [R|t; 0 0 0 1]
template <typename Scalar, Device Dev>
DenseStorage<Scalar, Dev> rigidTransform(
    const DenseStorage<Scalar, Dev>& R,  // 3×3 旋转矩阵
    const PackedVector3<Scalar>& t                // 平移向量
);
```

## 批量点变换

```cpp
/// 对 N×3 点云施加 4×4 刚体变换
/// 对每个点 p 计算: p' = R·p + t
template <typename Scalar, Device Dev>
DenseStorage<Scalar, Dev> transformPoints(
    const DenseStorage<Scalar, Dev>& T,      // 4×4 刚体变换
    const DenseStorage<Scalar, Dev>& points  // N×3 点云
);

template <typename Scalar>
void transformPoints(
    const DenseStorage<Scalar, Device::GPU>& T,
    const DenseStorage<Scalar, Device::GPU>& points,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr
);

template <typename Scalar>
void transformPointsAsync(
    const DenseStorage<Scalar, Device::GPU>& T,
    const DenseStorage<Scalar, Device::GPU>& points,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr
);
```

返回 N×3 变换后的点云。

- **CPU**：逐点循环计算
- **GPU**：`transformPointsKernel` 并行处理
- `T` 必须是 4×4，`points` 必须是 N×3，输出复用矩阵必须是 N×3
- 同步 GPU 输出复用重载返回前会同步传入 stream；`transformPointsAsync` 只提交 kernel

## 协方差矩阵

```cpp
/// 计算 N×3 点云的 3×3 协方差矩阵
/// 1. 计算质心
/// 2. 中心化
/// 3. C = (1/N) · centered^T · centered
template <typename Scalar, Device Dev>
DenseStorage<Scalar, Dev> covarianceMatrix(
    const DenseStorage<Scalar, Dev>& points  // N×3 点云
);

template <typename Scalar>
void covarianceMatrix(
    const DenseStorage<Scalar, Device::GPU>& points,
    DenseStorage<Scalar, Device::GPU>& output,
    cudaStream_t stream = nullptr
);

template <typename Scalar>
class GpuCovarianceWorkspace;

template <typename Scalar>
void covarianceMatrixAsync(
    const DenseStorage<Scalar, Device::GPU>& points,
    DenseStorage<Scalar, Device::GPU>& output,
    GpuCovarianceWorkspace<Scalar>& workspace,
    cudaStream_t stream = nullptr
);
```

返回 3×3 半正定矩阵。`points` 必须是 N×3 且 N >= 2；GPU 输出复用矩阵必须是 3×3。
常用于 PCA 法线估计和点云配准。

`covarianceMatrixAsync` 需要调用方持有 `GpuCovarianceWorkspace`，并保证它和输入/输出矩阵在 stream 完成前保持有效。
workspace 扩容时会保留旧缓冲到自身析构，以避免同一 workspace 在连续异步提交中释放仍被使用的临时内存。
同步 GPU 重载使用内部 workspace 并在返回前同步，适合直接读取结果。

## 典型工作流：点云配准

```cpp
#include <plamatrix/plamatrix.h>

// 1. 构建旋转矩阵 (绕 Z 轴 30°)
plamatrix::internal::PackedVector3<float> axis{0, 0, 1};
auto R = plamatrix::internal::rotationMatrix<float, plamatrix::internal::Device::CPU>(axis, M_PI / 6.0);

// 2. 构建刚体变换
plamatrix::internal::PackedVector3<float> translation{10.0f, 20.0f, 0.0f};
auto T = plamatrix::internal::rigidTransform<float, plamatrix::internal::Device::CPU>(R, translation);

// 3. 加载或生成点云 (N×3)
plamatrix::internal::DenseStorage<float, plamatrix::internal::Device::CPU> point_cloud(num_points, 3);
// ... 填充数据 ...

// 4. CPU 变换
auto transformed_cpu = plamatrix::internal::transformPoints<float, plamatrix::internal::Device::CPU>(T, point_cloud);

// 5. GPU 加速大批量变换
#ifdef PLAMATRIX_WITH_CUDA
auto pts_gpu = point_cloud.toGpu();
auto T_gpu = T.toGpu();
auto transformed_gpu = plamatrix::internal::transformPoints<float, plamatrix::internal::Device::GPU>(T_gpu, pts_gpu);
auto result = transformed_gpu.toCpu();

plamatrix::internal::DenseStorage<float, plamatrix::internal::Device::GPU> transformed_reuse(pts_gpu.rows(), 3);
plamatrix::internal::transformPointsAsync(T_gpu, pts_gpu, transformed_reuse, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
#endif

// 6. 计算协方差矩阵 (用于 PCA)
auto cov = plamatrix::internal::covarianceMatrix<float, plamatrix::internal::Device::CPU>(point_cloud);
auto eigenvalues = plamatrix::internal::eigh(cov);  // 特征值即为主成分方差

#ifdef PLAMATRIX_WITH_CUDA
plamatrix::internal::DenseStorage<float, plamatrix::internal::Device::GPU> cov_gpu(3, 3);
plamatrix::internal::GpuCovarianceWorkspace<float> workspace;
plamatrix::internal::covarianceMatrixAsync(pts_gpu, cov_gpu, workspace, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
#endif
```
