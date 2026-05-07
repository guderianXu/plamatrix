# 点云运算 API

## 数据结构

```cpp
template <typename Scalar>
struct Vec3 { Scalar x, y, z; };
```

## 旋转矩阵

```cpp
/// Rodrigues 公式: R = I + sin(θ)·K + (1-cos(θ))·K²
/// K 是归一化旋转轴的叉乘矩阵
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rotationMatrix(
    const Vec3<Scalar>& axis,  // 旋转轴 (自动归一化)
    Scalar angle               // 旋转角度 (弧度)
);
```

返回 3×3 旋转矩阵。

```cpp
// 绕 Z 轴旋转 90°
Vec3<float> z_axis{0, 0, 1};
auto R = rotationMatrix<float, Device::CPU>(z_axis, M_PI / 2.0);
```

## 刚体变换

```cpp
/// 构建 4×4 齐次变换矩阵: [R|t; 0 0 0 1]
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(
    const DenseMatrix<Scalar, Dev>& R,  // 3×3 旋转矩阵
    const Vec3<Scalar>& t                // 平移向量
);
```

## 批量点变换

```cpp
/// 对 N×3 点云施加 4×4 刚体变换
/// 对每个点 p 计算: p' = R·p + t
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> transformPoints(
    const DenseMatrix<Scalar, Dev>& T,      // 4×4 刚体变换
    const DenseMatrix<Scalar, Dev>& points  // N×3 点云
);
```

返回 N×3 变换后的点云。

- **CPU**：逐点循环计算
- **GPU**：`transformPointsKernel` 并行处理

## 协方差矩阵

```cpp
/// 计算 N×3 点云的 3×3 协方差矩阵
/// 1. 计算质心
/// 2. 中心化
/// 3. C = (1/N) · centered^T · centered
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(
    const DenseMatrix<Scalar, Dev>& points  // N×3 点云
);
```

返回 3×3 半正定矩阵。常用于 PCA 法线估计和点云配准。

## 典型工作流：点云配准

```cpp
#include <plamatrix/plamatrix.h>
using namespace plamatrix;

// 1. 构建旋转矩阵 (绕 Z 轴 30°)
Vec3<float> axis{0, 0, 1};
auto R = rotationMatrix<float, Device::CPU>(axis, M_PI / 6.0);

// 2. 构建刚体变换
Vec3<float> translation{10.0f, 20.0f, 0.0f};
auto T = rigidTransform<float, Device::CPU>(R, translation);

// 3. 加载或生成点云 (N×3)
DenseMatrix<float, Device::CPU> point_cloud(num_points, 3);
// ... 填充数据 ...

// 4. CPU 变换
auto transformed_cpu = transformPoints<float, Device::CPU>(T, point_cloud);

// 5. GPU 加速大批量变换
auto pts_gpu = point_cloud.toGpu();
auto T_gpu = T.toGpu();
auto transformed_gpu = transformPoints<float, Device::GPU>(T_gpu, pts_gpu);
auto result = transformed_gpu.toCpu();

// 6. 计算协方差矩阵 (用于 PCA)
auto cov = covarianceMatrix<float, Device::CPU>(point_cloud);
auto eigenvalues = eigh(cov);  // 特征值即为主成分方差
```
