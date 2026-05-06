# 点云 API

PlaMatrix 提供针对三维点云处理的核心运算：旋转矩阵生成、刚体变换构造、点云坐标变换和协方差矩阵计算。所有函数支持 CPU 和 GPU 两种后端。

## 头文件

```cpp
#include <plamatrix/plamatrix.h>
```

## 辅助类型

### Vec3 — 三维向量

```cpp
template <typename Scalar>
struct Vec3 {
    Scalar x, y, z;
};

// 使用示例
Vec3<float> axis = {0.0f, 0.0f, 1.0f};       // Z 轴
Vec3<double> translation = {1.0, 2.0, 3.0};   // 平移向量
```

## 旋转矩阵

### rotationMatrix — Rodrigues 旋转矩阵

基于 Rodrigues 旋转公式，从旋转轴和角度生成 3x3 旋转矩阵。

```cpp
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rotationMatrix(
    const Vec3<Scalar>& axis,
    Scalar angle
);
```

| 参数 | 说明 |
|------|------|
| `axis` | 旋转轴，内部自动归一化 |
| `angle` | 旋转角度（弧度制） |
| 返回值 | 3x3 旋转矩阵 |
| 异常 | `axis` 范数过小时抛出 `std::runtime_error` |

**示例**：

```cpp
// 绕 Z 轴旋转 90 度 (π/2)
Vec3<float> z_axis = {0.0f, 0.0f, 1.0f};
float angle = 3.1415926f / 2.0f;  // π/2

auto R = rotationMatrix<float, Device::CPU>(z_axis, angle);
// R 为 3x3 矩阵:
//   [ 0 -1  0]
//   [ 1  0  0]
//   [ 0  0  1]

// GPU 版本
auto R_gpu = rotationMatrix<float, Device::GPU>(z_axis, angle);
```

## 刚体变换

### rigidTransform — 4x4 刚体变换矩阵

将 3x3 旋转矩阵和 3x1 平移向量组合为 4x4 齐次变换矩阵。

```cpp
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> rigidTransform(
    const DenseMatrix<Scalar, Dev>& R,   // 3x3 旋转矩阵
    const Vec3<Scalar>& t                 // 平移向量
);
```

返回格式：

```
T = [R(3x3)  t(3x1)]
    [0 0 0    1    ]
```

**示例**：

```cpp
// 绕 Y 轴旋转 30 度，并平移到 (10, 0, 5)
Vec3<float> y_axis = {0.0f, 1.0f, 0.0f};
auto R = rotationMatrix<float, Device::CPU>(y_axis, 0.5236f);  // π/6 ≈ 0.5236

Vec3<float> t = {10.0f, 0.0f, 5.0f};

auto T = rigidTransform(R, t);
// T: 4x4 变换矩阵
```

### transformPoints — 点云坐标变换

将 N 个点通过 4x4 刚体变换矩阵进行坐标变换。

```cpp
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> transformPoints(
    const DenseMatrix<Scalar, Dev>& T,        // 4x4 变换矩阵
    const DenseMatrix<Scalar, Dev>& points    // Nx3 点云
);
```

| 参数 | 说明 |
|------|------|
| `T` | 4x4 刚体变换矩阵 |
| `points` | N x 3 点云矩阵，每行一个点的 (x, y, z) 坐标 |
| 返回值 | N x 3 变换后的点云 |
| 异常 | T 不是 4x4 或 points 不是 Nx3 时抛出 `std::runtime_error` |

变换公式：对于每个点 `p`，计算 `p' = R * p + t`（通过 4x4 齐次矩阵乘法实现）。

**示例**：

```cpp
// 创建点云：一个三角形的三个顶点
DenseMatrix<float, Device::CPU> points(3, 3);
// 点1: (1, 0, 0)
points(0, 0) = 1.0f; points(1, 0) = 0.0f; points(2, 0) = 0.0f;
// 点2: (0, 1, 0)
points(0, 1) = 0.0f; points(1, 1) = 1.0f; points(2, 1) = 0.0f;
// 点3: (0, 0, 1)
points(0, 2) = 0.0f; points(1, 2) = 0.0f; points(2, 2) = 1.0f;

// 绕 Z 轴旋转 90 度
Vec3<float> z_axis = {0.0f, 0.0f, 1.0f};
auto R = rotationMatrix<float, Device::CPU>(z_axis, 3.1415926f / 2.0f);
Vec3<float> t = {0.0f, 0.0f, 0.0f};
auto T = rigidTransform(R, t);

auto transformed = transformPoints(T, points);
// transformed 为 3x3 矩阵，包含旋转后的三个点
// 点1: (0, 1, 0)
// 点2: (-1, 0, 0)
// 点3: (0, 0, 1)
```

### GPU 加速变换

```cpp
// 数据在 GPU 上
auto points_gpu = points.toGpu();
auto T_gpu = rigidTransform(R.toGpu(), t);
auto result_gpu = transformPoints(T_gpu, points_gpu);
auto result = result_gpu.toCpu();
```

## 协方差矩阵

### covarianceMatrix — 点云协方差矩阵

计算点云的三维协方差矩阵（3x3），常用于 PCA、法向量估计等任务。

```cpp
template <typename Scalar, Device Dev>
DenseMatrix<Scalar, Dev> covarianceMatrix(
    const DenseMatrix<Scalar, Dev>& points  // Nx3 点云
);
```

| 参数 | 说明 |
|------|------|
| `points` | N x 3 点云矩阵 |
| 返回值 | 3x3 协方差矩阵（半正定） |

计算流程：
1. 计算点云质心（逐维度求均值）
2. 点云去中心化（每个点减去质心）
3. `C = (1/N) * centered^T * centered`（3x3 矩阵）

**示例**：

```cpp
// 随机散布的点云
DenseMatrix<float, Device::CPU> pts(1000, 3);
// ... 填充 1000 个点 ...

auto cov = covarianceMatrix(pts);
// cov 为 3x3 矩阵:
//   [var_x   cov_xy  cov_xz]
//   [cov_xy  var_y   cov_yz]
//   [cov_xz  cov_yz  var_z ]

// 对协方差矩阵做特征值分解，获取主方向
auto eigenvalues = eigh(cov);
float lambda_max = eigenvalues(0, 0);  // 最大特征值对应第一主方向
```

## 典型工作流：点云配准中的刚体变换

以下示例展示点云处理中的典型工作流——对输入点云施加刚体变换：

```cpp
#include <plamatrix/plamatrix.h>
#include <iostream>
#include <cmath>

using namespace plamatrix;

int main()
{
    // 1. 构造旋转（绕 Z 轴转 45 度）
    Vec3<float> axis = {0.0f, 0.0f, 1.0f};
    float angle = 3.1415926f / 4.0f;  // 45 度
    auto R = rotationMatrix<float, Device::CPU>(axis, angle);

    // 2. 构造刚体变换（旋转 + 平移）
    Vec3<float> translation = {5.0f, -3.0f, 2.0f};
    auto T = rigidTransform(R, translation);

    // 3. 读入点云（假设从文件读取）
    Index N = 100000;
    DenseMatrix<float, Device::CPU> cloud(N, 3);
    // ... 填充点云数据 ...

    // 4. 变换点云（CPU 模式）
    auto cloud_transformed = transformPoints(T, cloud);

    // 5. 或者用 GPU 加速大数据量变换
    auto cloud_gpu = cloud.toGpu();
    auto T_gpu = T.toGpu();
    auto result_gpu = transformPoints(T_gpu, cloud_gpu);
    auto result = result_gpu.toCpu();

    // 6. 计算变换后点云的协方差矩阵
    auto cov = covarianceMatrix(result);
    std::cout << "协方差矩阵 (3x3):" << std::endl;
    for (Index i = 0; i < 3; ++i)
    {
        for (Index j = 0; j < 3; ++j)
            std::cout << cov(i, j) << " ";
        std::cout << std::endl;
    }

    return 0;
}
```

## 性能建议

1. **大数据量用 GPU**：点数超过 10^5 时，GPU 加速的 `transformPoints` 和 `covarianceMatrix` 通常比 CPU 快 10-50 倍
2. **批量变换**：如需对同一点云施加多次不同变换，考虑将变换矩阵合并后一次调用而不是多次调用
3. **精度选择**：点云坐标通常在常规浮点精度范围内，`float` 足够满足大多数应用场景
4. **协方差矩阵用途**：计算出的 3x3 协方差矩阵可直接传入 `eigh()` 做特征值分解，得到点云分布的主方向（PCA）
