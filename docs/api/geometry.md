# Geometry API

PlaMatrix 的公开几何层采用 Eigen 5.0.1 的名称、模板顺序和组合写法。应用代码可把
`Eigen::` 替换为 `plamatrix::` 来使用本页覆盖的旋转、位姿和相似变换接口；这些类型由
`<plamatrix/plamatrix.h>` 统一导出。

## 旋转

公开类型包括 `RotationBase`、`QuaternionBase`、`Quaternion<Scalar, Options>`、
`Quaternionf`/`Quaterniond`、`AngleAxis<Scalar>` 和 `AngleAxisf`/`AngleAxisd`。

```cpp
plamatrix::AngleAxisd yaw(0.4, plamatrix::Vector3d::UnitZ());
plamatrix::Quaterniond orientation(yaw);
plamatrix::Matrix3d rotation = orientation.toRotationMatrix();
plamatrix::Vector3d rotated = orientation * plamatrix::Vector3d::UnitX();

plamatrix::Quaterniond target(
    plamatrix::AngleAxisd(-0.2, plamatrix::Vector3d::UnitY()));
plamatrix::Quaterniond halfway = orientation.slerp(0.5, target);
```

四元数构造函数的四个标量顺序与 Eigen 相同，为 `(w, x, y, z)`；`coeffs()` 的存储和返回
顺序为 `(x, y, z, w)`。需要显式表达顺序时可用 `FromCoeffsScalarFirst()`、
`FromCoeffsScalarLast()`、`coeffsScalarFirst()` 和 `coeffsScalarLast()`。此外提供
`Identity()`、`UnitRandom()`、`FromTwoVectors()`、`setFromTwoVectors()`、`inverse()`、
`conjugate()`、`normalize()`、`normalized()`、`dot()`、`angularDistance()`、`cast()`、
`isApprox()` 和四元数乘法。

`Quaternion` 可由 3×3 旋转矩阵或按 `(x, y, z, w)` 排列的四维列/行向量构造。
`AngleAxis` 可与四元数和旋转矩阵互转，其轴向量应像 Eigen 一样由调用方保持单位长度。

## 位姿和仿射变换

`Transform<Scalar, Dim, Mode, Options>` 及以下常用别名与 Eigen 对齐：

- `Isometry2f`、`Isometry2d`、`Isometry3f`、`Isometry3d`
- `Affine2f`、`Affine2d`、`Affine3f`、`Affine3d`
- `AffineCompact2f`、`AffineCompact2d`、`AffineCompact3f`、`AffineCompact3d`
- `Projective2f`、`Projective2d`、`Projective3f`、`Projective3d`
- `Translation2f`、`Translation2d`、`Translation3f`、`Translation3d`
- `UniformScaling<Scalar>` 和 `Scaling(scalar)`

```cpp
plamatrix::Translation3d position(1.0, 2.0, 3.0);
plamatrix::Quaterniond orientation(
    plamatrix::AngleAxisd(0.3, plamatrix::Vector3d::UnitZ()));
plamatrix::Isometry3d camera_pose = position * orientation;

plamatrix::Vector3d world = camera_pose * plamatrix::Vector3d(0.5, -0.2, 4.0);
plamatrix::Vector3d local = camera_pose.inverse() * world;
```

`matrix()`、`linear()`、`translation()` 和 `affine()` 暴露与 Eigen 同义的矩阵或块视图。
组合支持平移、旋转、均匀/逐轴缩放、二维 shear、逆变换以及点矩阵批量变换。
`translate()`/`rotate()`/`scale()` 右乘增量，`pretranslate()`/`prerotate()`/`prescale()`
左乘增量。`Isometry` 的 `rotation()` 返回线性块；其他模式通过 SVD 去除缩放，并提供
`computeRotationScaling()` 与 `computeScalingRotation()`。

## 欧拉角

与 Eigen 5.0.1 正式 Geometry 模块相同，欧拉角通过 3×3 矩阵成员函数读取：

```cpp
plamatrix::Vector3d ypr = rotation.canonicalEulerAngles(2, 1, 0);
plamatrix::Vector3d legacy = rotation.eulerAngles(2, 1, 0);
```

`canonicalEulerAngles()` 返回规范范围，`eulerAngles()` 保留 Eigen 的旧范围语义。轴编号为
0/1/2，分别表示 X/Y/Z；相邻轴不能相同。PlaMatrix 不另外增加 Eigen 正式 Geometry 模块之外的
SO(3)/SE(3) 对数映射、指数映射或雅可比接口。

## Umeyama 相似变换

`umeyama(source, destination, with_scaling)` 接受形状相同的点矩阵，每列为一个点，返回
`(Dim + 1) × (Dim + 1)` 齐次矩阵：

```cpp
plamatrix::Matrix<double, 3, plamatrix::Dynamic> source(3, count);
plamatrix::Matrix<double, 3, plamatrix::Dynamic> destination(3, count);
// 填充点坐标……
plamatrix::Matrix4d similarity = plamatrix::umeyama(source, destination, true);
plamatrix::Matrix4d rigid = plamatrix::umeyama(source, destination, false);
```

输入必须使用相同浮点标量、形状一致且包含有限值。估计尺度时，退化的源点集会明确报错。

这些固定尺寸几何操作在调用线程执行，避免为几个系数产生设备传输和调度开销。大批点的矩阵乘法
仍通过公开 `Matrix` 表达式进入自动 CPU/CUDA/Vulkan/OpenCL 后端选择。
