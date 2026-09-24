# 密集矩阵 API

面向应用代码的入口与 Eigen 5 一样是
`Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>`。前三个模板参数必填，后三个有
Eigen 同形的默认值；`Rows`/`Cols`/`MaxRows`/`MaxCols` 可取固定整数或 `Dynamic`。
`Matrix3d`、`MatrixXd`、`Vector3d`、`VectorXd`、固定/动态 `RowVector` 以及 float/int
对应类型是常用别名。矩阵拥有存储并支持深拷贝：

```cpp
plamatrix::Vector3d point(1.0, 2.0, 3.0);
plamatrix::Matrix3d rotation = plamatrix::Matrix3d::Identity();
plamatrix::MatrixXd measurements = plamatrix::MatrixXd::Zero(100, 3);
plamatrix::VectorXd weights = plamatrix::VectorXd::Ones(100);
auto rotated = rotation * point;
auto gram = measurements.transpose() * measurements;
```

也可像 Eigen 一样用逗号初始化标量。填写顺序按行从左到右，与底层列优先存储无关：

```cpp
#include <iostream>
#include <plamatrix/dense/matrix.h>

int main()
{
    plamatrix::Matrix3d matrix;
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0;
    std::cout << matrix << '\n';
}
```

动态矩阵须先确定形状。值过多或过少时抛出异常；`std::cout << matrix` 按逻辑行输出。
逗号初始化同时接受标量、矩阵和矩阵视图作为矩形块；同行块高度必须一致，且必须完整覆盖目标。
暂不支持对 `block()` 视图本身使用 `<<`。

`Zero`、`Ones`、`Identity`、`Constant` 同时有固定尺寸及运行时尺寸形式。`+`、`-`、
矩阵 `*`、标量 `*`/`/`、`+=`/`-=`/`*=`/`/=`、`cwiseProduct()`、`dot()`、
`norm()`、`squaredNorm()` 和浮点向量的 `normalized()`/`normalize()` 提供常见线性代数表达。
`sum()`、`mean()`、`prod()`、`minCoeff()`、`maxCoeff()`、`trace()` 可直接调用；
`minCoeff()`/`maxCoeff()` 同时提供向量的单索引指针重载和矩阵的行列索引指针重载。
`all()`、`any()`、`count()` 按系数的布尔值归约，空表达式分别返回 `true`、`false` 和 `0`，
空表达式的 `sum()`/`prod()` 分别返回 `0`/`1`。
`rowwise()`/`colwise()` 支持延迟的 `sum()`、`mean()` 与向量加减广播；
`array()` 支持延迟的逐元素加减乘除、加减标量、平方以及 `abs()`、`sqrt()`、`exp()`、`log()`。
这些数学函数目前在 CPU 求值，其中后三者要求浮点系数，整数 `abs()` 遇到不可表示的最小有符号值会抛错；数组表达式可直接串联
`values.array().sqrt().log()`。它们可接在矩阵或其他延迟表达式后，
再与普通矩阵表达式组合。浮点空矩阵的 `mean()` 与 Eigen 一样产生 NaN，
`minCoeff()`/`maxCoeff()` 要求表达式非空；
整数数组逐元素除以零也会抛错。
`RowVectorXd`/`RowVectorXf` 提供动态行向量别名。浮点矩阵可用 `Random()` 生成
`[-1, 1]` 内的值；`cast<T>()` 返回新标量类型矩阵。`isApprox()` 使用 Eigen 的 Frobenius 范数
相对比较语义；接近零应使用标量或矩阵重载的 `isMuchSmallerThan()`。
`conservativeResize()` 保留左上角重叠区域，新增系数初始化为零。
二元表达式通过公开的 `ScalarBinaryOpTraits<LhsScalar, RhsScalar, BinaryOp>` 决定结果标量；
内置算术类型默认使用 `std::common_type_t`，项目可为自定义标量或自动微分标量特化该 traits。
`NumTraits<T>` 提供 `Real`、`NonInteger`、`Nested`、精度和复数标志等 Eigen 同形信息。
`Matrix<float>` 与 `Matrix<double>` 的加减、逐元素运算、乘法和 `dot()` 会生成提升后的 double 结果。
复数矩阵支持复数标量乘除、`conjugate()`、`adjoint()`、左侧共轭的 `dot()`，并让
`norm()`/`squaredNorm()` 返回实数。当前 `adjoint()` 会物化结果矩阵；它的数值语义与 Eigen 一致，
返回类型还不是 Eigen 的延迟 `Transpose<CwiseUnaryOp<...>>`。
默认 `Matrix` 是列优先连续存储；`Options` 可指定 `RowMajor`，`MaxRows`/`MaxCols`
会限制动态维度的运行时上限。`coeff(row, col)`、`coeffRef(row, col)` 和向量的
`coeff(index)`、`coeffRef(index)` 提供 Eigen 风格的无边界检查访问；`operator[]` 仅用于向量并检查索引。
`innerSize()`/`outerSize()` 和 `innerStride()`/`outerStride()` 按实际存储顺序返回。
`AutoAlign`/`DontAlign` 保留 Eigen 的模板值和类型签名。`AutoAlign` 动态矩阵使用至少 64 字节
对齐的连续存储，便于 AVX2/AVX-512 和后续 packet evaluator 直接读取；`DontAlign` 使用普通分配。
与 Eigen 一样，动态矩阵的尺寸构造和 `resize()` 不初始化新增系数，需要确定初值时应使用
`Zero()`、`Ones()`、`Constant()` 或显式赋值。连续 `float`/`double` 的
`dot()`、`sum()`、`squaredNorm()` 及连续数组表达式归约会在运行时检测 AVX2/FMA，
使用专用 packet 内核并在不支持时回退到标量实现。packet evaluator 可把连续 `float`/`double`
矩阵、标量缩放、加减、常量平移、直接逐元素乘积和 `square()` 展开成一次归约遍历；大规模归约还会按
OpenMP 线程分块，每个线程保持局部 packet 累加器。其他基础逐元素核使用 OpenMP SIMD 求值。
混合标量、RowMajor、带 stride 的视图和任意非线性函数仍使用通用 evaluator。
RowMajor 的通用表达式在 CPU 求值；现有 CUDA
密集核继续使用列优先存储，避免错误解释调用方数据。
动态矩阵可 `resize()`；`block()` 精确返回 `Block<Derived>`，`block<Rows, Cols>()`
返回 `Block<Derived, Rows, Cols>`；`row()`、`col()` 的 `InnerPanel` 与 Eigen 5 保持一致。
这些公开 `Block` 类型和向量的 `segment()`、`head()`、`tail()` 都是非拥有视图；后三者精确返回
Eigen 同形的 `VectorBlock<Derived, Size>`，不再暴露内部存储视图。可变 CPU 视图支持矩阵或视图的
`=`、`+=`、`-=`；输入与输出重叠时先保存输入值，尺寸不一致时抛出异常。
`block<Rows, Cols>()` 的返回类型保留编译期块形状，可直接与矩阵或视图相加、相减、相乘；
`transpose()` 精确返回 `Transpose<Derived>` 或 `Transpose<const Derived>`，也可参与这些运算；
块和延迟表达式的转置对象保存其轻量表达式描述，普通矩阵的转置对象引用原矩阵，与 Eigen 一样要求原矩阵
活到表达式求值结束。运行时尺寸的 `block()` 返回动态形状 `Block`。
这些视图操作会访问主机存储；对设备驻留的 `Matrix` 调用可变视图会使 GPU 缓存失效。
跨固定/动态尺寸的复制会检查实际形状。普通 `+`、`-`、`*`、标量乘除、
`cwiseProduct()` 和 `transpose()` 现在组成延迟表达式，直到赋给 `Matrix` 或调用 `eval()`
才求值。CPU 上连续逐元素运算一次遍历输出；普通乘积直接写入新矩阵，
转置参与 CPU 乘法时按 stride 访问而不先复制矩阵。`auto expression = a + b` 保存
对左值矩阵的引用，因此这些矩阵必须活到求值结束；右值临时对象由表达式拥有。
连续实数表达式的 `sum()`、`dot()` 和 `squaredNorm()` 直接归约线性 evaluator，并使用
OpenMP SIMD，不会先物化表达式。固定尺寸与动态尺寸向量之间的 `dot()` 也直接读取两侧存储，
不再为了统一模板类型复制操作数。显式插入 `eval()` 仍然表示用户要求的物化边界。
当目标可能与输入重叠时，普通赋值先计算到新矩阵。已知无重叠时可用
`output.noalias() = left * right`；CPU 路径直接写入现有输出，CUDA 路径沿用自动后端选择并将结果留在设备端。
目前 CUDA 路径仍需分配结果缓冲区，尚未复用目标的已有设备存储；调用方须保证输入与输出不重叠。
表达式还可直接使用 `dot()`、`squaredNorm()`、`norm()`、`lpNorm<p>()`、`stableNorm()`、
`blueNorm()`、`hypotNorm()`、`trace()`、`allFinite()`、`isApprox()` 和
`isMuchSmallerThan()`，无需先 `eval()`。`lpNorm<Infinity>()` 返回绝对值最大的系数；三种稳定
L2 范数避免普通平方和在极大或极小系数上溢出或下溢。这些标量查询目前在 CPU 读取系数。
`cwiseAbs()`、矩阵或标量重载的 `cwiseMin()`/`cwiseMax()`、`unaryExpr()` 和 `binaryExpr()`
返回 Eigen 同名的 `CwiseUnaryOp`/`CwiseBinaryOp` 惰性公开表达式，可继续组合和归约。
GPU 路径保留自动选后端，包括 `noalias()` 乘法，但目前仍逐节点调用现有 CUDA 内核，不做跨节点 kernel 融合。
新加入的数组标量操作、逐元素除法、数学函数及轴向归约/广播目前在 CPU 求值；
内部强制 GPU 的验证策略会明确报错。

矩阵视图与结构化表达式还提供以下 Eigen 同名入口：

- `diagonal()`、`diagonal<Index>()` 和 `diagonal(index)` 返回可写 `Diagonal` 视图；
- 向量的 `asDiagonal()` 返回惰性的 `DiagonalWrapper`；
- `triangularView<Lower/Upper/UnitLower/UnitUpper/StrictlyLower/StrictlyUpper>()` 只读取或写入指定三角；
- `selfadjointView<Lower/Upper>()` 只存储一个三角，并在读取和矩阵运算时镜像另一侧；
- `reshaped()`、`reshaped(rows, cols)`、`reshaped<RowMajor>(rows, cols)` 改变逻辑形状和遍历顺序；
- `replicate<RowFactor, ColFactor>()`、`replicate(rowFactor, colFactor)` 和 `reverse()` 返回惰性表达式；
- `transposeInPlace()` 原地转置。动态拥有型矩阵可以交换行列尺寸，固定或非拥有视图必须为方阵。

这些表达式可以直接构造 `Matrix`，也可以参与 `+`、`-` 和矩阵乘法。对角、重排和反转视图保持零拷贝；
`replicate`、三角和自伴视图在逻辑矩阵之外的系数按结构规则求值。当前结构化表达式在 CPU 求值，调用方需要让
被引用的矩阵活到表达式求值结束。`transposeInPlace()` 和 `reverseInPlace()` 会先保存输入，因而对自身和重叠视图
保持别名安全。

```cpp
plamatrix::Matrix3d covariance = plamatrix::Matrix3d::Identity();
covariance.diagonal() += plamatrix::Vector3d(0.1, 0.2, 0.3);
plamatrix::Matrix3d symmetric = covariance.selfadjointView<plamatrix::Lower>();
plamatrix::Vector3d weighted = plamatrix::Vector3d(1.0, 2.0, 3.0).asDiagonal() *
                               plamatrix::Vector3d::Ones();
plamatrix::MatrixXd image(2, 3);
auto column_major_pixels = image.reshaped();
auto tiled = image.replicate<2, 2>();
plamatrix::Index minimum_row = 0;
plamatrix::Index minimum_col = 0;
double minimum = image.minCoeff(&minimum_row, &minimum_col);
auto clamped = image.cwiseMax(0.0).cwiseMin(1.0);
double robust_length = image.stableNorm();
```

使用与 Eigen 同形的 `Map<MatrixType, MapOptions, StrideType>` 将调用方拥有的 CPU 缓冲区
映射成矩阵，避免导入时复制。固定尺寸、动态尺寸、`const` 映射、`InnerStride`、
`OuterStride` 和显式 `Stride<Outer, Inner>` 都受支持，包含负 stride。inner 表示存储内层
相邻系数的步长；outer 对 RowMajor 表示行步长，对 ColMajor 表示列步长。映射也可参与
延迟表达式及 `array()`/`rowwise()`/`colwise()`。
调用方必须保证缓冲区在所有映射和使用它的延迟表达式求值之前保持有效。当前 `Map` 只映射 CPU 内存。

```cpp
double coordinates[6] = {1, 2, 3, 4, 5, 6}; // 列优先 2×3
plamatrix::Map<plamatrix::Matrix<double, 2, 3>> mapped(coordinates);
plamatrix::Matrix<double, 2, 3> normalized = mapped.array().abs();
plamatrix::Map<const plamatrix::MatrixXd> read_only(coordinates, 2, 3);
auto sums = read_only.colwise().sum();
```

非拥有函数参数可使用 Eigen 同形的 `Ref<PlainObjectType, Options, StrideType>`：

```cpp
void normalizeColumns(plamatrix::Ref<plamatrix::MatrixXd> values);
double norm(const plamatrix::Ref<const plamatrix::VectorXd>& values);
```

动态 `Ref` 可绑定同标量、同存储顺序的固定尺寸矩阵。可变 `Ref` 可直接绑定可写矩阵或临时返回的
非拥有块描述（例如将 `matrix.block(...)` 直接传给接收 `Ref<MatrixXd>` 的函数）；
`Ref<const ...>` 不复制并要求源对象在引用期间保持有效。`Ref` 可绑定公开 `Block`，并保留父矩阵的
运行时 outer/inner stride；因此列优先矩形块、行优先矩形块和显式动态 inner-stride 行视图均不复制。
默认 `Ref<MatrixXd>` 仍要求 inner stride 为 1；任意二维 stride 应显式使用
`Stride<Dynamic, Dynamic>`，单个跨步向量可使用 `InnerStride<Dynamic>`。这与 Eigen 的约束一致：
`Map` 可以直接描述调用方指定的任意 stride；`Ref` 只接受其模板参数声明允许的布局，可写 `Ref` 不会为不兼容
布局创建隐藏临时矩阵。

```cpp
plamatrix::MatrixXd matrix = plamatrix::MatrixXd::Zero(6, 6);
matrix.block<3, 3>(0, 0) = plamatrix::Matrix3d::Identity();
plamatrix::VectorXd values = plamatrix::VectorXd::Ones(6);
values.segment<3>(3) += plamatrix::Vector3d(1.0, 2.0, 3.0);
plamatrix::Matrix3d gram;
gram.noalias() = matrix.block<3, 3>(0, 0).transpose() * matrix.block<3, 3>(0, 0);
plamatrix::MatrixXd centered = (matrix.rowwise() - plamatrix::RowVectorXd::Ones(6)).array().square();
plamatrix::VectorXd row_sums = (matrix + centered).rowwise().sum();
```

平方矩阵可通过 `partialPivLu()` 创建可复用的 CPU 部分选主元 LU 分解，再对一个或多个
右端调用 `solve()`。该入口仅支持浮点标量，拒绝非有限、非方阵和奇异输入；
该入口在 CPU 求解。BA 业务调度与残差装配由 PlaBundle 负责。

`llt()`、`ldlt()`、`fullPivLu()`、`householderQr()`、`colPivHouseholderQr()`、
`jacobiSvd()` 和 `bdcSvd()` 使用与 Eigen 相同的入口名。列选主元 QR 接受宽矩阵与秩亏矩阵，
并提供秩、核维数和列置换；SVD 支持 thin/full U、V 选项及阈值控制。CUDA 可自动执行 LLT、
无选主元 QR、SVD 和自伴特征分解；OpenCL 支持相同三项通用分解的 `float`/`double` 路径，
Vulkan 支持对应的 `float` 路径。其余分解保持 CPU 语义，强制到不支持的 GPU 后端会报告错误。

`SelfAdjointEigenSolver<MatrixType>` 支持任意方阵尺寸，只读取输入下三角，按特征值升序返回
`eigenvalues()` 和列向量形式的 `eigenvectors()`；`compute()` 可复用同一对象，CUDA 可用时使用
cuSOLVER，OpenCL/Vulkan 可用时使用并行 Jacobi。一般实矩阵可使用 `EigenSolver`，广义问题可使用
`GeneralizedSelfAdjointEigenSolver` 或 `GeneralizedEigenSolver`。

`norm()` / `squaredNorm()` 对向量返回欧氏范数及其平方，对一般矩阵返回 Frobenius 范数及其平方；
拥有型矩阵、延迟表达式和带步长 `Map` 保持一致。`dot()` 仍只接受等长向量。
逗号初始化支持可转换为矩阵标量类型的数值，例如 `MatrixXd` 中可直接写整数文字量。

```cpp
plamatrix::Matrix3d covariance = plamatrix::Matrix3d::Identity();
plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3d> eigensolver(covariance);
plamatrix::Vector3d normal = eigensolver.eigenvectors().col(0);
```

```cpp
plamatrix::Matrix3d coefficients = plamatrix::Matrix3d::Identity();
plamatrix::Vector3d right(1.0, 2.0, 3.0);
auto factor = coefficients.partialPivLu();
plamatrix::Vector3d solution = factor.solve(right);
```

默认自动策略对小型或逐元素的主机数据使用 CPU，对计算量足够大的矩阵乘法及已驻留设备的输入使用 GPU。
达到各后端实测工作量门槛后的候选顺序是 CUDA、OpenCL、Vulkan；不支持当前标量类型、运行时不可用或
尚未达到对应门槛的候选会被跳过。当前通用 GEMM
在实测设备上由厂商 OpenCL 编译器生成的代码更稳定，因此 OpenCL 排在 Vulkan 前；显式指定 Vulkan 时
仍严格执行 Vulkan。阈值是启发式策略，没有兼容 GPU 时自动使用 CPU。应用代码无需指定设备或手动搬移数据。

读取 `const data()` 会在必要时下载结果；非 const `data()` 和可变系数访问还会使设备缓存失效。
CUDA 与 OpenCL 支持 `float`/`double` 基础算术；OpenCL 的 `double` 需要设备 FP64。Vulkan 当前支持 `float`
的乘法、加减、缩放和逐元素乘法。OpenCL/Vulkan GEMM 在设备支持 256 线程工作组时使用 16×16 分块
并处理非整块边界；Vulkan 在限制更严格的设备上使用 64 线程标量 fallback，并复用已创建的稠密算术
pipeline。列优先实数矩阵的 `sum()`、`dot()` 和 `squaredNorm()` 可在 CUDA、OpenCL 与 Vulkan
执行，并复用输入的设备驻留缓存。无选主元 `HouseholderQR`、`JacobiSVD`/`BDCSVD` 和
`SelfAdjointEigenSolver` 也参与相同自动策略；OpenCL 支持 `float`/`double`，Vulkan 支持 `float`。
RowMajor、轴向归约、一般转置表达式及其他分解仍在 CPU 求值。

`view()`、`transposeView()`、`add()/subtract()/scaled()`、`executionInfo()` 和驻留状态查询不再是公开接口。使用 `block()`、`transpose()`、运算符及 `eval()`；内部诊断与显式 GPU 原语见[后端实现接口](backend-internals.md)。
