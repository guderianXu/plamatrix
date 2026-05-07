#pragma once

// ============================================================================
// PlaMatrix — 面向点云处理的高性能矩阵运算库
// ============================================================================
//
// 本库提供三大功能模块:
//   1. 矩阵容器 — DenseMatrix (密集), COOMatrix/CSRMatrix (稀疏)
//   2. 线性代数 — gemm, SVD, QR, 特征值, 线性求解
//   3. 点云运算 — 旋转矩阵, 刚体变换, 批量变换, 协方差
//
// 核心设计:
//   - DeviceMatrix<Scalar, Device> 是基类，编译期绑定设备 (CPU/GPU)
//   - 矩阵列优先存储 (column-major)，兼容 cuBLAS Fortran 序
//   - 显式设备管理: toCpu() / toGpu() 触发数据传输
//   - GPU 侧通过 cuBLAS / cuSOLVER / 自定义 kernel 加速
//   - CPU 侧通过 OpenMP 多线程并行
//
// 使用示例:
//   #include <plamatrix/plamatrix.h>
//   using namespace plamatrix;
//
//   DenseMatrix<float, Device::CPU> A(1000, 1000);
//   auto A_gpu = A.toGpu();
//   auto C_gpu = gemm(A_gpu, A_gpu);
//   auto C = C_gpu.toCpu();
//
// 项目主页: https://github.com/guderianXu/plamatrix
// ============================================================================

#include "plamatrix/core/types.h"
#include "plamatrix/core/error_check.h"
#include "plamatrix/core/allocator.h"
#include "plamatrix/core/device_matrix.h"
#include "plamatrix/dense/dense_matrix.h"
#include "plamatrix/dense/dense_ops.h"
#include "plamatrix/ops/gemm.h"
#include "plamatrix/ops/decomposition.h"
#include "plamatrix/ops/solver.h"
#include "plamatrix/ops/point_cloud.h"
#include "plamatrix/sparse/coo_matrix.h"
#include "plamatrix/sparse/csr_matrix.h"
