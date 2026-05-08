#!/usr/bin/env python3
# ============================================================================
# 参考数据生成器 — 用 NumPy/SciPy 计算 ground truth
#
# 用法: python3 generate_reference.py
# 输出: test/reference/*.txt — PlaMatrix 测试加载并与自身结果比对
# ============================================================================

import numpy as np
from scipy import linalg
import os
import struct

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))

rng = np.random.RandomState(SEED)


def write_matrix(path, mat):
    """将矩阵写入文件: rows cols 后跟列优先 float64 数据
    numpy.tobytes() 无视内存布局永远返回 C-order，必须用 ravel('F') 显式展开"""
    rows, cols = mat.shape
    data = mat.astype(np.float64).ravel(order='F')
    with open(path, "wb") as f:
        f.write(struct.pack("qq", rows, cols))
        f.write(data.tobytes())


def write_vector(path, vec):
    """将向量写入文件: n 后跟 float64 数据"""
    n = len(vec)
    with open(path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(np.asarray(vec, dtype=np.float64).tobytes())


def read_matrix(path):
    """读取矩阵文件，返回 numpy array (C-order)"""
    with open(path, "rb") as f:
        rows = struct.unpack("q", f.read(8))[0]
        cols = struct.unpack("q", f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.float64)
    return np.asfortranarray(data.reshape((rows, cols)))


# ====================================================================
# 1. GEMM — 矩阵乘法
# ====================================================================
print("1. GEMM...")
A = rng.rand(128, 128).astype(np.float64)
B = rng.rand(128, 128).astype(np.float64)
C = A @ B  # NumPy matmul

write_matrix(f"{OUT}/gemm_A.bin", A)
write_matrix(f"{OUT}/gemm_B.bin", B)
write_matrix(f"{OUT}/gemm_C_ref.bin", C)

# 小矩阵 (精确验证) — NumPy 与 PlaMatrix column-major 一致
# A2: 2x3, B2: 3x2 → C2: 2x2
A2 = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float64)  # shape (2,3)
B2 = np.array([[7, 10], [8, 11], [9, 12]], dtype=np.float64)  # shape (3,2)
C2 = A2 @ B2
write_matrix(f"{OUT}/gemm_small_A.bin", A2)
write_matrix(f"{OUT}/gemm_small_B.bin", B2)
write_matrix(f"{OUT}/gemm_small_C_ref.bin", C2)

# ====================================================================
# 2. SVD — 奇异值分解
# ====================================================================
print("2. SVD...")
A_svd = rng.rand(32, 24).astype(np.float64)
U, S, Vt = linalg.svd(A_svd, full_matrices=True)

write_matrix(f"{OUT}/svd_A.bin", A_svd)
write_matrix(f"{OUT}/svd_U_ref.bin", U)
write_vector(f"{OUT}/svd_S_ref.bin", S)
write_matrix(f"{OUT}/svd_Vt_ref.bin", Vt)

# 小矩阵 SVD — 3x3, NumPy shape (3,3), Fortran write 后 PlaMatrix 列优先正确读取
A_svd2 = np.array([[1,4,7],[2,5,8],[3,6,9]], dtype=np.float64)  # 3x3
U2, S2, Vt2 = linalg.svd(A_svd2, full_matrices=True)
write_matrix(f"{OUT}/svd_small_A.bin", A_svd2)
write_matrix(f"{OUT}/svd_small_U_ref.bin", U2)
write_vector(f"{OUT}/svd_small_S_ref.bin", S2)
write_matrix(f"{OUT}/svd_small_Vt_ref.bin", Vt2)

# ====================================================================
# 3. QR — QR 分解
# ====================================================================
print("3. QR...")
A_qr = rng.rand(48, 32).astype(np.float64)
Q, R = linalg.qr(A_qr)

write_matrix(f"{OUT}/qr_A.bin", A_qr)
write_matrix(f"{OUT}/qr_Q_ref.bin", Q)
write_matrix(f"{OUT}/qr_R_ref.bin", R)

# ====================================================================
# 4. Eigh — 对称矩阵特征值
# ====================================================================
print("4. Eigh...")
A_sym = rng.rand(64, 64).astype(np.float64)
A_sym = A_sym + A_sym.T  # 对称化
eigvals = linalg.eigvalsh(A_sym)[::-1]  # 降序

write_matrix(f"{OUT}/eigh_A.bin", A_sym)
write_vector(f"{OUT}/eigh_ref.bin", eigvals)

# ====================================================================
# 5. Solve — 线性求解
# ====================================================================
print("5. Solve...")
A_sol = rng.rand(100, 100).astype(np.float64)
A_sol += np.eye(100) * 100.0  # 对角占优
x_true = np.arange(100, dtype=np.float64).reshape(100, 1)
b_vec = A_sol @ x_true
x_solved = linalg.solve(A_sol, b_vec)

write_matrix(f"{OUT}/solve_A.bin", A_sol)
write_matrix(f"{OUT}/solve_b.bin", b_vec)
write_matrix(f"{OUT}/solve_x_ref.bin", x_solved)

# ====================================================================
# 6. Transform — 点云变换
# ====================================================================
print("6. Transform...")

def rotation_matrix(axis, angle):
    """Rodrigues 公式 (NumPy 实现)"""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

axis = np.array([0.0, 0.0, 1.0])
angle = 0.7853981633974483  # 45 degrees
R_ref = rotation_matrix(axis, angle)
write_matrix(f"{OUT}/rotation_R_ref.bin", R_ref)

# 刚体变换 (4x4)
T_ref = np.eye(4)
T_ref[:3, :3] = R_ref
T_ref[:3, 3] = [10.0, 5.0, 3.0]  # 平移
write_matrix(f"{OUT}/rigid_T_ref.bin", T_ref)

# 点变换
N_pts = 5000
points = rng.rand(N_pts, 3).astype(np.float64)
# 齐次坐标 p_h = [p, 1], 变换后 = T @ p_h, 取前三列
pts_h = np.ones((N_pts, 4))
pts_h[:, :3] = points
transformed = (T_ref @ pts_h.T).T[:, :3]

write_matrix(f"{OUT}/transform_points_in.bin", points)
write_matrix(f"{OUT}/transform_points_ref.bin", transformed)

# ====================================================================
# 7. Covariance — 协方差矩阵
# ====================================================================
print("7. Covariance...")
n_pts = 8000
pts_cov = rng.rand(n_pts, 3).astype(np.float64)
centroid = np.mean(pts_cov, axis=0)
centered = pts_cov - centroid
cov_ref = (centered.T @ centered) / n_pts

write_matrix(f"{OUT}/covariance_points.bin", pts_cov)
write_matrix(f"{OUT}/covariance_ref.bin", cov_ref)

# ====================================================================
# 总结
# ====================================================================
import glob
files = sorted(glob.glob(f"{OUT}/*.bin"))
print(f"\n生成 {len(files)} 个参考数据文件:")
for f in files:
    size = os.path.getsize(f)
    print(f"  {os.path.basename(f):40s} {size:>8d} bytes")
print("\n完成。运行 C++ 测试以验证 PlaMatrix 结果。")
