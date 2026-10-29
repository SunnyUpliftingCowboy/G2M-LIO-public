#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Modified for the public preview on 2026-10-29: portable paths and output handling.
"""Synthetic plane-fitting and uncertainty comparison.

The experiment builds a 0.5 m voxel map from one noisy frame, keeps an
independent fixed query point, and compares FAST-LIO, VoxelMap, AKF-LIO and
the current G2M-LIO policy on a corner and a cylinder.  Monte Carlo trials
compare first-order normal/plane covariance with empirical errors.

This is an executable research prototype.  FAST-LIO's five-point QR fit and
VoxelMap's anchor octree branch follow their source rules.  AKF-LIO and Ours
reproduce the order-dependent iVox insertion, cross-voxel Mahalanobis merge,
primitive migration, 26-neighbour query, and source-specific plane rules.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    # ---- 字体 ----
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman'],
    'mathtext.fontset': 'stix',

    # ---- 字号（IEEE 推荐 8-10pt）----
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # ---- 图例 ----
    'legend.framealpha': 0.8,

    # ---- 坐标轴 ----
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    # ---- 导出 ----
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

OUTPUT_DIR = Path("results/plane_fitting")


def output_path(filename):
    return OUTPUT_DIR / filename


# ==================== 配置参数 ====================
LIDAR_COV = 0.001       # compared source code mapping/lidar_cov (m^2)
T_MAL = 7.82            # 马氏距离阈值
VOXEL_SIZE = 0.5         # 体素大小
VOXELMAP_ROOT_SIZE = 2.0 # source-style octree root: 2.0 -> 1.0 -> 0.5 m
VOXELMAP_MAX_LAYER = 2   # root is layer 0; deepest leaves are layer 2
NUM_RANDOM_PTS = 5       # Method1 随机选点数
NOISE_STD = 0.005        # synthetic isotropic point standard deviation (m)
MODEL_POINT_VAR = 0.001  # lio/lidar_cov used by the compared source code (m^2)
VOXEL_ORIGIN = np.zeros(3)  # C++ Pos2Grid uses floor(point / resolution)
LIDAR_RANGE_STD = 0.02
LIDAR_ANGLE_STD_DEG = 0.05
OURS_CROSS_VOXEL_MERGE = False  # current avia.yaml
OURS_MIN_SUPPORT_VOXELS = 5     # fitting experiment requested setting
OURS_MAX_SUPPORT_VOXELS = 7
NEIGHBOR_OFFSETS = tuple((i, j, k) for i in (-1, 0, 1)
                         for j in (-1, 0, 1) for k in (-1, 0, 1))


# ==================== 合成点云生成 ====================

def generate_tree_cloud(n_points=3000, radius=0.15, height=2.0, noise_std=NOISE_STD):
    """树干：圆柱体表面 + 噪声，局部似平面但有曲率"""
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(0, height, n_points)
    x = radius * np.cos(theta) + np.random.normal(0, noise_std, n_points)
    y = radius * np.sin(theta) + np.random.normal(0, noise_std, n_points)
    # ground truth normal: radial direction (outward from axis)
    gt_normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_points)])
    return np.column_stack([x, y, z]), gt_normals


def generate_corner_cloud(n_points=3000, size=1.0, noise_std=NOISE_STD):
    """墙角：两个垂直平面 L 形交汇，只在正方向延伸"""
    n_half = n_points // 2
    # 平面1: x≈0, y ∈ [0, size], z ∈ [0, 2*size]（只在 y>0 侧）
    pts1 = np.column_stack([
        np.random.normal(0, noise_std, n_half),
        np.random.uniform(0, size, n_half),
        np.random.uniform(0, 2 * size, n_half)
    ])
    gt_n1 = np.tile([1, 0, 0], (n_half, 1))

    # 平面2: y≈0, x ∈ [0, size], z ∈ [0, 2*size]（只在 x>0 侧）
    pts2 = np.column_stack([
        np.random.uniform(0, size, n_half),
        np.random.normal(0, noise_std, n_half),
        np.random.uniform(0, 2 * size, n_half)
    ])
    gt_n2 = np.tile([0, 1, 0], (n_half, 1))

    return np.vstack([pts1, pts2]), np.vstack([gt_n1, gt_n2])


# ==================== 体素划分 ====================

def voxelize(points, voxel_size, origin=VOXEL_ORIGIN):
    """将点云划分到体素，返回 {voxel_key: [indices]} """
    voxels = {}
    for i, p in enumerate(points):
        key = tuple(np.floor((p - origin) / voxel_size).astype(int))
        if key not in voxels:
            voxels[key] = []
        voxels[key].append(i)
    return voxels


def voxel_center(key, voxel_size, origin=VOXEL_ORIGIN):
    return origin + (np.array(key) + 0.5) * voxel_size


# ==================== 核心函数（复用 C++ 逻辑）====================

def make_primitive(pos, cov_pos=None, cov_geo=None, pt_num=1, voxel_ids=None):
    if cov_pos is None:
        cov_pos = LIDAR_COV * np.eye(3)
    if cov_geo is None:
        cov_geo = np.zeros((3, 3))
    if voxel_ids is None:
        voxel_ids = set()
    return {
        'pos': pos.copy(),
        'cov_pos': cov_pos.copy(),
        'cov_geo': cov_geo.copy(),
        'pt_num': pt_num,
        'voxel_ids': set(voxel_ids),
    }


def merge2(pt1, pt2):
    """复用 ivox3d_node.hpp Merge2"""
    n1, n2 = pt1['pt_num'], pt2['pt_num']
    n_total = n1 + n2
    ratio_1 = n1 / n_total
    ratio_2 = n2 / n_total

    p3 = ratio_1 * pt1['pos'] + ratio_2 * pt2['pos']

    cov_pos = ratio_1**2 * pt1['cov_pos'] + ratio_2**2 * pt2['cov_pos']

    offset1 = np.outer(pt1['pos'] - p3, pt1['pos'] - p3)
    offset2 = np.outer(pt2['pos'] - p3, pt2['pos'] - p3)

    if n_total - 1 > 0:
        cov_geo = ((n1 - 1) * pt1['cov_geo'] + (n2 - 1) * pt2['cov_geo']
                   + n1 * offset1 + n2 * offset2) / (n_total - 1)
    else:
        cov_geo = np.zeros((3, 3))

    return {
        'pos': p3,
        'cov_pos': cov_pos,
        'cov_geo': cov_geo,
        'pt_num': n_total,
        'voxel_ids': pt1['voxel_ids'] | pt2['voxel_ids'],
    }


def mal_distance(pt1, pt2):
    """复用 ivox3d_node.hpp MalDistance"""
    d = pt1['pos'] - pt2['pos']
    if pt1['pt_num'] > 1 and pt2['pt_num'] > 1:
        cov = pt1['cov_geo'] + pt2['cov_geo']
    elif pt1['pt_num'] > 1:
        cov = pt1['cov_geo'] + pt2['cov_pos']
    elif pt2['pt_num'] > 1:
        cov = pt1['cov_pos'] + pt2['cov_geo']
    else:
        cov = pt1['cov_pos'] + pt2['cov_pos']

    try:
        return float(d @ np.linalg.inv(cov) @ d)
    except np.linalg.LinAlgError:
        return float('inf')


# ==================== Method 1: 最近中心点 ====================

def method1_representative(points, voxels):
    """每个体素选离中心最近的点"""
    representatives = []
    rep_indices = []
    for key, indices in voxels.items():
        if len(indices) < 3:
            continue
        center = voxel_center(key, VOXEL_SIZE)
        pts = points[indices]
        dists = np.linalg.norm(pts - center, axis=1)
        best = indices[np.argmin(dists)]
        representatives.append(points[best])
        rep_indices.append(best)
    return np.array(representatives), rep_indices


def fit_plane_svd(pts):
    """SVD 拟合平面，返回 (centroid, normal)"""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


# ==================== Method 2: 马氏距离聚类 ====================

def method2_cluster_in_voxel(points_in_voxel, voxel_id, max_pts=30):
    """用马氏距离聚类体素内点云为高斯基元（对多点体素随机降采样加速）"""
    pts = points_in_voxel
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]

    primitives = [make_primitive(p, voxel_ids={voxel_id}) for p in pts]

    changed = True
    while changed and len(primitives) > 1:
        changed = False
        best_dist = float('inf')
        best_pair = (-1, -1)

        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                d = mal_distance(primitives[i], primitives[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        if best_dist < T_MAL:
            i, j = best_pair
            merged = merge2(primitives[i], primitives[j])
            primitives.pop(j)
            primitives.pop(i)
            primitives.append(merged)
            changed = True

    return primitives


# ==================== 平面拟合 ====================

def fit_plane_method1(anchor_pos, all_representatives, n_select=NUM_RANDOM_PTS):
    """Method1: 以锚点为中心，选欧氏距离最近的 n_select 个代表点，SVD 拟合"""
    if len(all_representatives) < n_select:
        return None, None, None, None
    dists = np.linalg.norm(all_representatives - anchor_pos, axis=1)
    nearest_idx = np.argsort(dists)[:n_select]
    selected = all_representatives[nearest_idx]
    centroid, normal = fit_plane_svd(selected)
    return centroid, normal, selected, nearest_idx


def fit_plane_method2a(anchor_prim, all_primitives):
    """Method2A: 从锚点基元出发，按马氏距离排序邻近基元，逐步融合，第二特征值 > EIGEN_STOP 时停止
    特征值计算时使用有偏协方差：cov_biased = cov_geo * n/(n-1)
    返回: (centroid, normal, n_pts, merged_positions, fused_primitive, merged_prims)"""
    if not all_primitives:
        return None, None, None, np.empty((0, 3)), None, []

    # 按马氏距离排序
    remaining = sorted(all_primitives,
                       key=lambda p: mal_distance(anchor_prim, p))
    current = remaining.pop(0)  # 锚点自身
    merged_positions = [current['pos'].copy()]
    merged_prims = [current]  # 记录每个参与融合的基元

    while remaining:
        candidate = merge2(current, remaining[0])
        # 有偏修正：乘以 n/(n-1)
        n = candidate['pt_num']
        if n > 1:
            cov_biased = candidate['cov_geo'] * (n / (n - 1.0))
        else:
            cov_biased = candidate['cov_geo']
        evals, _ = np.linalg.eigh(cov_biased + LIDAR_COV * np.eye(3))

        if evals[1] > EIGEN_STOP:
            break

        merged_prims.append(remaining[0])
        current = candidate
        merged_positions.append(remaining[0]['pos'].copy())
        remaining.pop(0)

    centroid = current['pos']
    n = current['pt_num']
    if n > 1:
        cov_biased = current['cov_geo'] * (n / (n - 1.0))
    else:
        cov_biased = current['cov_geo']
    evals, evecs = np.linalg.eigh(cov_biased + LIDAR_COV * np.eye(3))
    normal = evecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal, current['pt_num'], np.array(merged_positions), current, merged_prims


def fit_plane_method2b(anchor_prim, all_primitives):
    """Method2B: 从锚点基元出发，按马氏距离排序邻近基元，融合到来自 >= VOXEL_STOP 个体素时停止
    返回: (centroid, normal, n_pts, merged_positions, fused_primitive, merged_prims)"""
    if not all_primitives:
        return None, None, None, np.empty((0, 3)), None, []

    remaining = sorted(all_primitives,
                       key=lambda p: mal_distance(anchor_prim, p))
    current = remaining.pop(0)
    merged_positions = [current['pos'].copy()]
    merged_prims = [current]  # 记录每个参与融合的基元

    while remaining:
        merged_prims.append(remaining[0])
        merged_positions.append(remaining[0]['pos'].copy())
        candidate = merge2(current, remaining[0])
        current = candidate
        remaining.pop(0)

        if len(current['voxel_ids']) >= VOXEL_STOP:
            break

    centroid = current['pos']
    evals, evecs = np.linalg.eigh(current['cov_geo'] + LIDAR_COV * np.eye(3))
    normal = evecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal, current['pt_num'], np.array(merged_positions), current, merged_prims


# ==================== 可视化辅助 ====================

def draw_ellipsoid(ax, center, cov, n_std=1.0, color='blue', alpha=0.15,
                   wireframe=False):
    """绘制 3D 协方差椭球（不含 label，图例由外部 proxy artist 处理）
    Args:
        center: 椭球中心 (3,)
        cov: 3x3 协方差矩阵
        n_std: 椭球缩放倍数（标准差个数）
        color: 颜色
        alpha: 透明度
        wireframe: 是否用线框模式绘制
    """
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-10)  # 防止负特征值

    # 生成单位球面
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 12)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    # 按特征值缩放 + 按特征向量旋转
    radii = n_std * np.sqrt(evals)
    sphere = np.stack([sx * radii[0], sy * radii[1], sz * radii[2]], axis=-1)
    sphere = sphere @ evecs.T + center

    if wireframe:
        ax.plot_wireframe(sphere[..., 0], sphere[..., 1], sphere[..., 2],
                          color=color, alpha=alpha, linewidth=0.5)
    else:
        ax.plot_surface(sphere[..., 0], sphere[..., 1], sphere[..., 2],
                        color=color, alpha=alpha)


def draw_plane(ax, centroid, normal, size=0.5, color='blue', alpha=0.3):
    """在 3D 图中绘制平面"""
    normal = normal / np.linalg.norm(normal)
    # 构造平面上的两个正交向量
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    corners = [
        centroid + size * (-u - v),
        centroid + size * (u - v),
        centroid + size * (u + v),
        centroid + size * (-u + v),
    ]
    poly = Poly3DCollection([corners], alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)


def draw_normal_arrow(ax, centroid, normal, length=0.3, color='red'):
    ax.quiver(centroid[0], centroid[1], centroid[2],
              normal[0] * length, normal[1] * length, normal[2] * length,
              color=color, linewidth=2, arrow_length_ratio=0.3)


# ==================== 选取用于展示的体素 ====================

def select_interesting_voxel(points, voxels, cloud_type):
    """选取一个有代表性的体素做详细展示"""
    best_key = None
    best_count = 0
    for key, indices in voxels.items():
        if len(indices) > best_count and len(indices) >= 5:
            best_count = len(indices)
            best_key = key
    return best_key


# ==================== 四方法统一实验 ====================

METHODS = ('FAST-LIO', 'VoxelMap', 'AKF-LIO', 'Ours')

def plane_from_points(pts):
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    n /= max(np.linalg.norm(n), 1e-12)
    return c, n, np.cov(pts.T, bias=True)

def plane_parameter_covariance(pts, noise_var):
    """VoxelMap-style first-order covariance of [normal, centroid]."""
    fit = plane_from_points(pts)
    if fit is None:
        return None
    c, n, cov = fit
    vals, vecs = np.linalg.eigh(cov)
    v0, v1, v2 = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    result = np.zeros((6, 6)); count = len(pts)
    for point, x in zip(pts, pts-c):
        jac_n = np.zeros((3, 3))
        for m, vm in ((1, v1), (2, v2)):
            gap = vals[0] - vals[m]
            if abs(gap) < 1e-12:
                continue
            f = x @ (np.outer(vm, v0) + np.outer(v0, vm)) / (count * gap)
            jac_n += np.outer(vm, f)
        jac = np.vstack([jac_n, np.eye(3) / count])
        point_cov = (noise_var*np.eye(3) if np.isscalar(noise_var)
                     else np.asarray(noise_var(point)))
        result += jac @ point_cov @ jac.T
    return result

def make_result(center, normal, plane_cov, used, anchor, noise_var):
    j = np.r_[anchor - center, -normal]
    r_map = float(j @ plane_cov @ j)
    query_r = (float(normal@noise_var(anchor)@normal) if callable(noise_var)
               else float(noise_var))
    predicted_r = max(query_r + r_map, 1e-12)
    return (center,normal,plane_cov[:3,:3],predicted_r,used,predicted_r,
            max(r_map,0.0))

def representative_points(points, voxels):
    out = []
    for key, ids in voxels.items():
        center = voxel_center(key, VOXEL_SIZE)
        j = ids[int(np.argmin(np.linalg.norm(points[ids] - center, axis=1)))]
        out.append(points[j])
    return np.asarray(out)

def lidar_measurement_covariance(p):
    """Exact model in covariance_propagation.cc, before the 0.001 I floor."""
    r = max(float(np.linalg.norm(p)), 1e-6); u = p / r
    e1 = np.cross(u, [0., 0., 1.])
    if e1 @ e1 < 1e-8:
        e1 = np.cross(u, [1., 0., 0.])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1); e2 /= np.linalg.norm(e2)
    a = r * np.sin(np.deg2rad(LIDAR_ANGLE_STD_DEG))
    return (LIDAR_RANGE_STD**2 * np.outer(u, u)
            + a*a * (np.outer(e1, e1) + np.outer(e2, e2)))

def ours_point_covariance(p):
    return MODEL_POINT_VAR*np.eye(3) + lidar_measurement_covariance(p)

def point_primitive(p, idx, mode):
    cov_pos = MODEL_POINT_VAR*np.eye(3)
    if mode == 'ours':
        cov_pos = ours_point_covariance(p)
    return {'pos': p.copy(), 'cov_pos': cov_pos, 'cov_geo': np.zeros((3,3)),
            'cov_akf': MODEL_POINT_VAR*np.eye(3), 'pt_num': 1, 'ids': [idx]}

def stable_quadratic(d, cov):
    try:
        return float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        return float(d @ np.linalg.pinv(cov, hermitian=True) @ d)

def primitive_distance(a, b, mode):
    d = a['pos'] - b['pos']
    if mode == 'akf':
        cov = a['cov_akf'] + b['cov_akf']
    elif a['pt_num'] > 1 and b['pt_num'] > 1:
        cov = a['cov_geo'] + b['cov_geo']
    elif a['pt_num'] > 1:
        cov = a['cov_geo'] + b['cov_pos']
    elif b['pt_num'] > 1:
        cov = a['cov_pos'] + b['cov_geo']
    else:
        cov = a['cov_pos'] + b['cov_pos']
    return stable_quadratic(d, cov + 1e-12*np.eye(3))

def merge_primitive(a, b, mode):
    """Source Merge2(a,b); b supplies non-geometric fields in C++."""
    n1, n2 = a['pt_num'], b['pt_num']; n = n1+n2
    r1, r2 = n1/n, n2/n; p = r1*a['pos'] + r2*b['pos']
    out = {'pos': p, 'pt_num': n, 'ids': a['ids'] + b['ids']}
    if mode == 'akf':
        out['cov_akf'] = (r1*(a['cov_akf'] + np.outer(a['pos'],a['pos']))
                          + r2*(b['cov_akf'] + np.outer(b['pos'],b['pos']))
                          - np.outer(p,p))
        out['cov_pos'] = np.zeros((3,3)); out['cov_geo'] = out['cov_akf']
    else:
        out['cov_pos'] = r1*r1*a['cov_pos'] + r2*r2*b['cov_pos']
        da, db = a['pos']-p, b['pos']-p
        out['cov_geo'] = (((n1-1)*a['cov_geo'] + (n2-1)*b['cov_geo']
                           + n1*np.outer(da,da) + n2*np.outer(db,db))/(n-1))
        out['cov_akf'] = np.zeros((3,3))
    return out

def grid_key(p):
    return tuple(np.floor((p-VOXEL_ORIGIN)/VOXEL_SIZE).astype(int))

def neighboring_keys(key):
    return [(key[0]+d[0], key[1]+d[1], key[2]+d[2]) for d in NEIGHBOR_OFFSETS]

def build_ivox(points, mode):
    """Sequential AddPoints: nearest passing candidate per voxel, then global nearest."""
    grid = {}
    for idx, p in enumerate(points):
        new = point_primitive(p, idx, mode); candidates = []
        insertion_keys = (neighboring_keys(grid_key(p)) if mode == 'akf' or OURS_CROSS_VOXEL_MERGE
                          else [grid_key(p)])
        for key in insertion_keys:
            cell = grid.get(key)
            if not cell:
                continue
            ds = [primitive_distance(q, new, mode) for q in cell]
            j = int(np.argmin(ds))
            if ds[j] < T_MAL:
                candidates.append((ds[j], key, j))
        if candidates:
            _, old_key, j = min(candidates, key=lambda x:x[0])
            old = grid[old_key].pop(j)
            if not grid[old_key]:
                del grid[old_key]
            new = merge_primitive(new, old, mode)
        grid.setdefault(grid_key(new['pos']), []).append(new)
    return grid

def query_ivox(grid, anchor, mode, max_num=100):
    query = point_primitive(anchor, -1, mode); candidates = []
    for key in neighboring_keys(grid_key(anchor)):
        for q in grid.get(key, []):
            candidates.append((primitive_distance(q, query, mode), q))
    candidates.sort(key=lambda x:x[0])
    return [q for _,q in candidates[:max_num]]

def eig_p2q(prim, anchor, mode):
    cov = prim['cov_akf'] if mode == 'akf' else prim['cov_geo']
    vals, vecs = np.linalg.eigh(cov); delta = prim['pos']-anchor
    p2q = sum((delta @ vecs[:,i])**2/max(vals[i],1e-12) for i in (1,2))
    return vals, vecs, float(p2q)

def posthoc_normal_covariance(points, ids):
    """Common diagnostic only; it is not an AKF-LIO state variable."""
    return plane_parameter_covariance(points[ids], MODEL_POINT_VAR)[:3,:3]

def ours_normal_covariance(prim, vals, vecs):
    """Reconstruct Sigma_u0 from the active formula around laser_mapping.cc:1420."""
    v0,v1,v2 = vecs[:,0],vecs[:,1],vecs[:,2]; s = prim['cov_pos']
    nratio = prim['pt_num']/max(prim['pt_num']-1, 1)
    g1=max(vals[1]-vals[0],1e-8); g2=max(vals[2]-vals[0],1e-8)
    t1=nratio*(vals[1]*(v0@s@v0)+vals[0]*(v1@s@v1))/(g1*g1)
    t2=nratio*(vals[2]*(v0@s@v0)+vals[0]*(v2@s@v2))/(g2*g2)
    tc=nratio*vals[0]*(v1@s@v2)/(g2*g1)
    return (t1*np.outer(v1,v1)+t2*np.outer(v2,v2)
            +tc*(np.outer(v1,v2)+np.outer(v2,v1)))

def fit_fast(points, voxels, anchor, noise_var):
    reps = representative_points(points, voxels)
    if len(reps) < 5:
        return None
    sel = reps[np.argsort(np.linalg.norm(reps - anchor, axis=1))[:5]]
    # FAST-LIO's Ax=-1 QR fit, including its 0.1 m pointwise gate.
    coef, *_ = np.linalg.lstsq(sel, -np.ones(5), rcond=None)
    norm = np.linalg.norm(coef)
    if norm < 1e-12:
        return None
    n = coef / norm; d = 1.0 / norm
    mean = sel.mean(axis=0); c = mean - (mean @ n + d) * n
    if np.max(np.abs(sel @ n + d)) > 0.1:
        return None
    cov = plane_parameter_covariance(sel, noise_var)
    return make_result(c, n, cov, sel, anchor, noise_var)

def fit_voxelmap(points, voxels, anchor, noise_var):
    """VoxelMap octree construction and all-leaf probabilistic association.

    The source does *not* merge neighbouring leaves for a fit and it does not
    follow only the query-containing child.  A non-planar node is split into
    all eight children; residual construction recursively tests every planar
    leaf and retains the candidate with the greatest Gaussian likelihood.
    """
    # VoxelMap uses a coarse root voxel and refines it, unlike the fixed 0.5 m
    # grids used by the other methods in this comparison.
    root_voxels=voxelize(points,VOXELMAP_ROOT_SIZE)
    key=tuple(np.floor((anchor-VOXEL_ORIGIN)/VOXELMAP_ROOT_SIZE).astype(int))
    ids=root_voxels.get(key,[])
    if len(ids) <= 5:
        return None

    candidates=[]
    query_cov=(noise_var(anchor) if callable(noise_var)
               else float(noise_var)*np.eye(3))

    def build_node(node_ids, center, side, depth):
        # VoxelMap uses `size > layer_point_size[layer]` with size threshold 5.
        if len(node_ids) <= 5:
            return
        p=points[node_ids]; fit=plane_from_points(p)
        if fit is None:
            return
        c,n,scatter=fit
        if np.linalg.eigvalsh(scatter)[0] < 0.01:
            plane_cov=plane_parameter_covariance(p,noise_var)
            delta=anchor-c
            radius=np.sqrt(max(np.linalg.eigvalsh(scatter)[-1],0.0))
            distance=abs(float(n@delta))
            tangent_sq=max(float(delta@delta)-distance*distance,0.0)
            j=np.r_[delta,-n]
            plane_var=max(float(j@plane_cov@j),0.0)
            total_var=max(plane_var+float(n@query_cov@n),1e-12)
            # build_single_residual: finite plane-radius test and 3-sigma gate.
            if np.sqrt(tangent_sq) <= 3.0*radius and distance < 3.0*np.sqrt(total_var):
                probability=np.exp(-0.5*distance*distance/total_var)/np.sqrt(total_var)
                candidates.append((probability,c,n,plane_cov,p,total_var,plane_var))
            return
        if depth >= VOXELMAP_MAX_LAYER:
            return
        child_side=side/2.0
        bits=(p>=center).astype(int)
        for bx in (0,1):
            for by in (0,1):
                for bz in (0,1):
                    mask=np.all(bits==np.array([bx,by,bz]),axis=1)
                    if not np.any(mask):
                        continue
                    child_ids=np.asarray(node_ids)[mask]
                    direction=2*np.array([bx,by,bz])-1
                    child_center=center+direction*(side/4.0)
                    build_node(child_ids,child_center,child_side,depth+1)

    root_center=voxel_center(key,VOXELMAP_ROOT_SIZE)
    build_node(np.asarray(ids,dtype=int),root_center,VOXELMAP_ROOT_SIZE,0)
    if not candidates:
        return None
    _,c,n,plane_cov,used,total_var,plane_var=max(candidates,key=lambda x:x[0])
    return (c,n,plane_cov[:3,:3],total_var,used,total_var,plane_var)

def fit_akf(points, grid, anchor, noise_var, threshold=0.04):
    near = query_ivox(grid, anchor, 'akf')
    if not near: return None
    current = near[0]
    for q in near[1:]:
        if current['pt_num'] < 5:
            current = merge_primitive(q,current,'akf'); continue
        vals,_,p2q = eig_p2q(current,anchor,'akf')
        if vals[1] < threshold or p2q > 1.0:
            current = merge_primitive(q,current,'akf')
        else: break
    vals,vecs,p2q=eig_p2q(current,anchor,'akf')
    if current['pt_num']<5 or vals[1]<threshold or p2q>1.0: return None
    normal=vecs[:,0]; used=points[current['ids']]
    sig=posthoc_normal_covariance(points,current['ids'])
    # The checked workspace's fixed R=0.001 is an ablation setting, not an
    # estimated plane-error model.  Sigma_u0 here is explicitly post-hoc.
    return current['pos'],normal,sig,0.001,used,0.001,np.nan

def fit_ours(points, grid, anchor, noise_var,
             min_voxels=OURS_MIN_SUPPORT_VOXELS,
             max_voxels=OURS_MAX_SUPPORT_VOXELS, p2q_limit=9.0):
    near=query_ivox(grid,anchor,'ours')
    if not near: return None
    current=near[0]; support={grid_key(current['pos'])}
    for q in near[1:]:
        if len(support)>=max_voxels: break
        if current['pt_num']<5 or len(support)<min_voxels:
            support.add(grid_key(q['pos'])); current=merge_primitive(q,current,'ours'); continue
        _,_,p2q=eig_p2q(current,anchor,'ours')
        if p2q>p2q_limit:
            support.add(grid_key(q['pos'])); current=merge_primitive(q,current,'ours')
        else: break
    vals,vecs,p2q=eig_p2q(current,anchor,'ours')
    if len(support)<min_voxels or p2q>p2q_limit: return None
    normal=vecs[:,0]; sig=ours_normal_covariance(current,vals,vecs)
    delta=current['pos']-anchor; query_cov=point_primitive(anchor,-1,'ours')['cov_pos']
    plane_r=float(normal@current['cov_pos']@normal + delta@sig@delta)
    analytic_r=float(normal@query_cov@normal + plane_r)
    # Main formulation: thickness = thickness_p + thickness_pos + thickness_norm.
    # The fixed 0.001 line in the checked workspace is an ablation configuration.
    analytic_r=max(analytic_r,1e-12)
    return (current['pos'],normal,sig,analytic_r,points[current['ids']],
            analytic_r,max(plane_r,0.0))

def clean_scene_points(kind, rng, n=1200):
    """Generate latent geometry once; Monte Carlo trials must not resample it."""
    if kind == 'Cylinder':
        theta = rng.uniform(-np.pi, np.pi, n); z = rng.uniform(0.25, 3.25, n)
        clean = np.c_[1.50 + 0.30*np.cos(theta), 1.50 + 0.30*np.sin(theta), z]
        normals = np.c_[np.cos(theta), np.sin(theta), np.zeros(n)]
    elif kind == 'Corner':
        n1 = n // 2; n2 = n - n1
        p1 = np.c_[np.full(n1,1.25), rng.uniform(1.25,3.25,n1), rng.uniform(.25,3.25,n1)]
        p2 = np.c_[rng.uniform(1.25,3.25,n2), np.full(n2,1.25), rng.uniform(.25,3.25,n2)]
        clean = np.vstack([p1, p2]); normals = np.vstack([np.tile([1.,0,0], (n1,1)), np.tile([0,1.,0], (n2,1))])
    else:
        clean=np.c_[np.full(n,1.25),rng.uniform(1.25,3.25,n),
                    rng.uniform(.25,3.25,n)]
        normals=np.tile([1.,0.,0.],(n,1))
    return clean, normals

def inject_observation_noise(clean, rng, noise_model):
    """Independent repeated measurements of one fixed latent point cloud."""
    if noise_model == 'none':
        return clean.copy()
    if noise_model == 'isotropic':
        return clean + rng.normal(0, NOISE_STD, clean.shape)
    # Calibration experiment: data-generating covariance equals Ours' assumed
    # per-point covariance, including the source-code 0.001 I term.
    noise = np.vstack([rng.multivariate_normal(np.zeros(3), ours_point_covariance(p))
                       for p in clean])
    return clean + noise

def tangent_basis(normal):
    ref = np.array([0.,0.,1.]) if abs(normal[2]) < .9 else np.array([1.,0.,0.])
    b1 = np.cross(normal,ref); b1 /= np.linalg.norm(b1)
    b2 = np.cross(normal,b1)
    return np.column_stack([b1,b2])

def normal_tangent_error(estimate, truth):
    """Log-map-like angular error in the true normal's tangent plane."""
    cross=np.cross(truth,estimate); sine=np.linalg.norm(cross)
    angle=np.arctan2(sine,np.clip(truth@estimate,-1.,1.))
    return np.zeros(3) if sine < 1e-12 else angle*cross/sine

def run_experiment(trials=100, seed=42, noise_model='matched',
                   scenes=('Corner','Cylinder')):
    rng = np.random.default_rng(seed); rows = []; examples = {}
    trial_examples = {}; trial_inputs = {}
    best_corner_voxel_trial = None; best_corner_voxel_score = np.inf
    noise_var = (NOISE_STD**2 if noise_model == 'isotropic'
                 else (0.0 if noise_model == 'none' else ours_point_covariance))
    scene_offsets={'Corner':0,'Cylinder':1,'Plane':2}
    anchors={'Corner':np.array([1.25,1.35,1.60]),
             'Cylinder':np.array([1.80,1.50,1.60]),
             'Plane':np.array([1.25,2.25,1.60])}
    for scene in scenes:
        # A fixed latent cloud makes trial-to-trial variation attributable to
        # measurement noise and discrete association changes, not resampling.
        clean,normals=clean_scene_points(
            scene,np.random.default_rng(seed+scene_offsets[scene]))
        # The corner anchor lies on the x-facing wall, 0.10 m from the
        # intersection.  Plane and cylinder anchors also lie on their surfaces.
        anchor_true=anchors[scene]
        true_n = np.array([1.,0,0]); basis=tangent_basis(true_n)
        for k in range(trials):
            noisy=inject_observation_noise(clean,rng,noise_model)
            vox = voxelize(noisy, VOXEL_SIZE)
            akf_grid = build_ivox(noisy, 'akf')
            ours_grid = build_ivox(noisy, 'ours')
            # Independent current-frame query: never inserted into the map.
            if noise_model == 'matched':
                anchor=anchor_true+rng.multivariate_normal(np.zeros(3),ours_point_covariance(anchor_true))
            elif noise_model == 'isotropic':
                anchor=anchor_true+rng.normal(0,NOISE_STD,3)
            else: anchor=anchor_true.copy()
            examples.setdefault((scene,'_input'),(noisy,anchor,anchor_true.copy()))
            trial_inputs[(scene,k)]=(noisy,anchor,anchor_true.copy())
            funcs = (fit_fast, fit_voxelmap, fit_akf, fit_ours)
            maps = (vox, vox, akf_grid, ours_grid)
            for method, fn, map_data in zip(METHODS, funcs, maps):
                try: result = fn(noisy, map_data, anchor, noise_var)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError): result = None
                row = {'scene': scene, 'method': method, 'trial': k,
                       'valid': int(result is not None), 'noise_model': noise_model}
                if result is not None:
                    c,n,sig,pred_r,used,filter_r,plane_r=result
                    n *= 1 if np.dot(n, true_n) >= 0 else -1
                    used_arr=np.asarray(used)
                    # Sigma_g is support geometry (spread), not estimator
                    # uncertainty.  Sigma_p is uncertainty of the estimated
                    # support centroid under the independent-point model.
                    # VoxelMap uses a population scatter; G2M stores the
                    # corresponding unbiased merged scatter.
                    geo_cov=np.cov(used_arr.T,bias=(method=='VoxelMap'))
                    sigma_g=float(np.sqrt(max(n@geo_cov@n,0.0)))
                    if callable(noise_var):
                        center_cov=sum((noise_var(q) for q in used_arr),
                                       np.zeros((3,3)))/(len(used_arr)**2)
                    else:
                        center_cov=float(noise_var)*np.eye(3)/len(used_arr)
                    sigma_p=float(np.sqrt(max(n@center_cov@n,0.0)))
                    tang3=normal_tangent_error(n,true_n); tang=basis.T@tang3
                    sig2=basis.T@sig@basis
                    normal_sigma_deg=float(np.degrees(
                        np.sqrt(max(np.trace(sig2),0.0))))
                    normal_nees=stable_quadratic(tang,sig2+1e-12*np.eye(2))
                    residual=float(n@(anchor-c)); residual_nis=residual*residual/max(pred_r,1e-12)
                    row.update(angle_deg=np.degrees(np.arccos(np.clip(np.dot(n,true_n),-1,1))),
                               normal_error_sq=float(tang@tang), predicted_trace=float(np.trace(sig2)),
                               predicted_normal_sigma_deg=normal_sigma_deg,
                               tangent_1=float(tang[0]), tangent_2=float(tang[1]),
                               normal_nees=float(normal_nees), normalized_normal_nees=float(normal_nees/2),
                               predicted_R=float(pred_r), filter_R=float(filter_r),
                               predicted_plane_variance=float(plane_r),
                               support_normal_spread_sigma_g=sigma_g,
                               centroid_normal_uncertainty_sigma_p=sigma_p,
                               residual=residual,residual_nis=float(residual_nis),n_used=len(used),
                               noise_model=noise_model)
                    row['normal_covariance_source'] = ('native' if method in ('VoxelMap','Ours')
                                                       else 'post-hoc')
                    residual=float(n@(anchor-c))
                    example=(noisy,anchor,anchor_true.copy(),c,n,used,
                             residual,float(pred_r),float(plane_r),sigma_g,
                             sigma_p,normal_sigma_deg)
                    example_key=(scene,method)
                    trial_examples[(scene,method,k)]=example
                    examples.setdefault(example_key,example)
                    # Find the requested successful VoxelMap separation.  At
                    # the end, the whole Corner row is switched to this same
                    # Monte-Carlo trial so the visual comparison stays fair.
                    if scene=='Corner' and method=='VoxelMap':
                        score=float(row['angle_deg'])
                        if score < best_corner_voxel_score:
                            best_corner_voxel_score=score
                            best_corner_voxel_trial=k
                rows.append(row)
    if best_corner_voxel_trial is not None:
        k=best_corner_voxel_trial
        examples[('Corner','_input')]=trial_inputs[('Corner',k)]
        for method in METHODS:
            selected=trial_examples.get(('Corner',method,k))
            if selected is not None:
                examples[('Corner',method)]=selected
            else:
                examples.pop(('Corner',method),None)
    return rows, examples

def save_geometry_figure(examples,scenes=('Corner','Cylinder'),
                         output_stem='plane_fitting_comparison'):
    """Save an evenly spaced, high-contrast visualization of all plane fits."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(len(scenes),4,figsize=(20,4.6*len(scenes)),squeeze=False,
                          subplot_kw={'projection':'3d'})
    colors = ['#1f77b4','#9467bd','#d62728','#2ca02c']
    for r,scene in enumerate(scenes):
        for c, method in enumerate(METHODS):
            ax = axes[r,c]; ex = examples.get((scene,method))
            if ex:
                (pts,anchor,anchor_true,center,normal,used,residual,
                 predicted_r,plane_r,sigma_g,sigma_p,normal_sigma_deg)=ex
                ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=4,c='#8fa8c5',alpha=.48,
                           edgecolors='none',label='Raw cloud')
                used = np.asarray(used); show=used[::max(1,len(used)//400)]
                ax.scatter(show[:,0],show[:,1],show[:,2],s=15,c='#00a650',alpha=.95,
                           edgecolors='white',linewidths=.2,label='Fitting support')
                ax.scatter(*anchor_true,c='#111111',marker='*',s=95,edgecolors='white',
                           linewidths=.5,label='Surface anchor')
                ax.scatter(*anchor,facecolors='none',edgecolors='#ff8c00',marker='o',
                           s=58,linewidths=1.4,label='Noisy query')
                draw_plane(ax,center,normal,.48 if scene in ('Corner','Plane') else .36,
                           colors[c],.58)
                draw_normal_arrow(ax,center,normal,.34,'#e31a1c')
                # Every selected anchor has the same outward truth normal.
                true_normal=np.array([1.,0.,0.])
                angle_error=np.degrees(np.arccos(
                    np.clip(abs(float(normal@true_normal)),-1.,1.)))
                truth_end=center+.34*true_normal
                ax.plot([center[0],truth_end[0]],[center[1],truth_end[1]],
                        [center[2],truth_end[2]],color='#111111',ls='--',
                        lw=2.0,zorder=10)
                if method in ('VoxelMap','Ours'):
                    error_text=(rf'$|r|={abs(residual):.4f}\,\mathrm{{m}}$'+'\n'
                                +rf'$\Delta\theta={angle_error:.2f}^\circ$'+'\n'
                                +rf'$\sigma_{{plane}}={np.sqrt(plane_r):.4f}\,\mathrm{{m}}$'+'\n'
                                +rf'$\sigma_n={normal_sigma_deg:.2f}^\circ$')
                else:
                    error_text=(rf'$|r|={abs(residual):.4f}\,\mathrm{{m}}$'+'\n'
                                +rf'$\Delta\theta={angle_error:.2f}^\circ$')
                ax.text2D(.035,.94,error_text,transform=ax.transAxes,va='top',ha='left',
                          fontsize=8.2,color='#151515',
                          bbox=dict(boxstyle='round,pad=.28',facecolor='white',
                                    edgecolor='#777777',alpha=.88,linewidth=.6))
            else:
                pts,anchor,anchor_true=examples[(scene,'_input')]
                ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=4,c='#8fa8c5',alpha=.48,
                           edgecolors='none')
                ax.scatter(*anchor_true,c='#111111',marker='*',s=90,edgecolors='white',
                           linewidths=.5)
                ax.scatter(*anchor,facecolors='none',edgecolors='#ff8c00',marker='o',
                           s=58,linewidths=1.4)
                ax.text2D(.035,.94,'No valid plane',transform=ax.transAxes,
                          va='top',ha='left',fontsize=8.5,color='#9b1c1c',
                          bbox=dict(boxstyle='round,pad=.28',facecolor='white',
                                    edgecolor='#9b1c1c',alpha=.90,linewidth=.7))
            if scene in ('Corner','Plane'):
                ax.set_xlim(1.05,3.45); ax.set_ylim(1.05,3.45); ax.set_zlim(.05,3.45)
                ax.set_xticks(np.arange(1.0,3.6,.5)); ax.set_yticks(np.arange(1.0,3.6,.5)); ax.set_zticks(np.arange(.5,3.6,.5))
            else:
                ax.set_xlim(1.05,1.95); ax.set_ylim(1.05,1.95); ax.set_zlim(.05,3.45)
                ax.set_xticks(np.arange(1.1,2.0,.2)); ax.set_yticks(np.arange(1.1,2.0,.2)); ax.set_zticks(np.arange(.5,3.6,.5))
            # Identical display boxes make the two rows occupy equal space.  Data
            # units are intentionally not forced to equal scale for the tall cylinder.
            ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho')
            ax.set_xlabel('$x$ (m)',labelpad=6); ax.set_ylabel('$y$ (m)',labelpad=6)
            ax.set_zlabel('$z$ (m)',labelpad=6)
            ax.tick_params(axis='both',which='major',labelsize=8,pad=1,width=.8,
                           colors='#252525')
            title=f'{scene} — {method}'
            ax.set_title(title,pad=7,fontsize=11)
            ax.view_init(24,-55); ax.grid(True,alpha=.30)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.pane.set_facecolor((.96,.97,.98,1)); axis.pane.set_alpha(.55)
                axis._axinfo['grid']['color']=(.45,.45,.45,.38)
                axis._axinfo['grid']['linewidth']=.65
    legend=[Line2D([0],[0],marker='o',ls='',color='#8fa8c5',markersize=5,label='Raw cloud'),
            Line2D([0],[0],marker='o',ls='',color='#00a650',markersize=6,label='Fitting support'),
            Line2D([0],[0],marker='*',ls='',color='#111111',markersize=9,label='Surface anchor'),
            Line2D([0],[0],marker='o',ls='',markerfacecolor='none',
                   markeredgecolor='#ff8c00',markersize=6,label='Noisy query'),
            Patch(facecolor='#4c91c7',alpha=.58,label='Fitted plane'),
            Line2D([0],[0],color='#e31a1c',lw=2,label='Estimated normal'),
            Line2D([0],[0],color='#111111',ls='--',lw=2,label='Ground-truth normal')]
    if len(scenes)==1:
        fig.legend(handles=legend,loc='lower center',ncol=7,frameon=True,
                   bbox_to_anchor=(.5,.01),fontsize=9)
        fig.subplots_adjust(left=.025,right=.985,bottom=.15,top=.94,wspace=.04)
    else:
        fig.legend(handles=legend,loc='upper center',ncol=7,frameon=True,
                   bbox_to_anchor=(.5,.995),fontsize=9)
        fig.subplots_adjust(left=.025,right=.985,bottom=.045,top=.925,wspace=.04,hspace=.14)
    fig.savefig(output_path(f'{output_stem}.png'),dpi=240)
    fig.savefig(output_path(f'{output_stem}_small.jpg'),dpi=125,
                pil_kwargs={'quality':82,'optimize':True,'progressive':True})
    plt.close(fig)

def save_outputs(rows, examples):
    import csv
    colors = ['#1f77b4','#9467bd','#d62728','#2ca02c']
    out_csv = output_path('plane_fitting_results.csv')
    preferred = ['scene','method','trial','valid','noise_model','normal_covariance_source',
                 'angle_deg','tangent_1','tangent_2','normal_error_sq','predicted_trace',
                 'predicted_normal_sigma_deg',
                 'normal_nees','normalized_normal_nees','predicted_R','filter_R',
                 'predicted_plane_variance','support_normal_spread_sigma_g',
                 'centroid_normal_uncertainty_sigma_p','residual','residual_nis','n_used']
    present = set().union(*(r.keys() for r in rows))
    keys = [k for k in preferred if k in present]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    save_geometry_figure(examples)
    # Summary statistics: only normal uncertainty versus actual normal error.
    fig2, (a,b) = plt.subplots(1,2,figsize=(15,5.5))
    labels=[]; angle=[]; pred_trace=[]; emp_trace=[]; cover=[]; valid_rate=[]
    summary=[]
    for scene in ('Corner','Cylinder'):
        for method in METHODS:
            all_x=[q for q in rows if q['scene']==scene and q['method']==method]
            x=[q for q in all_x if q['valid']]
            labels.append(scene+'\n'+method); valid_rate.append(len(x)/len(all_x))
            if x:
                tang=np.array([[q['tangent_1'],q['tangent_2']] for q in x]); res=np.array([q['residual'] for q in x])
                et=float(np.trace(np.cov(tang.T, bias=True))); pt=float(np.mean([q['predicted_trace'] for q in x]))
                ang=float(np.sqrt(np.mean(np.square([q['angle_deg'] for q in x])))); rb=float(np.mean(res))
                nnees=float(np.mean([q['normalized_normal_nees'] for q in x])); nis=float(np.mean([q['residual_nis'] for q in x]))
                from scipy.stats import chi2
                ncov=float(np.mean([(chi2.ppf(.025,2)<=q['normal_nees']<=chi2.ppf(.975,2)) for q in x]))
            else:
                et=pt=ang=rb=nnees=nis=ncov=np.nan
            angle.append(ang); emp_trace.append(et); pred_trace.append(pt); cover.append(ncov)
            summary.append(dict(scene=scene,method=method,valid_rate=valid_rate[-1],angle_rmse_deg=ang,
                                normal_covariance_source=('native' if method in ('VoxelMap','Ours') else 'post-hoc'),
                                empirical_normal_cov_trace=et,predicted_normal_cov_trace=pt,
                                normalized_normal_nees=nnees,normal_95_coverage=ncov))
    with open(output_path('plane_fitting_summary.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    xx=np.arange(len(labels)); width=.36
    a.bar(xx-width/2,angle,width,color=np.tile(colors,2)); a.set_ylabel('Normal angular RMSE (deg)'); a.set_xticks(xx,labels,rotation=45,ha='right'); a.set_title('Actual normal estimation error')
    b.bar(xx-width/2,emp_trace,width,label='Empirical tangent error variance'); b.bar(xx+width/2,pred_trace,width,label='Predicted $tr(\\Sigma_{u_0})$'); b.set_yscale('log'); b.set_ylabel(r'Normal uncertainty / error (rad$^2$)'); b.set_xticks(xx,labels,rotation=45,ha='right'); b.legend(); b.set_title('Predicted uncertainty vs actual error')
    fig2.tight_layout(); fig2.savefig(output_path('plane_fitting_uncertainty.png'), dpi=240)

    # Direct calibration diagnostics.  For a calibrated model normalized mean
    # is one and empirical chi-square coverage follows the diagonal.
    from scipy.stats import chi2
    fig3,cal_axes=plt.subplots(2,3,figsize=(15,8.4),squeeze=False,
                              constrained_layout=True)
    nominal=np.linspace(.05,.99,40)
    for rr,scene in enumerate(('Corner','Cylinder')):
        axm,axn,axr=cal_axes[rr]
        groups=[]
        for method in METHODS:
            x=[q for q in rows if q['scene']==scene and q['method']==method and q['valid']]
            groups.append(x)
        xpos=np.arange(len(METHODS)); bw=.36
        nm=[np.mean([q['normalized_normal_nees'] for q in x]) if x else np.nan for x in groups]
        im=[np.mean([q['residual_nis'] for q in x]) if x else np.nan for x in groups]
        axm.bar(xpos-bw/2,nm,bw,label='Normal NEES / 2')
        axm.bar(xpos+bw/2,im,bw,label='Residual NIS')
        axm.axhline(1,color='k',ls='--',lw=1,label='Calibrated')
        axm.set_xticks(xpos,METHODS,rotation=20); axm.set_yscale('log')
        axm.set_ylabel('Normalized mean (ideal = 1)'); axm.set_title(scene)
        if rr==0: axm.legend(fontsize=7)
        for method,color,x in zip(METHODS,colors,groups):
            if not x: continue
            ne=np.array([q['normal_nees'] for q in x]); ni=np.array([q['residual_nis'] for q in x])
            style='-' if method in ('VoxelMap','Ours') else '--'
            axn.plot(nominal,[np.mean(ne<=chi2.ppf(a,2)) for a in nominal],style,color=color,label=method)
            axr.plot(nominal,[np.mean(ni<=chi2.ppf(a,1)) for a in nominal],style,color=color,label=method)
        for ax,title in ((axn,r'Normal NEES coverage ($\chi^2_2$)'),
                         (axr,r'Residual NIS coverage ($\chi^2_1$)')):
            ax.plot([0,1],[0,1],'k:',lw=1,label='Ideal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('Nominal probability')
            ax.set_ylabel('Empirical probability'); ax.set_title(title); ax.grid(alpha=.2)
        if rr==0: axr.legend(fontsize=7,ncol=2)
    noise_label=rows[0].get('noise_model','unknown') if rows else 'unknown'
    fig3.suptitle('Uncertainty calibration on a fixed latent cloud '
                  f'(noise: {noise_label})\nsolid: native covariance; '
                  'dashed: post-hoc diagnostic')
    fig3.savefig(output_path('plane_fitting_calibration.png'),dpi=240)
    save_normal_calibration(rows, colors)
    print(f'Saved {out_csv}, plane_fitting_summary.csv, plane_fitting_comparison.png, '
          'plane_fitting_uncertainty.png, plane_fitting_calibration.png')

def save_plane_experiment(trials=30,seed=42,noise_model='matched'):
    """Run and save the isolated single-plane four-method comparison."""
    import csv
    rows,examples=run_experiment(trials,seed,noise_model,scenes=('Plane',))
    save_geometry_figure(examples,scenes=('Plane',),
                         output_stem='plane_fitting_flat_comparison')
    preferred=['scene','method','trial','valid','noise_model','normal_covariance_source',
               'angle_deg','normal_error_sq','predicted_trace','normal_nees',
               'predicted_normal_sigma_deg',
               'normalized_normal_nees','predicted_R',
               'support_normal_spread_sigma_g',
               'centroid_normal_uncertainty_sigma_p','residual','n_used']
    preferred.insert(-2,'predicted_plane_variance')
    present=set().union(*(r.keys() for r in rows)); keys=[k for k in preferred if k in present]
    with open(output_path('plane_fitting_flat_results.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print('Saved plane_fitting_flat_comparison.png, '
          'plane_fitting_flat_comparison_small.jpg, plane_fitting_flat_results.csv')

def save_normal_calibration(rows, colors=None):
    """Plot only predicted normal uncertainty against actual normal error."""
    from scipy.stats import chi2
    if colors is None: colors=['#1f77b4','#9467bd','#d62728','#2ca02c']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    nominal=np.linspace(.01,.99,60); xpos=np.arange(len(METHODS))
    for col,scene in enumerate(('Corner','Cylinder')):
        groups=[[q for q in rows if q['scene']==scene and q['method']==method
                 and float(q.get('valid',0))>0] for method in METHODS]
        means=[np.mean([float(q['normalized_normal_nees']) for q in x]) if x else np.nan
               for x in groups]
        ax=axes[0,col]
        ax.bar(xpos,means,color=colors)
        ax.axhline(1,color='k',ls='--',lw=1.2,label='Ideal calibration')
        ax.set_xticks(xpos,METHODS,rotation=18); ax.set_yscale('log')
        ax.set_ylabel('Mean normal NEES / 2 (ideal = 1)')
        ax.set_title(f'{scene}: normalized normal error')
        ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8)
        ax=axes[1,col]
        for method,color,x in zip(METHODS,colors,groups):
            if not x: continue
            ne=np.array([float(q['normal_nees']) for q in x])
            style='-' if method in ('VoxelMap','Ours') else '--'
            ax.plot(nominal,[np.mean(ne<=chi2.ppf(a,2)) for a in nominal],
                    style,color=color,lw=1.8,label=method)
        ax.plot([0,1],[0,1],'k:',lw=1.3,label='Ideal')
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Nominal probability'); ax.set_ylabel('Empirical probability')
        ax.set_title(r'Normal-error coverage against $\chi^2_2$')
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    noise_label=next((q.get('noise_model','unknown') for q in rows if q.get('noise_model')),'unknown')
    fig.suptitle('Predicted normal uncertainty vs actual normal estimation error '
                 f'(fixed cloud, noise: {noise_label})')
    fig.savefig(output_path('plane_fitting_calibration.png'),dpi=240)

def replot_calibration_from_csv(path=None):
    """Rebuild the calibration panel without repeating expensive map trials."""
    path = output_path("plane_fitting_results.csv") if path is None else Path(path)
    import csv
    from scipy.stats import chi2
    numeric={'trial','valid','angle_deg','tangent_1','tangent_2','normal_error_sq',
             'predicted_trace','normal_nees','normalized_normal_nees','predicted_R',
             'filter_R','residual','residual_nis','n_used'}
    rows=[]
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            for key in numeric:
                if row.get(key,'')!='': row[key]=float(row[key])
            rows.append(row)
    save_normal_calibration(rows)
    print('Replotted plane_fitting_calibration.png from',path)
    return
    colors=['#1f77b4','#9467bd','#d62728','#2ca02c']
    fig,axes=plt.subplots(2,3,figsize=(15,8.4),squeeze=False,
                          constrained_layout=True)
    nominal=np.linspace(.05,.99,40)
    for rr,scene in enumerate(('Corner','Cylinder')):
        axm,axn,axr=axes[rr]
        groups=[[q for q in rows if q['scene']==scene and q['method']==method
                 and q.get('valid',0)>0] for method in METHODS]
        xpos=np.arange(len(METHODS)); bw=.36
        nm=[np.mean([q['normalized_normal_nees'] for q in x]) if x else np.nan for x in groups]
        im=[np.mean([q['residual_nis'] for q in x]) if x else np.nan for x in groups]
        axm.bar(xpos-bw/2,nm,bw,label='Normal NEES / 2')
        axm.bar(xpos+bw/2,im,bw,label='Residual NIS')
        axm.axhline(1,color='k',ls='--',lw=1,label='Calibrated')
        axm.set_xticks(xpos,METHODS,rotation=20); axm.set_yscale('log')
        axm.set_ylabel('Normalized mean (ideal = 1)'); axm.set_title(scene)
        if rr==0: axm.legend(fontsize=7)
        for method,color,x in zip(METHODS,colors,groups):
            if not x: continue
            ne=np.array([q['normal_nees'] for q in x]); ni=np.array([q['residual_nis'] for q in x])
            style='-' if method in ('VoxelMap','Ours') else '--'
            axn.plot(nominal,[np.mean(ne<=chi2.ppf(a,2)) for a in nominal],style,color=color,label=method)
            axr.plot(nominal,[np.mean(ni<=chi2.ppf(a,1)) for a in nominal],style,color=color,label=method)
        for ax,title in ((axn,r'Normal NEES coverage ($\chi^2_2$)'),
                         (axr,r'Residual NIS coverage ($\chi^2_1$)')):
            ax.plot([0,1],[0,1],'k:',lw=1,label='Ideal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('Nominal probability')
            ax.set_ylabel('Empirical probability'); ax.set_title(title); ax.grid(alpha=.2)
        if rr==0: axr.legend(fontsize=7,ncol=2)
    noise_label=next((q.get('noise_model','unknown') for q in rows if q.get('noise_model')),'unknown')
    fig.suptitle('Uncertainty calibration on a fixed latent cloud '
                 f'(noise: {noise_label})\nsolid: native covariance; '
                 'dashed: post-hoc diagnostic')
    fig.savefig(output_path('plane_fitting_calibration.png'),dpi=240)
    print('Replotted plane_fitting_calibration.png from',path)

def main():
    global OUTPUT_DIR
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--trials',type=int,default=100); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--noise-model',choices=('matched','isotropic','none'),default='matched',
                   help='matched: sample Ours assumed covariance; isotropic: 5 mm; none: deterministic cloud')
    p.add_argument('--replot-calibration',action='store_true',
                   help='rebuild calibration PNG from the existing trial CSV')
    p.add_argument('--flat-plane',action='store_true',
                   help='generate the isolated single-plane comparison')
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="directory for generated plots and CSV files")
    args=p.parse_args()
    if args.trials < 2:
        p.error("--trials must be at least 2 for covariance diagnostics")
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.replot_calibration:
        replot_calibration_from_csv(); return
    if args.flat_plane:
        save_plane_experiment(args.trials,args.seed,args.noise_model); return
    rows, examples=run_experiment(args.trials,args.seed,args.noise_model); save_outputs(rows,examples)


if __name__ == '__main__':
    main()
