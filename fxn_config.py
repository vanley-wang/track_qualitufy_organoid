"""
FXN 数据专用配置文件
适配 nnUNet_FXN_2023/FXN_0701 和 FXN_0703 的类器官追踪分析
"""

import os

# ============ 图像物理尺寸配置 ============
# 体素大小 (微米 μm)
# 注意：NIfTI header 中 pixdim 显示为 1.0，但实际显微镜分辨率可能不同
# 默认沿用项目 config.py 中的参数，如有不同请在此修改
VOXEL_SIZE_X = 5.0  # μm
VOXEL_SIZE_Y = 5.0  # μm
VOXEL_SIZE_Z = 5.5  # μm

# 图像尺寸将从 NIfTI 文件动态读取，此处仅作参考
IMAGE_SIZE_X = 800
IMAGE_SIZE_Y = 512
IMAGE_SIZE_Z = 800

# 计算单个体素的实际体积 (立方微米)
VOXEL_VOLUME_UM3 = VOXEL_SIZE_X * VOXEL_SIZE_Y * VOXEL_SIZE_Z

# 单位转换常数
UM3_TO_MM3 = 1e-9
UM2_TO_MM2 = 1e-6


# ============ ID 追踪配置 ============
# 最大匹配距离 (像素)
# 如果类器官在两天之间移动较大，可适当增大
MAX_DISTANCE_PIXELS = 100  # 对应 500 μm (100 * 5 μm)

# 最小连通域体积（体素数），过滤噪声
# 假设最小类器官直径约 50 μm，体积约 (4/3)*pi*(25)^3 ≈ 65450 μm³
# 体素体积 137.5 μm³ → 约 476 体素。取整 500 作为保守阈值
MIN_VOXEL_COUNT = 500


# ============ 量化参数阈值 ============
# 药效评估阈值 (体积变化率)
THRESHOLD_COMPLETE_RESPONSE = -0.9   # 缩小 90% 以上
THRESHOLD_PARTIAL_RESPONSE = -0.5    # 缩小 50% 以上
THRESHOLD_STABLE_DISEASE_MIN = -0.3  # -30%
THRESHOLD_STABLE_DISEASE_MAX = 0.3   # +30%
THRESHOLD_PROGRESSIVE_DISEASE = 0.3  # 增长 30% 以上


# ============ 文件路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入数据路径
FXN_0701_RAW_DIR = os.path.join(BASE_DIR, "nnUNet_FXN_2023", "FXN_0701")
FXN_0701_SEG_DIR = os.path.join(BASE_DIR, "nnUNet_FXN_2023", "FXN_0701_seg")
FXN_0703_RAW_DIR = os.path.join(BASE_DIR, "nnUNet_FXN_2023", "FXN_0703")
FXN_0703_SEG_DIR = os.path.join(BASE_DIR, "nnUNet_FXN_2023", "FXN_0703_seg")

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results_fxn")


# ============ 可视化配置 ============
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

# 两天颜色
COLOR_DAY3 = '#3498db'  # 蓝色
COLOR_DAY5 = '#2ecc71'  # 绿色

# 药效评估颜色
COLOR_CR = '#27ae60'   # 完全缓解 - 深绿
COLOR_PR = '#95a5a6'   # 部分缓解 - 灰
COLOR_SD = '#f39c12'   # 疾病稳定 - 橙
COLOR_PD = '#c0392b'   # 疾病进展 - 红

# 3D Slicer 用颜色表（每个 Segment 一个颜色）
# 使用 tab20 调色板预生成 20 种颜色，循环使用
import numpy as np

def generate_slicer_colors(n=20):
    """生成 3D Slicer 用的 RGB 颜色字符串列表"""
    cmap = [
        (1.0, 0.0, 0.0),      # 红
        (0.0, 1.0, 0.0),      # 绿
        (0.0, 0.0, 1.0),      # 蓝
        (1.0, 1.0, 0.0),      # 黄
        (1.0, 0.0, 1.0),      # 品红
        (0.0, 1.0, 1.0),      # 青
        (1.0, 0.5, 0.0),      # 橙
        (0.5, 0.0, 1.0),      # 紫
        (0.0, 0.5, 1.0),      # 天蓝
        (0.5, 1.0, 0.0),      #  lime
        (1.0, 0.0, 0.5),      # 玫瑰
        (0.0, 1.0, 0.5),      # 春绿
        (0.5, 0.0, 0.0),      # 暗红
        (0.0, 0.5, 0.0),      # 暗绿
        (0.0, 0.0, 0.5),      # 暗蓝
        (0.5, 0.5, 0.0),      # 橄榄
        (0.5, 0.0, 0.5),      # 暗紫
        (0.0, 0.5, 0.5),      #  teal
        (0.7, 0.7, 0.7),      # 灰
        (0.3, 0.3, 0.3),      # 深灰
    ]
    colors = []
    for i in range(max(n, 1)):
        r, g, b = cmap[i % len(cmap)]
        colors.append(f"{r:.6f} {g:.6f} {b:.6f}")
    return colors

SLICER_COLORS = generate_slicer_colors(20)


# ============ 辅助函数 ============
def voxels_to_volume_um3(voxel_count):
    return voxel_count * VOXEL_VOLUME_UM3


def voxels_to_volume_mm3(voxel_count):
    return voxel_count * VOXEL_VOLUME_UM3 * UM3_TO_MM3


def pixels_to_um(pixels, axis='x'):
    if axis.lower() == 'x':
        return pixels * VOXEL_SIZE_X
    elif axis.lower() == 'y':
        return pixels * VOXEL_SIZE_Y
    elif axis.lower() == 'z':
        return pixels * VOXEL_SIZE_Z
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


# ============ Well 分组映射 ============
# 每个 well 前缀（不含 _1 后缀）对应浓度组
WELL_GROUPS = {
    # Control
    'F2': 'Control', 'F3': 'Control', 'F4': 'Control', 'F5': 'Control',
    'F6': 'Control', 'F7': 'Control', 'F8': 'Control', 'F9': 'Control',
    'F10': 'Control', 'F11': 'Control', 'B11': 'Control', 'C11': 'Control',
    'D11': 'Control', 'E11': 'Control',
    # 20 uM
    'B2': '20uM', 'B3': '20uM', 'B4': '20uM',
    'C2': '20uM', 'C3': '20uM', 'C4': '20uM',
    # 40 uM
    'B5': '40uM', 'B6': '40uM', 'B7': '40uM',
    'C5': '40uM', 'C6': '40uM', 'C7': '40uM',
    # 80 uM
    'B8': '80uM', 'B9': '80uM', 'B10': '80uM',
    'C8': '80uM', 'C9': '80uM', 'C10': '80uM',
}

GROUP_COLORS = {
    'Control': '#3498db',  # 蓝
    '20uM': '#2ecc71',     # 绿
    '40uM': '#f39c12',     # 橙
    '80uM': '#e74c3c',     # 红
}


# ============ 多特征匹配权重 ============
# 加权距离公式: D_total = w1*D_centroid + w2*D_volume + w3*D_sphericity
MF_WEIGHT_CENTROID = 1.0    # 质心距离权重 (pixels)
MF_WEIGHT_VOLUME = 50.0     # 体积差异权重 (归一化)
MF_WEIGHT_SPHERICITY = 30.0 # 球形度差异权重

# 多特征匹配最大距离阈值（按组设置）
# Control 组变化小，阈值小；加药组变化大，阈值大
MF_MAX_DISTANCE_BY_GROUP = {
    'Control': 150,
    '20uM': 200,
    '40uM': 250,
    '80uM': 300,
}

# 默认阈值
MF_MAX_DISTANCE = 150  # pixels


# ============ 配准参数 ============
# 使用仿射配准（affine），比 rigid 灵活，比 bspline 稳定
AFFINE_MAX_ITERATIONS = 1000


def print_config():
    print("=" * 60)
    print("FXN Organoid Tracking Configuration")
    print("=" * 60)
    print("\n[Image Physical Size]")
    print(f"  Voxel size: {VOXEL_SIZE_X} x {VOXEL_SIZE_Y} x {VOXEL_SIZE_Z} um")
    print(f"  Single voxel volume: {VOXEL_VOLUME_UM3:.2f} um^3")
    print(f"  Reference image size: {IMAGE_SIZE_X} x {IMAGE_SIZE_Y} x {IMAGE_SIZE_Z} pixels")
    print("\n[Tracking Parameters]")
    print(f"  Max match distance: {MAX_DISTANCE_PIXELS} pixels = {MAX_DISTANCE_PIXELS * VOXEL_SIZE_X:.1f} um")
    print(f"  Min connected component size: {MIN_VOXEL_COUNT} voxels")
    print("\n[Response Thresholds]")
    print(f"  CR: shrink > {abs(THRESHOLD_COMPLETE_RESPONSE) * 100:.0f}%")
    print(f"  PR: shrink > {abs(THRESHOLD_PARTIAL_RESPONSE) * 100:.0f}%")
    print(f"  SD: change {THRESHOLD_STABLE_DISEASE_MIN * 100:.0f}% ~ {THRESHOLD_STABLE_DISEASE_MAX * 100:.0f}%")
    print(f"  PD: grow > {THRESHOLD_PROGRESSIVE_DISEASE * 100:.0f}%")
    print("\n[Data Paths]")
    print(f"  Day3 seg: {FXN_0701_SEG_DIR}")
    print(f"  Day5 seg: {FXN_0703_SEG_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)
