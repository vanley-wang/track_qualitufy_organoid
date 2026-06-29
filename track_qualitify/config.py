"""
配置文件 - 存储类器官分析的全局参数
"""

# ============ 图像物理尺寸配置 ============
# 体素大小 (微米 μm)
VOXEL_SIZE_X = 5.0  # μm
VOXEL_SIZE_Y = 5.0  # μm
VOXEL_SIZE_Z = 5.5  # μm

# 图像尺寸 (像素)
IMAGE_SIZE_X = 800  # pixels
IMAGE_SIZE_Y = 800  # pixels
IMAGE_SIZE_Z = 512  # pixels

# 计算单个体素的实际体积 (立方微米)
VOXEL_VOLUME_UM3 = VOXEL_SIZE_X * VOXEL_SIZE_Y * VOXEL_SIZE_Z  # 137.5 μm³

# 单位转换常数
UM3_TO_MM3 = 1e-9  # 1 μm³ = 1e-9 mm³
UM2_TO_MM2 = 1e-6  # 1 μm² = 1e-6 mm²

# 实际视野大小 (微米)
FIELD_OF_VIEW_X = IMAGE_SIZE_X * VOXEL_SIZE_X  # 4000 μm = 4 mm
FIELD_OF_VIEW_Y = IMAGE_SIZE_Y * VOXEL_SIZE_Y  # 4000 μm = 4 mm
FIELD_OF_VIEW_Z = IMAGE_SIZE_Z * VOXEL_SIZE_Z  # 2816 μm = 2.816 mm


# ============ ID追踪配置 ============
# 最大匹配距离 (像素)
MAX_DISTANCE_PIXELS = 50  # 对应 250 μm (50 * 5 μm)

# 转换为实际物理距离 (微米)
MAX_DISTANCE_UM = MAX_DISTANCE_PIXELS * VOXEL_SIZE_X  # 250 μm


# ============ 量化参数阈值 ============
# 最小类器官体积 (立方微米),过滤噪声
MIN_ORGANOID_VOLUME_UM3 = 1000  # 1000 μm³

# 体积变化率阈值 (用于药效评估)
THRESHOLD_COMPLETE_RESPONSE = -0.9  # 缩小90%以上视为完全缓解
THRESHOLD_PARTIAL_RESPONSE = -0.5   # 缩小50%以上视为部分缓解
THRESHOLD_STABLE_DISEASE_MIN = -0.3  # -30%到+30%为疾病稳定
THRESHOLD_STABLE_DISEASE_MAX = 0.3
THRESHOLD_PROGRESSIVE_DISEASE = 0.3  # 增长30%以上为疾病进展


# ============ 文件路径配置 ============
# 数据目录
DATA_DIR = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/ID_organoid"
RESULTS_DIR = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/results"

# 三天的标签文件 (ID匹配后的，颜色和位置都已修复)
DAY1_LABEL = "../results/day1_reference.nrrd"  # Day1作为基准
DAY2_LABEL = "../results/day2_matched_fixed_v2.nrrd"  # Day2匹配后（150像素阈值）
DAY3_LABEL = "../results/day3_matched_fixed.nrrd"  # Day3匹配后

# 三天的原始成像文件 (用于提取灰度值)
DAY1_IMAGE = "organoid_031_0000.nii.gz"
DAY2_IMAGE = "organoid_043_0000.nii.gz"
DAY3_IMAGE = "organoid_062_0000.nii.gz"

# 输出目录
OUTPUT_DIR = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/results"

# 输出文件名
OUTPUT_QUANTIFICATION = "organoid_quantification.csv"  # 单时间点量化结果
OUTPUT_LONGITUDINAL = "organoid_longitudinal_analysis.csv"  # 纵向分析结果
OUTPUT_SUMMARY = "organoid_summary_statistics.csv"  # 汇总统计


# ============ 可视化配置 ============
# 图表样式
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

# 颜色配置
COLOR_DAY1 = '#3498db'  # 蓝色
COLOR_DAY2 = '#e74c3c'  # 红色
COLOR_DAY3 = '#2ecc71'  # 绿色

# 药效评估颜色
COLOR_CR = '#27ae60'   # 完全缓解 - 深绿色
COLOR_PR = '#95a5a6'   # 部分缓解 - 灰绿色
COLOR_SD = '#f39c12'   # 疾病稳定 - 橙色
COLOR_PD = '#c0392b'   # 疾病进展 - 深红色


# ============ 辅助函数 ============
def pixels_to_um(pixels, axis='x'):
    """
    将像素距离转换为物理距离 (微米)
    
    参数:
        pixels: 像素数量
        axis: 'x', 'y', 或 'z'
    
    返回:
        物理距离 (微米)
    """
    if axis.lower() == 'x':
        return pixels * VOXEL_SIZE_X
    elif axis.lower() == 'y':
        return pixels * VOXEL_SIZE_Y
    elif axis.lower() == 'z':
        return pixels * VOXEL_SIZE_Z
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def voxels_to_volume_um3(voxel_count):
    """
    将体素数量转换为实际体积 (立方微米)
    
    参数:
        voxel_count: 体素数量
    
    返回:
        体积 (μm³)
    """
    return voxel_count * VOXEL_VOLUME_UM3


def voxels_to_volume_mm3(voxel_count):
    """
    将体素数量转换为实际体积 (立方毫米)
    
    参数:
        voxel_count: 体素数量
    
    返回:
        体积 (mm³)
    """
    return voxel_count * VOXEL_VOLUME_UM3 * UM3_TO_MM3


def print_config():
    """打印当前配置信息"""
    print("="*60)
    print("类器官量化分析 - 配置信息")
    print("="*60)
    print(f"\n【图像物理尺寸】")
    print(f"  体素大小: {VOXEL_SIZE_X} × {VOXEL_SIZE_Y} × {VOXEL_SIZE_Z} μm")
    print(f"  单体素体积: {VOXEL_VOLUME_UM3:.2f} μm³")
    print(f"  图像尺寸: {IMAGE_SIZE_X} × {IMAGE_SIZE_Y} × {IMAGE_SIZE_Z} pixels")
    print(f"  实际视野: {FIELD_OF_VIEW_X} × {FIELD_OF_VIEW_Y} × {FIELD_OF_VIEW_Z} μm")
    print(f"            = {FIELD_OF_VIEW_X/1000:.2f} × {FIELD_OF_VIEW_Y/1000:.2f} × {FIELD_OF_VIEW_Z/1000:.2f} mm")
    
    print(f"\n【ID追踪参数】")
    print(f"  最大匹配距离: {MAX_DISTANCE_PIXELS} pixels = {MAX_DISTANCE_UM} μm")
    
    print(f"\n【药效评估阈值】")
    print(f"  完全缓解 (CR): 体积缩小 > {abs(THRESHOLD_COMPLETE_RESPONSE)*100:.0f}%")
    print(f"  部分缓解 (PR): 体积缩小 > {abs(THRESHOLD_PARTIAL_RESPONSE)*100:.0f}%")
    print(f"  疾病稳定 (SD): 体积变化 {THRESHOLD_STABLE_DISEASE_MIN*100:.0f}% ~ {THRESHOLD_STABLE_DISEASE_MAX*100:.0f}%")
    print(f"  疾病进展 (PD): 体积增长 > {THRESHOLD_PROGRESSIVE_DISEASE*100:.0f}%")
    
    print(f"\n【数据路径】")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    print_config()
