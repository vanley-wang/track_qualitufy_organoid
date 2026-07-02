"""
类器官量化分析主脚本
提取每个类器官的完整形态学和密度参数
支持三天数据的批量处理
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
from skimage.measure import regionprops, marching_cubes
from scipy.ndimage import center_of_mass
import warnings
warnings.filterwarnings('ignore')

# 导入配置
import config


def load_nifti_or_nrrd(file_path):
    """
    加载NIfTI或NRRD格式的文件
    
    参数:
        file_path: 文件路径
    
    返回:
        data: numpy数组
        affine: 仿射矩阵 (如果是NIfTI)
    """
    if file_path.endswith('.nrrd'):
        data, header = nrrd.read(file_path)
        return data.astype(int), None
    else:
        nii = nib.load(file_path)
        return nii.get_fdata().astype(int), nii.affine


def calculate_sphericity(volume_um3, surface_area_um2):
    """
    计算球形度 (Sphericity)
    球形度 = π^(1/3) * (6*V)^(2/3) / A
    完美球体的球形度 = 1
    
    参数:
        volume_um3: 体积 (μm³)
        surface_area_um2: 表面积 (μm²)
    
    返回:
        sphericity: 球形度 (0-1)
    """
    if surface_area_um2 == 0:
        return 0
    
    sphericity = (np.pi ** (1/3)) * ((6 * volume_um3) ** (2/3)) / surface_area_um2
    return min(sphericity, 1.0)  # 理论最大值为1


def calculate_surface_area(label_data, label_id, spacing):
    """
    使用Marching Cubes算法计算表面积
    
    参数:
        label_data: 标签数组
        label_id: 当前类器官的ID
        spacing: 体素间距 (z, y, x) 单位:微米
    
    返回:
        surface_area: 表面积 (μm²)
    """
    try:
        # 创建二值mask
        mask = (label_data == label_id).astype(np.uint8)
        
        # 使用Marching Cubes提取表面
        verts, faces, normals, values = marching_cubes(
            mask, 
            level=0.5,
            spacing=spacing  # (z, y, x)
        )
        
        # 计算每个三角面的面积
        # faces是 (N, 3) 数组,每行是三角形的三个顶点索引
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # 向量叉乘计算面积
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        
        total_surface_area = np.sum(areas)
        return total_surface_area
        
    except Exception as e:
        print(f"      警告: 计算表面积失败 (ID={label_id}): {e}")
        return 0


def quantify_single_organoid(region, image_data, day_name, spacing_um):
    """
    量化单个类器官的所有参数
    
    参数:
        region: skimage.measure.regionprops 对象
        image_data: 原始成像数据 (用于提取灰度值)
        day_name: 'Day1', 'Day2', 'Day3'
        spacing_um: 体素间距 (z, y, x) 单位:微米
    
    返回:
        dict: 包含所有量化参数
    """
    organoid_id = region.label
    
    # ===== 基础参数 =====
    voxel_count = region.area  # 体素数量
    
    # ===== 体积参数 =====
    volume_um3 = config.voxels_to_volume_um3(voxel_count)
    volume_mm3 = config.voxels_to_volume_mm3(voxel_count)
    
    # ===== 质心坐标 (像素) =====
    centroid_z, centroid_y, centroid_x = region.centroid
    
    # ===== 质心坐标 (物理坐标, 微米) =====
    centroid_z_um = centroid_z * config.VOXEL_SIZE_Z
    centroid_y_um = centroid_y * config.VOXEL_SIZE_Y
    centroid_x_um = centroid_x * config.VOXEL_SIZE_X
    
    # ===== 边界框 =====
    bbox = region.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
    bbox_size_z = (bbox[3] - bbox[0]) * config.VOXEL_SIZE_Z
    bbox_size_y = (bbox[4] - bbox[1]) * config.VOXEL_SIZE_Y
    bbox_size_x = (bbox[5] - bbox[2]) * config.VOXEL_SIZE_X
    
    # ===== 等效直径 =====
    # 假设类器官是球体,计算等效直径
    equivalent_diameter_um = 2 * (3 * volume_um3 / (4 * np.pi)) ** (1/3)
    
    # ===== 表面积 (需要完整的label data) =====
    # 这里先用regionprops的估算,后面会重新计算
    # regionprops没有直接的表面积,需要从完整数据计算
    
    # ===== 长短轴 =====
    # 使用region的major_axis_length和minor_axis_length (像素单位)
    # 需要转换为物理单位
    # 注意: regionprops返回的是2D投影的轴长,对3D数据需要特殊处理
    # 这里使用bbox作为近似
    axes_lengths_um = sorted([bbox_size_x, bbox_size_y, bbox_size_z], reverse=True)
    major_axis_um = axes_lengths_um[0]
    intermediate_axis_um = axes_lengths_um[1]
    minor_axis_um = axes_lengths_um[2]
    
    aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else 0
    
    # ===== 紧凑度 (Compactness) =====
    # 完美球体的紧凑度最大
    # 这里先占位,等计算表面积后更新
    
    # ===== 灰度值统计 (从原始成像) =====
    if image_data is not None:
        # 提取该类器官区域的灰度值
        mask = region.image  # 二值mask (只包含当前类器官的bbox区域)
        intensity_values = image_data[region.slice][mask]
        
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)
        min_intensity = np.min(intensity_values)
        max_intensity = np.max(intensity_values)
        median_intensity = np.median(intensity_values)
        
        # 灰度值的变异系数 (Coefficient of Variation)
        cv_intensity = std_intensity / mean_intensity if mean_intensity > 0 else 0
    else:
        mean_intensity = std_intensity = min_intensity = max_intensity = median_intensity = cv_intensity = np.nan
    
    # ===== 整理结果 =====
    result = {
        'Organoid_ID': organoid_id,
        'Day': day_name,
        
        # 体积参数
        'Voxel_Count': voxel_count,
        'Volume_um3': volume_um3,
        'Volume_mm3': volume_mm3,
        'Equivalent_Diameter_um': equivalent_diameter_um,
        
        # 形态参数
        'Major_Axis_um': major_axis_um,
        'Intermediate_Axis_um': intermediate_axis_um,
        'Minor_Axis_um': minor_axis_um,
        'Aspect_Ratio': aspect_ratio,
        
        # 质心 (像素)
        'Centroid_X_pixel': centroid_x,
        'Centroid_Y_pixel': centroid_y,
        'Centroid_Z_pixel': centroid_z,
        
        # 质心 (物理坐标)
        'Centroid_X_um': centroid_x_um,
        'Centroid_Y_um': centroid_y_um,
        'Centroid_Z_um': centroid_z_um,
        
        # 边界框
        'BBox_Size_X_um': bbox_size_x,
        'BBox_Size_Y_um': bbox_size_y,
        'BBox_Size_Z_um': bbox_size_z,
        
        # 灰度值
        'Mean_Intensity': mean_intensity,
        'Std_Intensity': std_intensity,
        'Min_Intensity': min_intensity,
        'Max_Intensity': max_intensity,
        'Median_Intensity': median_intensity,
        'CV_Intensity': cv_intensity,
        
        # 表面积和球形度 (稍后计算)
        'Surface_Area_um2': np.nan,
        'Sphericity': np.nan,
        'Compactness': np.nan,
    }
    
    return result


def quantify_one_timepoint(label_path, image_path, day_name, calculate_surface=True):
    """
    量化一个时间点的所有类器官
    
    参数:
        label_path: 标签文件路径
        image_path: 原始成像文件路径
        day_name: 'Day1', 'Day2', 'Day3'
        calculate_surface: 是否计算表面积 (耗时较长)
    
    返回:
        DataFrame: 包含所有类器官的量化结果
    """
    print(f"\n{'='*60}")
    print(f"正在量化 {day_name}...")
    print(f"{'='*60}")
    
    # 加载标签数据
    print(f"  [1/4] 加载标签文件: {os.path.basename(label_path)}")
    label_data, _ = load_nifti_or_nrrd(label_path)
    
    # 加载原始成像数据
    if image_path and os.path.exists(image_path):
        print(f"  [2/4] 加载成像文件: {os.path.basename(image_path)}")
        image_data, _ = load_nifti_or_nrrd(image_path)
    else:
        print(f"  [2/4] 警告: 未找到成像文件,将跳过灰度值分析")
        image_data = None
    
    # 提取区域属性
    print(f"  [3/4] 提取区域属性...")
    regions = regionprops(label_data)
    print(f"        检测到 {len(regions)} 个类器官")
    
    # 定义体素间距 (z, y, x)
    spacing_um = (config.VOXEL_SIZE_Z, config.VOXEL_SIZE_Y, config.VOXEL_SIZE_X)
    
    # 量化每个类器官
    print(f"  [4/4] 量化参数...")
    results = []
    
    for i, region in enumerate(regions, 1):
        # 基础量化
        result = quantify_single_organoid(region, image_data, day_name, spacing_um)
        
        # 计算表面积 (可选,较耗时)
        if calculate_surface:
            if i <= 5 or i % 10 == 0:  # 只打印部分进度
                print(f"        处理 ID {region.label} ({i}/{len(regions)})")
            
            surface_area = calculate_surface_area(label_data, region.label, spacing_um)
            result['Surface_Area_um2'] = surface_area
            
            # 更新球形度
            if surface_area > 0:
                result['Sphericity'] = calculate_sphericity(result['Volume_um3'], surface_area)
                # 紧凑度
                result['Compactness'] = (result['Volume_um3'] ** 2) / (surface_area ** 3)
        
        results.append(result)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 过滤小体积噪声
    original_count = len(df)
    df = df[df['Volume_um3'] >= config.MIN_ORGANOID_VOLUME_UM3].copy()
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        print(f"\n  ⚠️  过滤掉 {filtered_count} 个小于 {config.MIN_ORGANOID_VOLUME_UM3} μm³ 的小对象")
    
    print(f"\n  ✓ {day_name} 量化完成! 有效类器官数量: {len(df)}")
    print(f"    平均体积: {df['Volume_mm3'].mean():.4f} mm³")
    print(f"    体积范围: {df['Volume_mm3'].min():.4f} - {df['Volume_mm3'].max():.4f} mm³")
    
    return df


def quantify_all_timepoints(output_dir=None, calculate_surface=True):
    """
    量化所有三天的数据
    
    参数:
        output_dir: 输出目录
        calculate_surface: 是否计算表面积
    
    返回:
        DataFrame: 合并的所有时间点数据
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("类器官量化分析 - 批量处理")
    print("="*60)
    config.print_config()
    
    # 定义三天的数据路径
    timepoints = [
        {
            'day': 'Day1',
            'label': os.path.join(config.DATA_DIR, config.DAY1_LABEL),
            'image': os.path.join(config.DATA_DIR, config.DAY1_IMAGE),
        },
        {
            'day': 'Day2',
            'label': os.path.join(config.DATA_DIR, config.DAY2_LABEL),
            'image': os.path.join(config.DATA_DIR, config.DAY2_IMAGE),
        },
        {
            'day': 'Day3',
            'label': os.path.join(config.DATA_DIR, config.DAY3_LABEL),
            'image': os.path.join(config.DATA_DIR, config.DAY3_IMAGE),
        },
    ]
    
    # 检查文件是否存在
    all_exist = True
    for tp in timepoints:
        if not os.path.exists(tp['label']):
            print(f"❌ 错误: 未找到标签文件 {tp['label']}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  请先运行 track_3days.py 生成追踪后的标签文件!")
        return None
    
    # 量化每个时间点
    all_results = []
    for tp in timepoints:
        df = quantify_one_timepoint(
            label_path=tp['label'],
            image_path=tp['image'],
            day_name=tp['day'],
            calculate_surface=calculate_surface
        )
        all_results.append(df)
    
    # 合并所有结果
    df_all = pd.concat(all_results, ignore_index=True)
    
    # 保存结果
    output_path = os.path.join(output_dir, config.OUTPUT_QUANTIFICATION)
    df_all.to_csv(output_path, index=False, float_format='%.6f')
    
    print("\n" + "="*60)
    print("量化完成!")
    print("="*60)
    print(f"总类器官数 (含三天): {len(df_all)}")
    print(f"结果已保存至: {output_path}")
    print("="*60)
    
    return df_all


if __name__ == "__main__":
    # 运行完整量化
    df = quantify_all_timepoints(calculate_surface=True)
    
    if df is not None:
        # 显示示例数据
        print("\n示例数据 (前5行):")
        print(df.head())
        
        # 按天统计
        print("\n按天统计:")
        print(df.groupby('Day')['Volume_mm3'].describe())
