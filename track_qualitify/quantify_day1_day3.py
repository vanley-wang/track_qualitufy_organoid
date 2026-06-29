"""
量化分析 - Day1 和 Day3
提取类器官的形态学和密度参数
"""

import numpy as np
import nrrd
import nibabel as nib
from skimage.measure import regionprops, marching_cubes
from scipy.spatial.distance import cdist
import pandas as pd
import os
import sys

# 添加配置文件路径
sys.path.append(os.path.dirname(__file__))
from config import *


def calculate_surface_area(mask):
    """
    使用marching cubes算法计算表面积
    """
    try:
        verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=(VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z))
        
        # 计算每个三角形的面积
        triangles = verts[faces]
        v0 = triangles[:, 0, :]
        v1 = triangles[:, 1, :]
        v2 = triangles[:, 2, :]
        
        # 使用叉积计算面积
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        total_area = np.sum(areas)
        
        return total_area
    except:
        return np.nan


def quantify_single_organoid(label_data, image_data, label_value, day_name):
    """
    量化单个类器官的所有参数
    
    参数:
        label_data: 标签数据 (分割mask)
        image_data: 原始成像数据
        label_value: 该类器官的标签值
        day_name: 时间点名称 (Day1, Day2, Day3)
    
    返回:
        包含所有量化参数的字典
    """
    # 创建该类器官的mask
    mask = (label_data == label_value)
    
    if not np.any(mask):
        return None
    
    # 使用regionprops提取基本参数
    props = regionprops(mask.astype(int), intensity_image=image_data)[0]
    
    # ===== 体积相关 =====
    volume_voxels = props.area  # 体素数量
    volume_um3 = voxels_to_volume_um3(volume_voxels)  # 微米³
    volume_mm3 = volume_um3 * UM3_TO_MM3  # 毫米³
    
    # ===== 形态参数 =====
    # 球形度 (使用等效球体)
    equivalent_diameter_um = props.equivalent_diameter * VOXEL_SIZE_X
    
    # 表面积
    surface_area_um2 = calculate_surface_area(mask)
    surface_area_mm2 = surface_area_um2 * UM2_TO_MM2 if not np.isnan(surface_area_um2) else np.nan
    
    # 球形度 (越接近1越圆)
    if not np.isnan(surface_area_um2) and surface_area_um2 > 0:
        ideal_surface = 4 * np.pi * (3 * volume_um3 / (4 * np.pi)) ** (2/3)
        sphericity = ideal_surface / surface_area_um2
    else:
        sphericity = np.nan
    
    # 长短轴
    major_axis_um = props.major_axis_length * VOXEL_SIZE_X
    minor_axis_um = props.minor_axis_length * VOXEL_SIZE_X
    axis_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else np.nan
    
    # 延展度和实度
    extent = props.extent  # 占边界框的比例
    solidity = props.solidity  # 占凸包的比例
    
    # ===== 质心坐标 =====
    centroid_pixels = props.centroid
    centroid_um = (
        centroid_pixels[0] * VOXEL_SIZE_X,
        centroid_pixels[1] * VOXEL_SIZE_Y,
        centroid_pixels[2] * VOXEL_SIZE_Z
    )
    
    # ===== 灰度值统计 =====
    intensity_values = image_data[mask]
    mean_intensity = props.mean_intensity
    max_intensity = np.max(intensity_values)
    min_intensity = np.min(intensity_values)
    std_intensity = np.std(intensity_values)
    median_intensity = np.median(intensity_values)
    
    # 灰度值分位数
    intensity_q25 = np.percentile(intensity_values, 25)
    intensity_q75 = np.percentile(intensity_values, 75)
    
    # ===== 边界框 =====
    bbox = props.bbox
    bbox_volume_um3 = (
        (bbox[3] - bbox[0]) * VOXEL_SIZE_X *
        (bbox[4] - bbox[1]) * VOXEL_SIZE_Y *
        (bbox[5] - bbox[2]) * VOXEL_SIZE_Z
    )
    
    # 返回结果
    result = {
        'Organoid_ID': label_value,
        'Day': day_name,
        
        # 体积
        'Volume_voxels': volume_voxels,
        'Volume_um3': volume_um3,
        'Volume_mm3': volume_mm3,
        
        # 形态
        'Surface_Area_um2': surface_area_um2,
        'Surface_Area_mm2': surface_area_mm2,
        'Sphericity': sphericity,
        'Equivalent_Diameter_um': equivalent_diameter_um,
        'Major_Axis_um': major_axis_um,
        'Minor_Axis_um': minor_axis_um,
        'Axis_Ratio': axis_ratio,
        'Extent': extent,
        'Solidity': solidity,
        
        # 质心
        'Centroid_X_pixels': centroid_pixels[0],
        'Centroid_Y_pixels': centroid_pixels[1],
        'Centroid_Z_pixels': centroid_pixels[2],
        'Centroid_X_um': centroid_um[0],
        'Centroid_Y_um': centroid_um[1],
        'Centroid_Z_um': centroid_um[2],
        
        # 灰度值
        'Mean_Intensity': mean_intensity,
        'Median_Intensity': median_intensity,
        'Std_Intensity': std_intensity,
        'Min_Intensity': min_intensity,
        'Max_Intensity': max_intensity,
        'Intensity_Q25': intensity_q25,
        'Intensity_Q75': intensity_q75,
        
        # 边界框
        'BBox_Volume_um3': bbox_volume_um3,
    }
    
    return result


def main():
    print("="*70)
    print("类器官量化分析 - Day1 和 Day3")
    print("="*70)
    
    # 读取Day1数据
    print("\n读取Day1数据...")
    label1, _ = nrrd.read(os.path.join(DATA_DIR, DAY1_LABEL))
    image1 = nib.load(os.path.join(DATA_DIR, DAY1_IMAGE)).get_fdata()
    print(f"  ✓ Day1 标签: {label1.shape}")
    print(f"  ✓ Day1 成像: {image1.shape}")
    
    # 读取Day3数据
    print("\n读取Day3数据...")
    label3, _ = nrrd.read(os.path.join(DATA_DIR, DAY3_LABEL))
    image3 = nib.load(os.path.join(DATA_DIR, DAY3_IMAGE)).get_fdata()
    print(f"  ✓ Day3 标签: {label3.shape}")
    print(f"  ✓ Day3 成像: {image3.shape}")
    
    # 获取类器官ID
    labels1 = sorted(np.unique(label1)[1:])  # 排除背景0
    labels3 = sorted(np.unique(label3)[1:])
    
    print(f"\n类器官数量:")
    print(f"  Day1: {len(labels1)} 个")
    print(f"  Day3: {len(labels3)} 个")
    
    # 找到共同的ID
    common_ids = sorted(set(labels1) & set(labels3))
    print(f"  共同追踪: {len(common_ids)} 个")
    
    # 量化所有类器官
    results = []
    
    print("\n" + "="*70)
    print("量化Day1类器官...")
    print("="*70)
    for i, label_id in enumerate(labels1, 1):
        print(f"  [{i}/{len(labels1)}] 量化 Organoid ID={label_id}...", end='')
        result = quantify_single_organoid(label1, image1, label_id, 'Day1')
        if result:
            results.append(result)
            print(f" ✓ (体积: {result['Volume_mm3']:.3f} mm³)")
        else:
            print(" ✗ 失败")
    
    print("\n" + "="*70)
    print("量化Day3类器官...")
    print("="*70)
    for i, label_id in enumerate(labels3, 1):
        print(f"  [{i}/{len(labels3)}] 量化 Organoid ID={label_id}...", end='')
        result = quantify_single_organoid(label3, image3, label_id, 'Day3')
        if result:
            results.append(result)
            print(f" ✓ (体积: {result['Volume_mm3']:.3f} mm³)")
        else:
            print(" ✗ 失败")
    
    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'organoid_quantification_day1_day3.csv')
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("量化完成！")
    print("="*70)
    print(f"总共量化: {len(results)} 个类器官 (Day1: {len(labels1)}, Day3: {len(labels3)})")
    print(f"结果已保存: {output_path}")
    
    # 显示统计摘要
    print("\n" + "="*70)
    print("统计摘要")
    print("="*70)
    
    df_day1 = df[df['Day'] == 'Day1']
    df_day3 = df[df['Day'] == 'Day3']
    
    print("\nDay1:")
    print(f"  体积 (mm³): {df_day1['Volume_mm3'].mean():.3f} ± {df_day1['Volume_mm3'].std():.3f}")
    print(f"  球形度: {df_day1['Sphericity'].mean():.3f} ± {df_day1['Sphericity'].std():.3f}")
    print(f"  平均灰度: {df_day1['Mean_Intensity'].mean():.1f} ± {df_day1['Mean_Intensity'].std():.1f}")
    
    print("\nDay3:")
    print(f"  体积 (mm³): {df_day3['Volume_mm3'].mean():.3f} ± {df_day3['Volume_mm3'].std():.3f}")
    print(f"  球形度: {df_day3['Sphericity'].mean():.3f} ± {df_day3['Sphericity'].std():.3f}")
    print(f"  平均灰度: {df_day3['Mean_Intensity'].mean():.1f} ± {df_day3['Mean_Intensity'].std():.1f}")
    
    print("\n" + "="*70)
    

if __name__ == "__main__":
    main()
