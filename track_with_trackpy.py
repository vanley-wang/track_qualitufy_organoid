"""
使用Trackpy进行改进的类器官追踪
支持运动预测和多特征匹配
"""

import numpy as np
import pandas as pd
import nrrd
from skimage.measure import regionprops
import trackpy as tp
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_features_from_labels(label_data, image_data):
    """
    从标签数据中提取特征用于追踪
    
    返回:
        DataFrame with columns: [label, x, y, z, volume, sphericity, mean_intensity]
    """
    props = regionprops(label_data, intensity_image=image_data)
    
    features = []
    for prop in props:
        centroid = prop.centroid
        
        # 计算球形度
        volume = prop.area
        equivalent_diameter = prop.equivalent_diameter
        if volume > 0:
            ideal_surface = 4 * np.pi * (3 * volume / (4 * np.pi)) ** (2/3)
            # 简化的球形度估计
            sphericity = (volume ** (2/3)) / (prop.bbox[3] - prop.bbox[0] + 1)
        else:
            sphericity = 0
        
        features.append({
            'label': prop.label,
            'x': centroid[0],
            'y': centroid[1],
            'z': centroid[2],
            'volume': volume,
            'sphericity': sphericity,
            'mean_intensity': prop.mean_intensity
        })
    
    return pd.DataFrame(features)


def prepare_tracking_dataframe(features_day1, features_day2, features_day3):
    """
    准备trackpy所需的DataFrame格式
    
    Trackpy需要: frame, x, y, z, [其他特征]
    """
    # 添加frame列
    features_day1['frame'] = 0
    features_day2['frame'] = 1
    features_day3['frame'] = 2
    
    # 合并所有时间点
    df = pd.concat([features_day1, features_day2, features_day3], ignore_index=True)
    
    return df


def track_with_trackpy(df, search_range=100, memory=0, adaptive_stop=None):
    """
    使用trackpy进行追踪
    
    参数:
        df: DataFrame with [frame, x, y, z, ...]
        search_range: 最大搜索距离(像素)
        memory: 允许目标消失的帧数
        adaptive_stop: 自适应停止阈值
    
    返回:
        tracked_df: 带有particle列的DataFrame
    """
    print(f"\n使用Trackpy追踪...")
    print(f"  搜索范围: {search_range} 像素")
    print(f"  Memory: {memory} 帧")
    
    # Trackpy追踪
    tracked = tp.link(
        df,
        search_range=search_range,
        memory=memory,
        adaptive_stop=adaptive_stop,
        pos_columns=['x', 'y', 'z']
    )
    
    return tracked


def create_label_mapping_from_tracking(tracked_df):
    """
    从trackpy的追踪结果创建标签映射
    
    返回:
        day2_mapping: {original_label: new_label}
        day3_mapping: {original_label: new_label}
    """
    day2_mapping = {}
    day3_mapping = {}
    
    # Day1作为参考
    day1_data = tracked_df[tracked_df['frame'] == 0]
    
    # 为每个track(particle)找到对应关系
    for particle_id in tracked_df['particle'].unique():
        track = tracked_df[tracked_df['particle'] == particle_id].sort_values('frame')
        
        # 找到Day1的标签(如果存在)
        day1_in_track = track[track['frame'] == 0]
        if len(day1_in_track) > 0:
            day1_label = int(day1_in_track.iloc[0]['label'])
            
            # Day2映射
            day2_in_track = track[track['frame'] == 1]
            if len(day2_in_track) > 0:
                day2_original = int(day2_in_track.iloc[0]['label'])
                day2_mapping[day2_original] = day1_label
            
            # Day3映射
            day3_in_track = track[track['frame'] == 2]
            if len(day3_in_track) > 0:
                day3_original = int(day3_in_track.iloc[0]['label'])
                day3_mapping[day3_original] = day1_label
    
    return day2_mapping, day3_mapping


def apply_label_mapping(original_data, label_mapping, handle_unmapped='special_id'):
    """
    应用标签映射到原始数据
    
    参数:
        original_data: 原始标签数组
        label_mapping: {original_label: new_label}
        handle_unmapped: 如何处理未映射的标签
            - 'remove': 移除(设为0)
            - 'special_id': 分配特殊ID范围(从1000开始)
            - 'keep_original': 保留原始ID
    
    返回:
        mapped_data: 重新标记的数组
        unmapped_info: 未映射标签的信息字典
    """
    mapped_data = np.zeros_like(original_data)
    unmapped_info = {}
    
    # 应用映射
    for orig_label, new_label in label_mapping.items():
        mapped_data[original_data == orig_label] = new_label
    
    # 处理没有映射的标签
    unmapped_labels = np.unique(original_data)
    unmapped_labels = unmapped_labels[unmapped_labels > 0]
    unmapped_labels = [l for l in unmapped_labels if l not in label_mapping]
    
    if len(unmapped_labels) > 0:
        if handle_unmapped == 'remove':
            # 不处理,保持为0(背景)
            for orig_label in unmapped_labels:
                unmapped_info[orig_label] = 0
        elif handle_unmapped == 'special_id':
            # 分配特殊ID范围
            next_id = 1000
            for orig_label in unmapped_labels:
                mapped_data[original_data == orig_label] = next_id
                unmapped_info[orig_label] = next_id
                next_id += 1
        elif handle_unmapped == 'keep_original':
            # 保留原始ID
            for orig_label in unmapped_labels:
                mapped_data[original_data == orig_label] = orig_label
                unmapped_info[orig_label] = orig_label
    
    return mapped_data, unmapped_info


def save_tracking_results(mapped_data, output_path, reference_header, original_header):
    """
    保存追踪结果,重新生成正确的Segment信息包括Extent
    """
    from skimage.measure import regionprops
    
    # 创建输出header,保留原始的space origin等信息
    output_header = original_header.copy()
    
    # 删除原有的所有Segment信息
    keys_to_remove = [k for k in output_header.keys() if k.startswith('Segment')]
    for key in keys_to_remove:
        del output_header[key]
    
    # 获取实际存在的label
    unique_labels = np.unique(mapped_data)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景
    
    # 从reference_header获取颜色信息
    ref_segment_colors = {}
    if 'Segment0_ID' in reference_header:
        for key in reference_header.keys():
            if key.endswith('_ID'):
                seg_name = key.replace('_ID', '')
                seg_id = reference_header[key]
                color_key = f"{seg_name}_Color"
                if color_key in reference_header:
                    ref_segment_colors[seg_id] = reference_header[color_key]
    
    # 使用regionprops计算每个label的extent
    props = regionprops(mapped_data.astype(np.int32))
    label_extents = {}
    for prop in props:
        bbox = prop.bbox  # (min_row, min_col, min_depth, max_row, max_col, max_depth)
        # 转换为Slicer的extent格式: minX maxX minY maxY minZ maxZ
        extent = f"{bbox[0]} {bbox[3]-1} {bbox[1]} {bbox[4]-1} {bbox[2]} {bbox[5]-1}"
        label_extents[prop.label] = extent
    
    # 为每个实际存在的label创建Segment信息
    for idx, label_id in enumerate(sorted(unique_labels)):
        seg_prefix = f"Segment{idx}"
        
        # 设置ID(必须是Segment_数字格式,3D Slicer要求)
        seg_id_str = f"Segment_{label_id}"
        output_header[f"{seg_prefix}_ID"] = seg_id_str
        
        # 设置颜色(从reference获取,如果存在的话)
        if seg_id_str in ref_segment_colors:
            output_header[f"{seg_prefix}_Color"] = ref_segment_colors[seg_id_str]
        else:
            # 生成默认颜色
            import colorsys
            hue = (idx * 0.618033988749895) % 1.0  # 黄金分割角
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            output_header[f"{seg_prefix}_Color"] = f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}"
        
        # 设置名称
        output_header[f"{seg_prefix}_Name"] = f"Segment_{label_id}"
        
        # 其他必要属性
        output_header[f"{seg_prefix}_LabelValue"] = str(label_id)
        output_header[f"{seg_prefix}_Layer"] = "0"
        
        # 设置正确的Extent
        if label_id in label_extents:
            output_header[f"{seg_prefix}_Extent"] = label_extents[label_id]
        else:
            # 数据范围
            output_header[f"{seg_prefix}_Extent"] = f"0 {mapped_data.shape[0]-1} 0 {mapped_data.shape[1]-1} 0 {mapped_data.shape[2]-1}"
    
    # 保存
    nrrd.write(output_path, mapped_data.astype(np.int16), output_header)
    print(f"  ✓ 已保存: {output_path} (包含 {len(unique_labels)} 个segment)")


def visualize_trajectories(tracked_df, output_path):
    """
    可视化追踪轨迹
    """
    fig = plt.figure(figsize=(15, 5))
    
    # XY平面
    ax1 = fig.add_subplot(131)
    for particle in tracked_df['particle'].unique():
        track = tracked_df[tracked_df['particle'] == particle].sort_values('frame')
        ax1.plot(track['x'], track['y'], 'o-', alpha=0.5, markersize=4)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('XY Projection')
    ax1.grid(True, alpha=0.3)
    
    # XZ平面
    ax2 = fig.add_subplot(132)
    for particle in tracked_df['particle'].unique():
        track = tracked_df[tracked_df['particle'] == particle].sort_values('frame')
        ax2.plot(track['x'], track['z'], 'o-', alpha=0.5, markersize=4)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Z (pixels)')
    ax2.set_title('XZ Projection')
    ax2.grid(True, alpha=0.3)
    
    # 时间轴上的数量
    ax3 = fig.add_subplot(133)
    for frame in sorted(tracked_df['frame'].unique()):
        count = len(tracked_df[tracked_df['frame'] == frame]['particle'].unique())
        ax3.bar(frame, count, alpha=0.7)
    ax3.set_xlabel('Time Point (0=Day1, 1=Day2, 2=Day3)')
    ax3.set_ylabel('Number of Tracked Organoids')
    ax3.set_title('Organoid Count')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['Day1', 'Day2', 'Day3'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 轨迹图已保存: {output_path}")
    plt.close()


def generate_tracking_report(tracked_df, day2_mapping, day3_mapping):
    """
    生成追踪报告
    """
    print("\n" + "="*70)
    print("Trackpy追踪报告")
    print("="*70)
    
    # 统计信息
    total_tracks = tracked_df['particle'].nunique()
    day1_count = len(tracked_df[tracked_df['frame'] == 0])
    day2_count = len(tracked_df[tracked_df['frame'] == 1])
    day3_count = len(tracked_df[tracked_df['frame'] == 2])
    
    print(f"\n总追踪轨迹数: {total_tracks}")
    print(f"  Day1: {day1_count} 个类器官")
    print(f"  Day2: {day2_count} 个类器官")
    print(f"  Day3: {day3_count} 个类器官")
    
    # 完整轨迹统计
    complete_tracks = 0
    for particle in tracked_df['particle'].unique():
        track = tracked_df[tracked_df['particle'] == particle]
        if len(track['frame'].unique()) == 3:
            complete_tracks += 1
    
    print(f"\n完整轨迹 (Day1→Day2→Day3): {complete_tracks} 条")
    
    # 匹配统计
    print(f"\nDay2匹配情况:")
    print(f"  匹配到Day1: {len(day2_mapping)} 个")
    print(f"  新增: {day2_count - len(day2_mapping)} 个")
    
    print(f"\nDay3匹配情况:")
    print(f"  匹配到Day1: {len(day3_mapping)} 个")
    print(f"  新增: {day3_count - len(day3_mapping)} 个")
    
    # 轨迹长度分布
    track_lengths = tracked_df.groupby('particle')['frame'].nunique()
    print(f"\n轨迹长度分布:")
    print(f"  1帧 (只出现在一个时间点): {(track_lengths == 1).sum()}")
    print(f"  2帧: {(track_lengths == 2).sum()}")
    print(f"  3帧 (完整追踪): {(track_lengths == 3).sum()}")
    
    print("="*70)


def main():
    print("="*70)
    print("基于Trackpy的类器官追踪")
    print("="*70)
    
    # 读取数据
    print("\n读取数据...")
    data1, header1 = nrrd.read('ID_organoid/organoid_031_tracked.nii.seg.nrrd')
    data2, header2 = nrrd.read('ID_organoid/organoid_043_tracked.nii.seg.nrrd')
    data3, header3 = nrrd.read('ID_organoid/organoid_062_tracked.nii.seg.nrrd')
    
    # 读取成像数据
    import nibabel as nib
    image1 = nib.load('ID_organoid/organoid_031_0000.nii.gz').get_fdata()
    image2 = nib.load('ID_organoid/organoid_043_0000.nii.gz').get_fdata()
    image3 = nib.load('ID_organoid/organoid_062_0000.nii.gz').get_fdata()
    
    print(f"  ✓ Day1: {len(np.unique(data1))-1} 个类器官")
    print(f"  ✓ Day2: {len(np.unique(data2))-1} 个类器官")
    print(f"  ✓ Day3: {len(np.unique(data3))-1} 个类器官")
    
    # 提取特征
    print("\n提取特征...")
    features1 = extract_features_from_labels(data1, image1)
    features2 = extract_features_from_labels(data2, image2)
    features3 = extract_features_from_labels(data3, image3)
    
    # 准备trackpy输入
    df = prepare_tracking_dataframe(features1, features2, features3)
    
    # 计算自适应搜索范围
    # 基于Day1到Day2的平均位移
    print("\n计算自适应搜索范围...")
    day1_centroids = df[df['frame'] == 0][['x', 'y', 'z']].values
    day2_centroids = df[df['frame'] == 1][['x', 'y', 'z']].values
    
    from scipy.spatial.distance import cdist
    if len(day1_centroids) > 0 and len(day2_centroids) > 0:
        distances = cdist(day1_centroids, day2_centroids)
        min_distances = distances.min(axis=1)
        avg_displacement = np.median(min_distances)
        search_range = int(avg_displacement * 1.8)  # 1.8倍中位数(降低避免过多候选)
        search_range = max(search_range, 60)  # 最小60像素
        search_range = min(search_range, 100)  # 最大100像素(避免SubnetOversizeException)
    else:
        search_range = 80
    
    print(f"  自适应搜索范围: {search_range} 像素 (median displacement: {avg_displacement:.1f})")
    
    # 使用trackpy追踪
    tracked_df = track_with_trackpy(df, search_range=search_range, memory=1)
    
    # 创建标签映射
    print("\n创建标签映射...")
    day2_mapping, day3_mapping = create_label_mapping_from_tracking(tracked_df)
    
    # 应用映射 - 提供3种处理方式
    print("\n应用标签映射...")
    print("  处理未追踪到的类器官:")
    print("    - 'remove': 移除(只保留追踪到的)")
    print("    - 'special_id': 分配特殊ID(从1000开始,便于区分)")
    print("    - 'keep_original': 保留原始ID")
    print("  当前使用: special_id (推荐)")
    
    data2_mapped, day2_unmapped = apply_label_mapping(data2.copy(), day2_mapping, handle_unmapped='special_id')
    data3_mapped, day3_unmapped = apply_label_mapping(data3.copy(), day3_mapping, handle_unmapped='special_id')
    
    print(f"  Day2: {len(day2_mapping)} 个追踪到, {len(day2_unmapped)} 个新增(ID≥1000)")
    print(f"  Day3: {len(day3_mapping)} 个追踪到, {len(day3_unmapped)} 个新增(ID≥1000)")
    
    # 保存结果
    print("\n保存结果...")
    os.makedirs('results', exist_ok=True)
    
    save_tracking_results(data2_mapped, 'results/day2_trackpy_matched.seg.nrrd', header1, header2)
    save_tracking_results(data3_mapped, 'results/day3_trackpy_matched.seg.nrrd', header1, header3)
    
    # 另外保存"仅追踪"版本(移除新增的)
    data2_tracked_only, _ = apply_label_mapping(data2.copy(), day2_mapping, handle_unmapped='remove')
    data3_tracked_only, _ = apply_label_mapping(data3.copy(), day3_mapping, handle_unmapped='remove')
    
    save_tracking_results(data2_tracked_only, 'results/day2_tracked_only.seg.nrrd', header1, header2)
    save_tracking_results(data3_tracked_only, 'results/day3_tracked_only.seg.nrrd', header1, header3)
    print(f"  ✓ 已保存: results/day2_tracked_only.seg.nrrd (仅包含从Day1追踪到的)")
    print(f"  ✓ 已保存: results/day3_tracked_only.seg.nrrd (仅包含从Day1追踪到的)")
    
    # 保存Day1参考
    nrrd.write('results/day1_reference.seg.nrrd', data1.astype(np.int16), header1)
    print(f"  ✓ 已保存: results/day1_reference.seg.nrrd")
    
    # 保存追踪日志
    tracked_df.to_csv('results/trackpy_tracking_log.csv', index=False)
    print(f"  ✓ 追踪日志: results/trackpy_tracking_log.csv")
    
    # 可视化
    print("\n生成可视化...")
    visualize_trajectories(tracked_df, 'results/trackpy_trajectories.png')
    
    # 生成报告
    generate_tracking_report(tracked_df, day2_mapping, day3_mapping)
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print("\n生成的文件:")
    print("  1. results/day1_reference.seg.nrrd")
    print("  2. results/day2_trackpy_matched.seg.nrrd (包含新增,ID≥1000)")
    print("  3. results/day3_trackpy_matched.seg.nrrd (包含新增,ID≥1000)")
    print("  4. results/day2_tracked_only.seg.nrrd (仅Day1追踪到的)")
    print("  5. results/day3_tracked_only.seg.nrrd (仅Day1追踪到的)")
    print("  6. results/trackpy_tracking_log.csv")
    print("  7. results/trackpy_trajectories.png")
    print("\n推荐在3D Slicer中:")
    print("  - 直接拖拽 *_tracked_only.seg.nrrd 查看纯追踪效果")
    print("  - 拖拽 *_trackpy_matched.seg.nrrd 查看完整数据(ID≥1000是新增)")
    print("  - .seg.nrrd文件会自动识别为Segmentation!")
    print("="*70)


if __name__ == "__main__":
    main()
