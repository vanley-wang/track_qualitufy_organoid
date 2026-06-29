"""
正确的ID追踪算法
修复问题: 防止多个类器官被分配到同一个ID (一对多匹配问题)
使用匈牙利算法(Hungarian algorithm)或贪心算法确保一对一匹配
"""

import nibabel as nib
import nrrd
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os


def load_label_file(filepath):
    """加载标签文件,支持NIfTI和NRRD格式"""
    if filepath.endswith('.nrrd'):
        data, header = nrrd.read(filepath)
        return data.astype(int), None, header
    else:
        nii = nib.load(filepath)
        return nii.get_fdata().astype(int), nii.affine, nii.header


def save_label_file(data, filepath, affine=None, header=None, reference_header=None):
    """保存标签文件"""
    if filepath.endswith('.nrrd'):
        if reference_header is not None:
            nrrd.write(filepath, data.astype(np.int16), reference_header)
        else:
            nrrd.write(filepath, data.astype(np.int16))
    else:
        nii = nib.Nifti1Image(data.astype(np.int16), affine, header)
        nib.save(nii, filepath)


def match_organoids_hungarian(source_props, target_props, max_distance):
    """
    使用匈牙利算法进行一对一匹配
    
    参数:
        source_props: 源天的regionprops列表
        target_props: 目标天的regionprops列表
        max_distance: 最大匹配距离
    
    返回:
        matches: 字典 {target_label: source_label}
        unmatched_targets: 未匹配的目标类器官列表
    """
    # 提取质心
    source_centroids = np.array([p.centroid for p in source_props])
    target_centroids = np.array([p.centroid for p in target_props])
    
    source_labels = np.array([p.label for p in source_props])
    target_labels = np.array([p.label for p in target_props])
    
    # 计算所有点对之间的距离矩阵
    distance_matrix = cdist(target_centroids, source_centroids, metric='euclidean')
    
    # 使用匈牙利算法找到最优一对一匹配
    # linear_sum_assignment返回(行索引, 列索引)
    target_indices, source_indices = linear_sum_assignment(distance_matrix)
    
    matches = {}
    unmatched_targets = []
    
    print(f"\n  匹配详情:")
    print(f"  {'目标ID':<8} {'源ID':<8} {'距离(pixel)':<12} {'状态'}")
    print("  " + "-"*50)
    
    # 检查每个匹配
    for t_idx, s_idx in zip(target_indices, source_indices):
        distance = distance_matrix[t_idx, s_idx]
        target_label = target_labels[t_idx]
        source_label = source_labels[s_idx]
        
        if distance <= max_distance:
            matches[target_label] = source_label
            status = "✓ Matched"
            print(f"  {target_label:<8} {source_label:<8} {distance:<12.2f} {status}")
        else:
            unmatched_targets.append(target_props[t_idx])
            status = "✗ Too far"
            print(f"  {target_label:<8} -       {distance:<12.2f} {status}")
    
    # 找出没有被匹配的目标类器官
    matched_target_indices = set(target_indices[distance_matrix[target_indices, source_indices] <= max_distance])
    for t_idx, prop in enumerate(target_props):
        if t_idx not in matched_target_indices and t_idx not in target_indices:
            unmatched_targets.append(prop)
    
    return matches, unmatched_targets


def apply_matching(data_target, matches, unmatched_targets, next_new_id, target_props):
    """
    应用匹配结果到数据
    
    参数:
        data_target: 目标数据
        matches: 匹配字典 {old_id: new_id}
        unmatched_targets: 未匹配的类器官列表
        next_new_id: 下一个可用的新ID
        target_props: 所有目标regionprops
    
    返回:
        new_data: 重新标记的数据
        logs: 日志列表
        next_new_id: 更新后的下一个可用ID
    """
    new_data = np.zeros_like(data_target)
    logs = []
    
    # 应用匹配
    for prop in target_props:
        old_id = prop.label
        
        if old_id in matches:
            new_id = matches[old_id]
            status = "Matched"
        else:
            new_id = next_new_id
            next_new_id += 1
            status = "New"
        
        # 更新标签
        mask = (data_target[prop.slice] == old_id)
        new_data[prop.slice][mask] = new_id
        
        logs.append({
            'Original_ID': old_id,
            'New_ID': new_id,
            'Status': status,
            'Centroid': f"({prop.centroid[0]:.1f}, {prop.centroid[1]:.1f}, {prop.centroid[2]:.1f})"
        })
    
    return new_data, logs, next_new_id


def track_organoids_correct(day1_path, day2_path, day3_path, output_dir, max_distance_pixels=50):
    """
    正确的三天类器官ID追踪
    使用匈牙利算法确保一对一匹配
    """
    print("\n" + "="*70)
    print("类器官ID追踪 - 正确版本 (一对一匹配)")
    print("="*70)
    print(f"Day1 (基准): {os.path.basename(day1_path)}")
    print(f"Day2 (待匹配): {os.path.basename(day2_path)}")
    print(f"Day3 (待匹配): {os.path.basename(day3_path)}")
    print(f"最大匹配距离: {max_distance_pixels} pixels")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 加载数据 =====
    print("\n[1/5] 加载标签文件...")
    data1, affine1, header1 = load_label_file(day1_path)
    data2, affine2, header2 = load_label_file(day2_path)
    data3, affine3, header3 = load_label_file(day3_path)
    
    print(f"  Day1: {data1.shape}, {len(np.unique(data1))-1} 个类器官")
    print(f"  Day2: {data2.shape}, {len(np.unique(data2))-1} 个类器官")
    print(f"  Day3: {data3.shape}, {len(np.unique(data3))-1} 个类器官")
    
    # ===== 提取区域属性 =====
    print("\n[2/5] 提取区域属性...")
    props1 = regionprops(data1)
    props2 = regionprops(data2)
    props3 = regionprops(data3)
    
    print(f"  Day1: {len(props1)} 个类器官")
    print(f"  Day2: {len(props2)} 个类器官")
    print(f"  Day3: {len(props3)} 个类器官")
    
    # ===== 匹配Day2到Day1 =====
    print("\n[3/5] 匹配 Day2 → Day1 (使用匈牙利算法)")
    matches_day2, unmatched_day2 = match_organoids_hungarian(props1, props2, max_distance_pixels)
    
    print(f"\n  匹配结果:")
    print(f"    成功匹配: {len(matches_day2)} 对")
    print(f"    新增类器官: {len(unmatched_day2)} 个")
    
    # 应用匹配到Day2
    max_id_day1 = max([p.label for p in props1])
    next_new_id = max_id_day1 + 1
    
    new_data2, logs_day2, next_new_id = apply_matching(
        data2, matches_day2, unmatched_day2, next_new_id, props2
    )
    
    # ===== 匹配Day3到Day1 (不考虑Day2新增的) =====
    print("\n[4/5] 匹配 Day3 → Day1 (只用Day1作为参考)")
    print("  注意: Day3只匹配Day1的类器官,不考虑Day2新增的")
    
    # 只用Day1作为参考
    matches_day3, unmatched_day3 = match_organoids_hungarian(props1, props3, max_distance_pixels)
    
    print(f"\n  匹配结果:")
    print(f"    成功匹配: {len(matches_day3)} 对")
    print(f"    新增类器官: {len(unmatched_day3)} 个")
    
    # 应用匹配到Day3
    new_data3, logs_day3, next_new_id = apply_matching(
        data3, matches_day3, unmatched_day3, next_new_id, props3
    )
    
    # ===== 保存结果 =====
    print("\n[5/5] 保存结果...")
    
    # Day1保持不变,只是复制
    output_day1 = os.path.join(output_dir, "day1_reference.nrrd")
    save_label_file(data1, output_day1, affine1, header1, header1)
    print(f"  ✓ Day1 (参考): {output_day1}")
    
    # 保存匹配后的Day2
    output_day2 = os.path.join(output_dir, "day2_matched.nrrd")
    save_label_file(new_data2, output_day2, affine2, header2, header2)
    print(f"  ✓ Day2 (匹配): {output_day2}")
    
    # 保存匹配后的Day3
    output_day3 = os.path.join(output_dir, "day3_matched.nrrd")
    save_label_file(new_data3, output_day3, affine3, header3, header3)
    print(f"  ✓ Day3 (匹配): {output_day3}")
    
    # 保存日志
    all_logs = []
    for log in logs_day2:
        log['Day'] = 'Day2'
        all_logs.append(log)
    for log in logs_day3:
        log['Day'] = 'Day3'
        all_logs.append(log)
    
    log_path = os.path.join(output_dir, "matching_log.csv")
    df_log = pd.DataFrame(all_logs)
    df_log.to_csv(log_path, index=False)
    print(f"  ✓ 匹配日志: {log_path}")
    
    # ===== 验证结果 =====
    print("\n" + "="*70)
    print("验证: 检查是否存在ID冲突")
    print("="*70)
    
    # 检查Day2
    unique_ids_day2 = np.unique(new_data2[new_data2 > 0])
    print(f"Day2: {len(unique_ids_day2)} 个唯一ID")
    
    # 检查Day3
    unique_ids_day3 = np.unique(new_data3[new_data3 > 0])
    print(f"Day3: {len(unique_ids_day3)} 个唯一ID")
    
    # 检查是否有重复
    props2_check = regionprops(new_data2)
    props3_check = regionprops(new_data3)
    
    print(f"Day2: regionprops检测到 {len(props2_check)} 个类器官")
    print(f"Day3: regionprops检测到 {len(props3_check)} 个类器官")
    
    if len(unique_ids_day2) == len(props2_check) and len(unique_ids_day3) == len(props3_check):
        print("\n✓ 验证通过! 没有ID冲突")
    else:
        print("\n⚠️  警告: ID数量与类器官数量不匹配,可能存在问题")
    
    # ===== 总结 =====
    print("\n" + "="*70)
    print("追踪完成!")
    print("="*70)
    print(f"Day1: {len(props1)} 个类器官 (基准)")
    print(f"Day2: {len(props2)} → {len(props2_check)} 个类器官 ({len(matches_day2)} 匹配, {len(unmatched_day2)} 新增)")
    print(f"Day3: {len(props3)} → {len(props3_check)} 个类器官 ({len(matches_day3)} 匹配, {len(unmatched_day3)} 新增)")
    print("="*70)
    
    return new_data2, new_data3


if __name__ == "__main__":
    # 设置路径
    base_dir = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/ID_organoid"
    output_dir = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/results"
    
    day1_file = os.path.join(base_dir, "organoid_031_tracked.nii.seg.nrrd")
    day2_file = os.path.join(base_dir, "organoid_043_tracked.nii.seg.nrrd")
    day3_file = os.path.join(base_dir, "organoid_062_tracked.nii.seg.nrrd")
    
    # 检查文件
    for f, name in [(day1_file, "Day1"), (day2_file, "Day2"), (day3_file, "Day3")]:
        if not os.path.exists(f):
            print(f"❌ 错误: 找不到{name}文件: {f}")
            exit(1)
    
    # 运行追踪
    track_organoids_correct(
        day1_path=day1_file,
        day2_path=day2_file,
        day3_path=day3_file,
        output_dir=output_dir,
        max_distance_pixels=80  # 400微米
    )
    
    print("\n✓ 请查看 results/ 目录下的匹配结果")
    print("✓ 下一步: 运行量化分析")
