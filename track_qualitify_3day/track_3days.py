import nibabel as nib
import nrrd
import numpy as np
from skimage.measure import regionprops
from scipy.spatial import cKDTree
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
        # 使用参考header或创建新的
        if reference_header is not None:
            nrrd.write(filepath, data.astype(np.int16), reference_header)
        else:
            nrrd.write(filepath, data.astype(np.int16))
    else:
        nii = nib.Nifti1Image(data.astype(np.int16), affine, header)
        nib.save(nii, filepath)

def track_organoids_3days(day1_path, day2_path, day3_path, output_dir, max_distance_pixels=100):
    """
    通过质心最近邻匹配,跟踪三天的类器官,使同一个类器官在不同天的ID保持一致。
    
    参数:
        day1_path: DAY1标签文件路径
        day2_path: DAY2标签文件路径  
        day3_path: DAY3标签文件路径
        output_dir: 输出目录
        max_distance_pixels: 最大匹配距离(像素)
    """
    print("="*60)
    print("开始三天类器官ID匹配...")
    print(f"DAY1 (基准): {os.path.basename(day1_path)}")
    print(f"DAY2 (待匹配): {os.path.basename(day2_path)}")
    print(f"DAY3 (待匹配): {os.path.basename(day3_path)}")
    print("="*60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ===== 步骤1: 加载所有数据 =====
    print("\n[1/4] 正在加载标签文件...")
    data1, affine1, header1 = load_label_file(day1_path)
    data2, affine2, header2 = load_label_file(day2_path)
    data3, affine3, header3 = load_label_file(day3_path)
    
    print(f"   DAY1: {data1.shape}, 唯一标签: {len(np.unique(data1))-1}")
    print(f"   DAY2: {data2.shape}, 唯一标签: {len(np.unique(data2))-1}")
    print(f"   DAY3: {data3.shape}, 唯一标签: {len(np.unique(data3))-1}")

    # ===== 步骤2: 提取DAY1的质心 (作为基准) =====
    print("\n[2/4] 提取DAY1质心(基准)...")
    props1 = regionprops(data1)
    if not props1:
        print("错误: DAY1标签文件中没有检测到对象!")
        return
    
    day1_centroids = np.array([p.centroid for p in props1])
    day1_ids = np.array([p.label for p in props1])
    print(f"   检测到 {len(day1_ids)} 个类器官")
    
    # 建立KDTree用于快速最近邻搜索
    tree1 = cKDTree(day1_centroids)

    # ===== 步骤3: 匹配DAY2到DAY1 =====
    print("\n[3/4] 匹配DAY2到DAY1...")
    new_data2, log2, next_id = match_to_reference(
        data2, day1_ids, tree1, max_distance_pixels, 
        max(day1_ids), "DAY2"
    )
    
    # DAY2匹配完成后,用更新的质心和ID作为DAY3的参考
    # 合并DAY1和已匹配的DAY2的信息
    props2_matched = regionprops(new_data2)
    day1_2_centroids = []
    day1_2_ids = []
    
    # 添加DAY1的所有类器官
    for p in props1:
        day1_2_centroids.append(p.centroid)
        day1_2_ids.append(p.label)
    
    # 添加DAY2新出现的类器官
    for p in props2_matched:
        if p.label not in day1_ids:
            day1_2_centroids.append(p.centroid)
            day1_2_ids.append(p.label)
    
    day1_2_centroids = np.array(day1_2_centroids)
    day1_2_ids = np.array(day1_2_ids)
    tree1_2 = cKDTree(day1_2_centroids)
    
    print(f"   DAY1+DAY2 共有 {len(day1_2_ids)} 个已知类器官")

    # ===== 步骤4: 匹配DAY3到DAY1+DAY2 =====
    print("\n[4/4] 匹配DAY3到DAY1+DAY2...")
    new_data3, log3, _ = match_to_reference(
        data3, day1_2_ids, tree1_2, max_distance_pixels,
        next_id, "DAY3"
    )

    # ===== 保存结果 =====
    print("\n正在保存结果...")
    
    # 保存修正后的标签文件(保持原格式)
    output_day2 = os.path.join(output_dir, "organoid_043_tracked_matched.nrrd")
    output_day3 = os.path.join(output_dir, "organoid_062_tracked_matched.nrrd")
    
    save_label_file(new_data2, output_day2, affine2, header2, header2)
    save_label_file(new_data3, output_day3, affine3, header3, header3)
    
    print(f"   ✓ DAY2修正后保存至: {output_day2}")
    print(f"   ✓ DAY3修正后保存至: {output_day3}")
    
    # 保存匹配日志
    log_path = os.path.join(output_dir, "tracking_log.csv")
    all_logs = log2 + log3
    df_log = pd.DataFrame(all_logs)
    df_log.to_csv(log_path, index=False)
    print(f"   ✓ 匹配日志保存至: {log_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("匹配统计:")
    print(f"  DAY1基准类器官数: {len(day1_ids)}")
    print(f"  DAY2匹配成功: {sum(1 for x in log2 if x['Status']=='Matched')}")
    print(f"  DAY2新增类器官: {sum(1 for x in log2 if x['Status']=='New')}")
    print(f"  DAY3匹配成功: {sum(1 for x in log3 if x['Status']=='Matched')}")
    print(f"  DAY3新增类器官: {sum(1 for x in log3 if x['Status']=='New')}")
    print("="*60)
    print("完成! ✓")


def match_to_reference(data_target, ref_ids, ref_tree, max_distance, current_max_id, day_name):
    """
    将目标数据匹配到参考数据
    
    返回:
        new_data: 修正后的标签数据
        logs: 匹配日志列表
        next_new_id: 下一个可用的新ID
    """
    props_target = regionprops(data_target)
    new_data = np.zeros_like(data_target)
    logs = []
    next_new_id = current_max_id + 1
    
    for p in props_target:
        current_centroid = p.centroid
        old_id = p.label
        
        # 在参考中找最近的邻居
        distance, index = ref_tree.query(current_centroid, k=1)
        
        if distance <= max_distance:
            # 匹配成功
            new_id = ref_ids[index]
            status = "Matched"
        else:
            # 距离太远,认为是新类器官
            new_id = next_new_id
            next_new_id += 1
            status = "New"
        
        # 更新标签
        mask = (data_target[p.slice] == old_id)
        new_data[p.slice][mask] = new_id
        
        logs.append({
            'Day': day_name,
            'Original_ID': old_id,
            'New_ID': new_id,
            'Distance': round(distance, 2),
            'Status': status,
            'Centroid': f"({p.centroid[0]:.1f}, {p.centroid[1]:.1f}, {p.centroid[2]:.1f})"
        })
        
        print(f"   {day_name} ID {old_id:3d} -> {new_id:3d} (距离: {distance:6.2f}, {status})")
    
    return new_data, logs, next_new_id


if __name__ == "__main__":
    # 设置文件路径
    base_dir = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/ID_organoid"
    
    day1_file = os.path.join(base_dir, "organoid_031_tracked.nii.seg.nrrd")
    day2_file = os.path.join(base_dir, "organoid_043_tracked.nii.seg.nrrd")
    day3_file = os.path.join(base_dir, "organoid_062_tracked.nii.seg.nrrd")
    
    output_directory = "/mnt/cache_ssd/wangwanli/mnt_new/sambashare/track_qualitify_organoid/results"
    
    # 检查文件是否存在
    if not os.path.exists(day1_file):
        print(f"错误: 找不到DAY1文件: {day1_file}")
        exit(1)
    if not os.path.exists(day2_file):
        print(f"错误: 找不到DAY2文件: {day2_file}")
        exit(1)
    if not os.path.exists(day3_file):
        print(f"错误: 找不到DAY3文件: {day3_file}")
        exit(1)
    
    # 运行跟踪
    track_organoids_3days(
        day1_path=day1_file,
        day2_path=day2_file, 
        day3_path=day3_file,
        output_dir=output_directory,
        max_distance_pixels=200  # 可以根据需要调整这个阈值
    )
