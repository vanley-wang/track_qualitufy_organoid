import nibabel as nib
import numpy as np
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import pandas as pd
import os

def track_organoids(day1_path, day2_path, output_path, max_distance_pixels=50):
    """
    通过质心最近邻匹配，将 Day 2 的标签 ID 修改为与 Day 1 一致。
    """
    print(f"正在加载文件...")
    print(f"Day 1 (基准): {day1_path}")
    print(f"Day 2 (待修正): {day2_path}")

    # 1. 加载 NIfTI 文件
    nii1 = nib.load(day1_path)
    nii2 = nib.load(day2_path)
    data1 = nii1.get_fdata().astype(int)
    data2 = nii2.get_fdata().astype(int)

    # 2. 提取 Day 1 的属性 (ID 和 质心)
    
    print("正在计算 Day 1 质心...")
    props1 = regionprops(data1)
    if not props1:
        print("错误：Day 1 标签文件中没有检测到对象！")
        return

    # 建立 Day 1 的查找表： {ID: Centroid}
    # regionprops 的 label 默认是从 1 开始的，但也可能不连续
    day1_centroids = []
    day1_ids = []
    for p in props1:
        day1_centroids.append(p.centroid)
        day1_ids.append(p.label)
    
    day1_centroids = np.array(day1_centroids)
    day1_ids = np.array(day1_ids)

    # 使用 KDTree 加速最近邻搜索
    tree = cKDTree(day1_centroids)

    # 3. 提取 Day 2 的属性
    print("正在计算 Day 2 质心并匹配...")
    props2 = regionprops(data2)
    
    # 创建一个新的空数组来存放修正后的标签
    new_data2 = np.zeros_like(data2)
    
    matched_log = []
    used_day1_ids = set()

    # 下一个可用的新 ID (用于那些在 Day 1 没出现过的新生类器官)
    # 假设最大 ID 是两者中最大的那个 + 1
    next_new_id = max(np.max(data1), np.max(data2)) + 1

    # 4. 遍历 Day 2 的每一个球进行匹配
    for p2 in props2:
        current_centroid = p2.centroid
        current_id_old = p2.label
        
        # 在 Day 1 中找最近的邻居
        # k=1 表示找最近的一个
        distance, index = tree.query(current_centroid, k=1)
        
        matched_id = None
        status = ""

        # 判断匹配是否有效
        # 必须在距离阈值内 (防止把十万八千里外的球强行匹配)
        if distance <= max_distance_pixels:
            candidate_id = day1_ids[index]
            
            # 简单的冲突处理：如果这个 Day 1 ID 已经被别人认领了，说明有两个球挤在一起
            # 这里简单处理：谁更近谁得。但在简单代码里，我们允许重复或者直接分配
            # 为简单起见，这里直接分配，假设配准足够好
            matched_id = candidate_id
            status = "Matched"
        else:
            # 距离太远，认为是新生成的类器官，分配新 ID
            matched_id = next_new_id
            next_new_id += 1
            status = "New Object"

        # 填入新数据
        # 把原图中等于 current_id_old 的区域，在新图中赋值为 matched_id
        # 使用切片加速赋值 (p2.slice 提供了边界框)
        mask = (data2[p2.slice] == current_id_old)
        new_data2[p2.slice][mask] = matched_id
        
        matched_log.append({
            'Day2_Old_ID': current_id_old,
            'Day1_Matched_ID': matched_id if status == "Matched" else None,
            'Distance': distance,
            'Status': status,
            'New_ID': matched_id
        })

    # 5. 保存结果
    print(f"正在保存修正后的文件: {output_path}")
    new_nii = nib.Nifti1Image(new_data2.astype(np.int16), nii2.affine, nii2.header)
    nib.save(new_nii, output_path)
    
    # 保存匹配日志，方便你检查
    df_log = pd.DataFrame(matched_log)
    log_path = output_path.replace('.nii.gz', '_match_log.csv')
    df_log.to_csv(log_path, index=False)
    print(f"匹配日志已保存: {log_path}")
    print("完成！")

# --- 这里修改你的文件路径 ---
# 注意：Day2 的文件必须是你已经配准好(Aligned) 且 经过 Islands 分割过的
d1_file = 'Day1_Label.nii.gz'        # 你的 Day 1 标签
d2_file = 'Day2_Label_Aligned.nii.gz' # 你的 Day 2 标签 (配准过，且做了 Split Islands)
out_file = 'Day2_Label_Tracked.nii.gz' # 输出文件

# 运行 (如果文件存在)
if os.path.exists(d1_file) and os.path.exists(d2_file):
    track_organoids(d1_file, d2_file, out_file)
else:
    print("请检查文件路径是否正确！")