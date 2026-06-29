"""
诊断trackpy追踪结果
"""
import pandas as pd
import numpy as np
import nrrd

# 读取追踪日志
df = pd.read_csv('results/trackpy_tracking_log.csv')

print("="*70)
print("诊断Trackpy追踪问题")
print("="*70)

# 分析Day1作为参考的映射
day1 = df[df['frame'] == 0]
day2 = df[df['frame'] == 1]
day3 = df[df['frame'] == 2]

print("\n【1. 原始标签统计】")
print(f"Day1原始标签: {sorted(day1['label'].unique())}")
print(f"Day2原始标签: {sorted(day2['label'].unique())}")
print(f"Day3原始标签: {sorted(day3['label'].unique())}")

# 分析映射关系
print("\n【2. Day2映射分析】")
day2_mapping = {}
for particle in df['particle'].unique():
    track = df[df['particle'] == particle]
    day1_in_track = track[track['frame'] == 0]
    day2_in_track = track[track['frame'] == 1]
    
    if len(day1_in_track) > 0 and len(day2_in_track) > 0:
        day1_label = int(day1_in_track.iloc[0]['label'])
        day2_label = int(day2_in_track.iloc[0]['label'])
        day2_mapping[day2_label] = day1_label

print(f"Day2有映射的标签数: {len(day2_mapping)}")
print(f"Day2映射关系 (前10个): {dict(list(day2_mapping.items())[:10])}")

day2_unmapped = set(day2['label'].unique()) - set(day2_mapping.keys())
print(f"Day2没有映射的标签 (Day1不存在): {sorted(day2_unmapped)}")

# 分析Day3映射
print("\n【3. Day3映射分析】")
day3_mapping = {}
for particle in df['particle'].unique():
    track = df[df['particle'] == particle]
    day1_in_track = track[track['frame'] == 0]
    day3_in_track = track[track['frame'] == 2]
    
    if len(day1_in_track) > 0 and len(day3_in_track) > 0:
        day1_label = int(day1_in_track.iloc[0]['label'])
        day3_label = int(day3_in_track.iloc[0]['label'])
        day3_mapping[day3_label] = day1_label

print(f"Day3有映射的标签数: {len(day3_mapping)}")
print(f"Day3映射关系 (前10个): {dict(list(day3_mapping.items())[:10])}")

day3_unmapped = set(day3['label'].unique()) - set(day3_mapping.keys())
print(f"Day3没有映射的标签 (Day1不存在): {sorted(day3_unmapped)}")

# 检查生成的文件
print("\n【4. 检查生成的文件】")
data2, _ = nrrd.read('results/day2_trackpy_matched.nrrd')
data3, _ = nrrd.read('results/day3_trackpy_matched.nrrd')

data2_labels = set(np.unique(data2))
data2_labels.discard(0)
data3_labels = set(np.unique(data3))
data3_labels.discard(0)

print(f"day2_trackpy_matched.nrrd 标签: {sorted(data2_labels)}")
print(f"day3_trackpy_matched.nrrd 标签: {sorted(data3_labels)}")

# 对比原始Day1
data1, _ = nrrd.read('ID_organoid/organoid_031_tracked.nii.seg.nrrd')
day1_labels = set(np.unique(data1))
day1_labels.discard(0)

print(f"\nDay1原始标签: {sorted(day1_labels)}")
print(f"Day2新增标签 (不在Day1): {sorted(data2_labels - day1_labels)}")
print(f"Day3新增标签 (不在Day1): {sorted(data3_labels - day1_labels)}")

# 问题诊断
print("\n【5. 问题诊断】")
print("-"*70)

if len(day2_unmapped) > 0:
    print(f"⚠️ 问题: Day2有 {len(day2_unmapped)} 个类器官在Day1不存在")
    print(f"   但 apply_label_mapping() 给它们分配了新ID: {sorted(data2_labels - day1_labels)}")
    print(f"   这导致报告显示'新增: 0 个',实际应该是'新增: {len(day2_unmapped)} 个'")

if len(day3_unmapped) > 0:
    print(f"⚠️ 问题: Day3有 {len(day3_unmapped)} 个类器官在Day1不存在")
    print(f"   但 apply_label_mapping() 给它们分配了新ID: {sorted(data3_labels - day1_labels)}")
    print(f"   这导致报告显示'新增: 0 个',实际应该是'新增: {len(day3_unmapped)} 个'")

print("\n【6. 正确的追踪统计】")
print("-"*70)
# 统计只基于Day1存在的类器官
day1_particles = set(day1['particle'].unique())
day2_particles_from_day1 = set()
day3_particles_from_day1 = set()

for particle in df['particle'].unique():
    track = df[df['particle'] == particle]
    if len(track[track['frame'] == 0]) > 0:  # Day1存在
        if len(track[track['frame'] == 1]) > 0:
            day2_particles_from_day1.add(particle)
        if len(track[track['frame'] == 2]) > 0:
            day3_particles_from_day1.add(particle)

print(f"Day1的 {len(day1_particles)} 个类器官中:")
print(f"  → 在Day2追踪到: {len(day2_particles_from_day1)} 个")
print(f"  → 在Day3追踪到: {len(day3_particles_from_day1)} 个")
print(f"  → 完整追踪(Day1→Day2→Day3): {len(day2_particles_from_day1 & day3_particles_from_day1)} 个")

print(f"\nDay2实际有: {len(day2)} 个类器官")
print(f"  → 来自Day1: {len(day2_mapping)} 个")
print(f"  → 新增: {len(day2) - len(day2_mapping)} 个")

print(f"\nDay3实际有: {len(day3)} 个类器官")
print(f"  → 来自Day1: {len(day3_mapping)} 个")
print(f"  → 新增: {len(day3) - len(day3_mapping)} 个")

print("\n" + "="*70)
print("建议:")
print("="*70)
print("问题在于 apply_label_mapping() 函数自动给未映射的标签分配新ID")
print("这导致:")
print("  1. 无法区分哪些是Day1追踪到的,哪些是新增的")
print("  2. 在3D Slicer中,新增的类器官也显示为Day1的颜色")
print("\n解决方案:")
print("  选项1: 移除未映射标签,只保留追踪到的")
print("  选项2: 给未映射标签分配特殊范围的ID (如从1000开始)")
print("  选项3: 保存两个版本 - 仅追踪版本 + 完整版本")
print("="*70)
