"""
帮助理解trackpy追踪结果和在3D Slicer中如何查看
"""
import pandas as pd
import nrrd
import numpy as np

print("="*70)
print("Trackpy追踪结果文件说明")
print("="*70)

# 读取追踪日志
df = pd.read_csv('results/trackpy_tracking_log.csv')

print("\n【生成的文件类型】")
print("-" * 70)
print("1. day1_reference.nrrd           - Day1原始数据(参考)")
print("2. day2_trackpy_matched.nrrd     - Day2完整数据(追踪+新增)")
print("3. day3_trackpy_matched.nrrd     - Day3完整数据(追踪+新增)")
print("4. day2_tracked_only.nrrd        - Day2仅追踪到的(纯净版)")
print("5. day3_tracked_only.nrrd        - Day3仅追踪到的(纯净版)")
print("-" * 70)

print("\n【关键理解】")
print("-" * 70)
print("✓ 追踪到的类器官: label ID = 1~23 (Day1的ID)")
print("✓ 新增的类器官:   label ID ≥ 1000 (便于区分)")
print("✓ 在3D Slicer中同一个ID = 同一个类器官在不同时间点")
print("-" * 70)

# 读取文件检查
print("\n【文件内容检查】")
data1, _ = nrrd.read('results/day1_reference.nrrd')
data2_matched, _ = nrrd.read('results/day2_trackpy_matched.nrrd')
data3_matched, _ = nrrd.read('results/day3_trackpy_matched.nrrd')
data2_tracked, _ = nrrd.read('results/day2_tracked_only.nrrd')
data3_tracked, _ = nrrd.read('results/day3_tracked_only.nrrd')

labels1 = sorted([x for x in np.unique(data1) if x > 0])
labels2_matched = sorted([x for x in np.unique(data2_matched) if x > 0])
labels3_matched = sorted([x for x in np.unique(data3_matched) if x > 0])
labels2_tracked = sorted([x for x in np.unique(data2_tracked) if x > 0])
labels3_tracked = sorted([x for x in np.unique(data3_tracked) if x > 0])

print(f"\nDay1参考: {len(labels1)} 个类器官")
print(f"  Label IDs: {labels1}")

print(f"\nDay2完整版: {len(labels2_matched)} 个类器官")
tracked_2 = [x for x in labels2_matched if x < 1000]
new_2 = [x for x in labels2_matched if x >= 1000]
print(f"  追踪到的: {tracked_2}")
print(f"  新增的:   {new_2}")

print(f"\nDay3完整版: {len(labels3_matched)} 个类器官")
tracked_3 = [x for x in labels3_matched if x < 1000]
new_3 = [x for x in labels3_matched if x >= 1000]
print(f"  追踪到的: {tracked_3}")
print(f"  新增的:   {new_3}")

print(f"\nDay2纯净版: {len(labels2_tracked)} 个类器官(仅追踪到的)")
print(f"  Label IDs: {labels2_tracked}")

print(f"\nDay3纯净版: {len(labels3_tracked)} 个类器官(仅追踪到的)")
print(f"  Label IDs: {labels3_tracked}")

# 完整追踪分析
print("\n【完整追踪的19个类器官 (Day1→Day2→Day3)】")
print("-" * 70)
complete_particles = df.groupby('particle')['frame'].nunique()
complete_particles = complete_particles[complete_particles == 3].index

for i, pid in enumerate(sorted(complete_particles), 1):
    track = df[df['particle'] == pid].sort_values('frame')
    
    day1_label = int(track[track['frame'] == 0].iloc[0]['label'])
    day2_label = int(track[track['frame'] == 1].iloc[0]['label'])
    day3_label = int(track[track['frame'] == 2].iloc[0]['label'])
    
    day1_vol = int(track[track['frame'] == 0].iloc[0]['volume'])
    day2_vol = int(track[track['frame'] == 1].iloc[0]['volume'])
    day3_vol = int(track[track['frame'] == 2].iloc[0]['volume'])
    
    print(f"{i:2d}. Day1的label {day1_label:2d} (体积{day1_vol:7d}) → "
          f"Day2原始label {day2_label:2d} (体积{day2_vol:7d}) → "
          f"Day3原始label {day3_label:2d} (体积{day3_vol:7d})")
    print(f"    → 在结果文件中都标记为 label {day1_label}")

print("\n【Day2新增的7个类器官(Day1没有的)】")
print("-" * 70)
day2_new_particles = set(df[df['frame'] == 1]['particle'].unique()) - set(df[df['frame'] == 0]['particle'].unique())
for i, pid in enumerate(sorted(day2_new_particles), 1):
    track = df[df['particle'] == pid]
    if 1 in track['frame'].values:
        row = track[track['frame'] == 1].iloc[0]
        orig_label = int(row['label'])
        volume = int(row['volume'])
        new_id = 1000 + i
        print(f"{i}. 原始Day2 label {orig_label:2d} (体积{volume:7d}) → "
              f"在day2_trackpy_matched.nrrd中标记为 {new_id}")

print("\n【Day3新增的4个类器官(Day1没有的)】")
print("-" * 70)
day3_new_particles = set(df[df['frame'] == 2]['particle'].unique()) - set(df[df['frame'] == 0]['particle'].unique())
for i, pid in enumerate(sorted(day3_new_particles), 1):
    track = df[df['particle'] == pid]
    if 2 in track['frame'].values:
        row = track[track['frame'] == 2].iloc[0]
        orig_label = int(row['label'])
        volume = int(row['volume'])
        new_id = 1000 + i
        print(f"{i}. 原始Day3 label {orig_label:2d} (体积{volume:7d}) → "
              f"在day3_trackpy_matched.nrrd中标记为 {new_id}")

print("\n" + "="*70)
print("在3D Slicer中如何查看")
print("="*70)
print("\n【方法1: 查看完整追踪(推荐初学者)】")
print("  1. 加载 day1_reference.nrrd")
print("  2. 加载 day2_tracked_only.nrrd")
print("  3. 加载 day3_tracked_only.nrrd")
print("  → 相同颜色 = 同一个类器官在不同时间点")
print("  → 只显示从Day1成功追踪到的类器官")

print("\n【方法2: 查看完整数据(包含新增)】")
print("  1. 加载 day1_reference.nrrd")
print("  2. 加载 day2_trackpy_matched.nrrd")
print("  3. 加载 day3_trackpy_matched.nrrd")
print("  → ID < 1000: 追踪到的类器官(相同颜色)")
print("  → ID ≥ 1000: 新增的类器官(不同颜色)")

print("\n【如何找到对应关系】")
print("  在Slicer的Segmentation模块中:")
print("  - 展开Segments列表")
print("  - 每个Segment显示为 'Segment_X'")
print("  - X就是label ID(例如Segment_2就是label 2)")
print("  - ID < 1000的在Day1/Day2/Day3中颜色相同")

print("\n【举例说明】")
print("  如果你想看Day1的label 2类器官:")
print("  1. 在day1_reference.nrrd中找 Segment_2")
print("  2. 在day2_tracked_only.nrrd中找 Segment_2 (同一个类器官)")
print("  3. 在day3_tracked_only.nrrd中找 Segment_2 (同一个类器官)")
print("  → 它们会显示相同的颜色!")

print("\n" + "="*70)
