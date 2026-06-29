"""
解释trackpy_tracking_log.csv的内容
"""
import pandas as pd

# 读取追踪日志
df = pd.read_csv('results/trackpy_tracking_log.csv')

print("="*70)
print("Trackpy 追踪结果解释")
print("="*70)

print("\n【CSV文件每列的含义】")
print("-" * 70)
print("label:          原始分割标签的ID号")
print("x, y, z:        类器官质心的3D坐标(像素单位)")
print("volume:         类器官体积(体素数量)")
print("sphericity:     球形度(越大越圆)")
print("mean_intensity: 平均灰度强度")
print("frame:          时间点 (0=Day1, 1=Day2, 2=Day3)")
print("particle:       trackpy分配的追踪ID(同一个类器官在不同时间的ID相同)")
print("-" * 70)

# 统计信息
print("\n【基本统计】")
day1 = df[df['frame'] == 0]
day2 = df[df['frame'] == 1]
day3 = df[df['frame'] == 2]

print(f"Day1 (frame=0): {len(day1)} 个类器官")
print(f"Day2 (frame=1): {len(day2)} 个类器官")
print(f"Day3 (frame=2): {len(day3)} 个类器官")
print(f"总追踪轨迹数: {df['particle'].nunique()} 条")

# 完整轨迹分析
print("\n【轨迹长度分布】")
track_lengths = df.groupby('particle')['frame'].nunique()
print(f"只在1个时间点出现: {(track_lengths == 1).sum()} 条")
print(f"在2个时间点出现:   {(track_lengths == 2).sum()} 条")
print(f"在3个时间点出现:   {(track_lengths == 3).sum()} 条 ← 完整追踪")

# 显示几个完整追踪的例子
print("\n【完整追踪示例 (Day1→Day2→Day3)】")
complete_particles = track_lengths[track_lengths == 3].index[:5]
for pid in complete_particles:
    track = df[df['particle'] == pid].sort_values('frame')
    print(f"\n  Particle {pid}:")
    for _, row in track.iterrows():
        day_name = ['Day1', 'Day2', 'Day3'][int(row['frame'])]
        print(f"    {day_name}: 原始label={int(row['label']):2d}, "
              f"位置=({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f}), "
              f"体积={int(row['volume']):6d}")

# Day1作为参考的匹配统计
print("\n【以Day1为参考的匹配情况】")
day1_particles = set(day1['particle'].unique())
day2_particles = set(day2['particle'].unique())
day3_particles = set(day3['particle'].unique())

day1_to_day2 = len(day1_particles & day2_particles)
day1_to_day3 = len(day1_particles & day3_particles)
day1_to_both = len(day1_particles & day2_particles & day3_particles)

print(f"Day1的23个类器官中:")
print(f"  → {day1_to_day2} 个在Day2找到")
print(f"  → {day1_to_day3} 个在Day3找到")
print(f"  → {day1_to_both} 个在Day2和Day3都找到 ← 完整追踪")

# 新增的类器官
day2_new = len(day2_particles - day1_particles)
day3_new = len(day3_particles - day1_particles)
print(f"\nDay2新增: {day2_new} 个类器官(Day1没有的)")
print(f"Day3新增: {day3_new} 个类器官(Day1没有的)")

# 丢失的类器官
day1_lost_in_day2 = len(day1_particles - day2_particles)
day1_lost_in_day3 = len(day1_particles - day3_particles)
print(f"\nDay1的类器官在Day2丢失: {day1_lost_in_day2} 个")
print(f"Day1的类器官在Day3丢失: {day1_lost_in_day3} 个")

# 查看你选中的几行数据
print("\n【你选中的几行数据解释】")
print("-" * 70)
selected = df[(df['frame'] == 2) & (df['label'].isin([2, 1, 14, 52]))]
for _, row in selected.iterrows():
    print(f"\nDay3的原始label {int(row['label'])}: ")
    print(f"  → 被追踪为 particle {int(row['particle'])}")
    
    # 找这个particle在其他时间点的情况
    track = df[df['particle'] == row['particle']].sort_values('frame')
    if len(track) > 1:
        print(f"  → 这是一个追踪到的类器官:")
        for _, t in track.iterrows():
            day = ['Day1', 'Day2', 'Day3'][int(t['frame'])]
            print(f"     {day}: label={int(t['label']):2d}, 体积={int(t['volume']):6d}")
    else:
        print(f"  → 这是Day3新增的类器官(Day1/Day2都没有)")

print("\n" + "="*70)
print("总结:")
print("="*70)
print("✓ 你的数据处理是对的!脚本读取的是分割标签文件(.nrrd)")
print("✓ particle列是追踪的关键:相同particle表示同一个类器官在不同时间点")
print("✓ 脚本自动从分割标签中提取质心、体积等特征进行追踪")
print("✓ 生成的 results/*.nrrd 文件已经重新标记过,可以在3D Slicer查看")
print("="*70)
