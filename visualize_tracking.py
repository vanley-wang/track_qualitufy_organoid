"""
生成追踪可视化结果 - 使用matplotlib 3D可视化
"""
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pandas as pd

print("="*70)
print("生成追踪可视化")
print("="*70)

# 读取数据
print("\n读取数据...")
data1, _ = nrrd.read('results/day1_reference.nrrd')
data2_tracked, _ = nrrd.read('results/day2_tracked_only.nrrd')
data3_tracked, _ = nrrd.read('results/day3_tracked_only.nrrd')

# 读取追踪日志
df = pd.read_csv('results/trackpy_tracking_log.csv')

# 找到完整追踪的类器官
complete_particles = df.groupby('particle')['frame'].nunique()
complete_particles = complete_particles[complete_particles == 3].index

print(f"完整追踪的类器官: {len(complete_particles)} 个")

# 生成颜色映射
np.random.seed(42)
colors = plt.cm.tab20(np.linspace(0, 1, 20))
np.random.shuffle(colors)

# ===== 可视化1: 3D质心轨迹图 =====
print("\n生成3D轨迹图...")
fig = plt.figure(figsize=(20, 6))

# 三个视角
views = [
    (1, 3, 1, 'XY平面 (从上往下看)', 0, 1),
    (1, 3, 2, 'XZ平面 (从前往后看)', 0, 2),
    (1, 3, 3, 'YZ平面 (从右往左看)', 1, 2)
]

for subplot_idx, subplot_rows, subplot_cols, title, dim1, dim2 in views:
    ax = fig.add_subplot(subplot_rows, subplot_cols, subplot_idx)
    
    for i, pid in enumerate(sorted(complete_particles)[:19]):  # 只显示完整追踪的
        track = df[df['particle'] == pid].sort_values('frame')
        
        coords = track[['x', 'y', 'z']].values
        color = colors[i % len(colors)]
        
        # 绘制轨迹
        ax.plot(coords[:, dim1], coords[:, dim2], 'o-', 
                color=color, alpha=0.6, linewidth=2, markersize=8)
        
        # 标记Day1起点
        ax.plot(coords[0, dim1], coords[0, dim2], 'o', 
                color=color, markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        # 标记Day3终点
        ax.plot(coords[-1, dim1], coords[-1, dim2], 's', 
                color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel(['X (pixels)', 'Y (pixels)', 'Z (pixels)'][dim1], fontsize=12)
    ax.set_ylabel(['X (pixels)', 'Y (pixels)', 'Z (pixels)'][dim2], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('results/tracking_trajectories_2d.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: results/tracking_trajectories_2d.png")
plt.close()

# ===== 可视化2: 真正的3D图 =====
print("\n生成真3D轨迹图...")
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

for i, pid in enumerate(sorted(complete_particles)[:19]):
    track = df[df['particle'] == pid].sort_values('frame')
    coords = track[['x', 'y', 'z']].values
    color = colors[i % len(colors)]
    
    # 绘制轨迹
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
            'o-', color=color, alpha=0.6, linewidth=2, markersize=6)
    
    # 标记Day1
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], 
               c=[color], s=200, marker='o', edgecolors='black', linewidths=2)
    
    # 标记Day3
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], 
               c=[color], s=150, marker='s', edgecolors='black', linewidths=2)

ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_zlabel('Z (pixels)', fontsize=12)
ax.set_title('3D Organoid Tracking Trajectories\n(○ = Day1, □ = Day3)', 
             fontsize=14, fontweight='bold')

plt.savefig('results/tracking_trajectories_3d.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: results/tracking_trajectories_3d.png")
plt.close()

# ===== 可视化3: 体积变化图 =====
print("\n生成体积变化图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 每个类器官的体积变化
ax = axes[0, 0]
for i, pid in enumerate(sorted(complete_particles)[:19]):
    track = df[df['particle'] == pid].sort_values('frame')
    volumes = track['volume'].values
    day1_label = int(track[track['frame'] == 0].iloc[0]['label'])
    
    ax.plot([0, 1, 2], volumes, 'o-', 
            color=colors[i % len(colors)], alpha=0.7, linewidth=2, 
            markersize=8, label=f'Label {day1_label}')

ax.set_xlabel('Time Point', fontsize=12)
ax.set_ylabel('Volume (voxels)', fontsize=12)
ax.set_title('Volume Changes Over Time', fontsize=14, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Day1', 'Day2', 'Day3'])
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)

# 3.2 体积变化率分布
ax = axes[0, 1]
growth_rates = []
for pid in complete_particles:
    track = df[df['particle'] == pid].sort_values('frame')
    v1 = track[track['frame'] == 0].iloc[0]['volume']
    v3 = track[track['frame'] == 2].iloc[0]['volume']
    growth_rate = (v3 - v1) / v1 * 100
    growth_rates.append(growth_rate)

ax.hist(growth_rates, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Volume Change (%)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Volume Changes (Day1→Day3)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3.3 位移距离分布
ax = axes[1, 0]
displacements = []
for pid in complete_particles:
    track = df[df['particle'] == pid].sort_values('frame')
    coords1 = track[track['frame'] == 0][['x', 'y', 'z']].values[0]
    coords3 = track[track['frame'] == 2][['x', 'y', 'z']].values[0]
    displacement = np.linalg.norm(coords3 - coords1)
    displacements.append(displacement)

ax.hist(displacements, bins=15, color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Displacement (pixels)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Displacements (Day1→Day3)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3.4 体积 vs 位移散点图
ax = axes[1, 1]
initial_volumes = []
for pid in complete_particles:
    track = df[df['particle'] == pid].sort_values('frame')
    v1 = track[track['frame'] == 0].iloc[0]['volume']
    initial_volumes.append(v1)

scatter = ax.scatter(initial_volumes, displacements, 
                     c=growth_rates, cmap='RdYlGn', 
                     s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('Initial Volume (Day1, voxels)', fontsize=12)
ax.set_ylabel('Displacement (pixels)', fontsize=12)
ax.set_title('Initial Volume vs Displacement\n(color = volume change %)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Volume Change (%)')

plt.tight_layout()
plt.savefig('results/tracking_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: results/tracking_analysis.png")
plt.close()

# ===== 可视化4: 生成详细报告 =====
print("\n生成追踪详细报告...")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

report_lines = [
    "Trackpy Organoid Tracking Report",
    "=" * 60,
    "",
    f"Total tracked organoids (Day1→Day2→Day3): {len(complete_particles)}",
    "",
    "Individual Tracking Results:",
    "-" * 60,
]

for i, pid in enumerate(sorted(complete_particles), 1):
    track = df[df['particle'] == pid].sort_values('frame')
    
    day1_label = int(track[track['frame'] == 0].iloc[0]['label'])
    day1_vol = int(track[track['frame'] == 0].iloc[0]['volume'])
    day2_vol = int(track[track['frame'] == 1].iloc[0]['volume'])
    day3_vol = int(track[track['frame'] == 2].iloc[0]['volume'])
    
    coords1 = track[track['frame'] == 0][['x', 'y', 'z']].values[0]
    coords3 = track[track['frame'] == 2][['x', 'y', 'z']].values[0]
    displacement = np.linalg.norm(coords3 - coords1)
    
    vol_change = (day3_vol - day1_vol) / day1_vol * 100
    
    report_lines.append(
        f"{i:2d}. Label {day1_label:2d}: Vol {day1_vol:7d}→{day2_vol:7d}→{day3_vol:7d} "
        f"({vol_change:+6.1f}%), Disp: {displacement:.1f}px"
    )

report_lines.extend([
    "",
    "-" * 60,
    "Statistics:",
    f"  Mean volume change: {np.mean(growth_rates):.1f}%",
    f"  Std volume change:  {np.std(growth_rates):.1f}%",
    f"  Mean displacement:  {np.mean(displacements):.1f} pixels",
    f"  Max displacement:   {np.max(displacements):.1f} pixels",
    "",
    "Files generated:",
    "  - results/tracking_trajectories_2d.png (2D projections)",
    "  - results/tracking_trajectories_3d.png (3D view)",
    "  - results/tracking_analysis.png (statistical analysis)",
    "  - results/tracking_report.png (this report)",
])

report_text = "\n".join(report_lines)
ax.text(0.05, 0.95, report_text, 
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('results/tracking_report.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: results/tracking_report.png")
plt.close()

print("\n" + "="*70)
print("可视化完成!")
print("="*70)
print("\n生成的可视化文件:")
print("  1. results/tracking_trajectories_2d.png - 2D投影轨迹(3个视角)")
print("  2. results/tracking_trajectories_3d.png - 真3D轨迹图")
print("  3. results/tracking_analysis.png        - 统计分析图(体积/位移)")
print("  4. results/tracking_report.png          - 详细追踪报告")
print("\n这些图片可以直接查看,验证追踪准确性!")
print("="*70)
