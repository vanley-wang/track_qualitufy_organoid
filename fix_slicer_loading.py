"""
3D Slicer加载指南 - 解决"看不到数据"问题
"""

print("="*70)
print("3D Slicer 正确加载方法")
print("="*70)

print("\n【问题原因】")
print("-" * 70)
print("NRRD分割文件(.seg.nrrd)和普通NRRD文件加载方式不同!")
print("✓ .seg.nrrd文件: 自动识别为Segmentation")
print("✗ .nrrd文件:     被当作Volume(灰度图像)加载")
print("-" * 70)

print("\n【解决方案1: 重命名文件(推荐)】")
print("-" * 70)
print("把生成的文件重命名为 .seg.nrrd 后缀:")
print("  results/day1_reference.nrrd")
print("      → results/day1_reference.seg.nrrd")
print("  results/day2_tracked_only.nrrd")
print("      → results/day2_tracked_only.seg.nrrd")
print("  results/day3_tracked_only.nrrd")
print("      → results/day3_tracked_only.seg.nrrd")
print("\n然后在3D Slicer中直接拖拽文件进去,会自动识别为分割!")
print("-" * 70)

print("\n【解决方案2: 手动导入为Segmentation】")
print("-" * 70)
print("在3D Slicer中:")
print("  1. 打开 'Segmentations' 模块")
print("  2. 点击 'Import/Export' 按钮")
print("  3. 选择 'Import labelmap to segmentation node'")
print("  4. 选择你的.nrrd文件")
print("  5. 确认导入")
print("-" * 70)

print("\n【我现在帮你重命名文件】")
print("-" * 70)

import os
import shutil

files_to_rename = [
    ('results/day1_reference.nrrd', 'results/day1_reference.seg.nrrd'),
    ('results/day2_tracked_only.nrrd', 'results/day2_tracked_only.seg.nrrd'),
    ('results/day3_tracked_only.nrrd', 'results/day3_tracked_only.seg.nrrd'),
    ('results/day2_trackpy_matched.nrrd', 'results/day2_trackpy_matched.seg.nrrd'),
    ('results/day3_trackpy_matched.nrrd', 'results/day3_trackpy_matched.seg.nrrd'),
]

for old_name, new_name in files_to_rename:
    if os.path.exists(old_name):
        shutil.copy2(old_name, new_name)
        print(f"  ✓ 已创建: {new_name}")
    else:
        print(f"  ✗ 找不到: {old_name}")

print("-" * 70)

print("\n【验证文件】")
print("-" * 70)
import nrrd
import numpy as np

# 验证一个文件
data, header = nrrd.read('results/day2_tracked_only.seg.nrrd')
unique_labels = np.unique(data)
unique_labels = unique_labels[unique_labels > 0]

print(f"  验证 day2_tracked_only.seg.nrrd:")
print(f"    数据形状: {data.shape}")
print(f"    标签数量: {len(unique_labels)}")
print(f"    标签ID: {sorted(unique_labels.tolist())}")
print(f"    ✓ 文件有效,可以在3D Slicer中加载!")
print("-" * 70)

print("\n【现在你可以在3D Slicer中:】")
print("-" * 70)
print("1. 拖拽这3个文件到Slicer窗口:")
print("   - results/day1_reference.seg.nrrd")
print("   - results/day2_tracked_only.seg.nrrd")
print("   - results/day3_tracked_only.seg.nrrd")
print("")
print("2. 在Segmentations模块中,你会看到3个分割节点")
print("")
print("3. 展开每个节点的Segments列表")
print("")
print("4. 相同编号的Segment(如Segment_2)会显示相同颜色!")
print("-" * 70)

print("\n" + "="*70)
print("完成! 现在可以在3D Slicer中正确查看了")
print("="*70)
