#!/usr/bin/env python3
"""验证颜色匹配逻辑的修复"""

import nrrd
import numpy as np

# 模拟save_tracking_results中的逻辑
print("=== 模拟颜色匹配逻辑 ===\n")

# 1. 读取reference header
_, reference_header = nrrd.read('results/day1_reference.seg.nrrd')

# 2. 构建ref_segment_colors字典(就像代码中的逻辑)
ref_segment_colors = {}
if 'Segment0_ID' in reference_header:
    for key in reference_header.keys():
        if key.endswith('_ID'):
            seg_name = key.replace('_ID', '')
            seg_id = reference_header[key]
            color_key = f"{seg_name}_Color"
            if color_key in reference_header:
                ref_segment_colors[seg_id] = reference_header[color_key]

print(f"ref_segment_colors字典大小: {len(ref_segment_colors)}")
print(f"前3个条目:")
for seg_id, color in list(ref_segment_colors.items())[:3]:
    print(f"  Key: {repr(seg_id)} -> Color: {color}")

print("\n" + "="*60 + "\n")

# 3. 模拟处理mapped_data中的label
print("=== 模拟label_id查找 ===\n")

# 假设我们有这些label_id (整数)
test_label_ids = [1, 2, 12, 13, 99]

for label_id in test_label_ids:
    # 错误的方式(修复前)
    old_found = label_id in ref_segment_colors
    
    # 正确的方式(修复后)
    seg_id_str = f"Segment_{label_id}"
    new_found = seg_id_str in ref_segment_colors
    
    print(f"label_id={label_id}:")
    print(f"  旧方式(直接查找int): {old_found}")
    print(f"  新方式(转为Segment_{label_id}): {new_found}")
    if new_found:
        print(f"  → 找到颜色: {ref_segment_colors[seg_id_str]}")
    print()

print("="*60 + "\n")

# 4. 验证其他可能的类型问题
print("=== 检查其他潜在的类型问题 ===\n")

# 检查label_extents字典
from skimage.measure import regionprops

# 创建测试数据
test_data = np.zeros((100, 100, 100), dtype=np.int32)
test_data[10:20, 10:20, 10:20] = 1
test_data[30:40, 30:40, 30:40] = 12

props = regionprops(test_data)
label_extents = {}
for prop in props:
    bbox = prop.bbox
    extent = f"{bbox[0]} {bbox[3]-1} {bbox[1]} {bbox[4]-1} {bbox[2]} {bbox[5]-1}"
    label_extents[prop.label] = extent

print(f"label_extents字典:")
for label, extent in label_extents.items():
    print(f"  Key: {repr(label)} (type: {type(label).__name__}) -> Extent: {extent}")

print("\n查找测试:")
for test_id in [1, 12, 99]:
    found = test_id in label_extents
    print(f"  {test_id} in label_extents: {found}")

print("\n✅ 所有类型检查完成!")
