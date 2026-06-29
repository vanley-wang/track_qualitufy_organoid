"""
详细检查Segment元数据的ID和Name
"""
import nrrd
import numpy as np

filepath = 'results/day2_tracked_only.seg.nrrd'

print("="*80)
print(f"检查文件: {filepath}")
print("="*80)

data, header = nrrd.read(filepath)
unique_labels = sorted([int(x) for x in np.unique(data) if x > 0])

print(f"\n数据中实际的label ID: {unique_labels}")
print(f"总共: {len(unique_labels)} 个label")

print("\n" + "="*80)
print("Segment元数据详情:")
print("="*80)

# 收集所有Segment信息
segment_info = {}
for key in sorted(header.keys()):
    if key.startswith('Segment'):
        parts = key.split('_', 1)
        if len(parts) == 2:
            seg_name = parts[0]  # Segment0, Segment1, etc.
            prop_name = parts[1]  # ID, Name, Color, etc.
            
            if seg_name not in segment_info:
                segment_info[seg_name] = {}
            segment_info[seg_name][prop_name] = header[key]

# 显示每个Segment
for seg_name in sorted(segment_info.keys(), key=lambda x: int(x.replace('Segment', ''))):
    info = segment_info[seg_name]
    print(f"\n{seg_name}:")
    for prop in ['ID', 'Name', 'LabelValue', 'Color', 'Layer', 'Extent']:
        if prop in info:
            value = info[prop]
            # 检查类型
            print(f"  {prop:15s} = {repr(value):50s} (type: {type(value).__name__})")

print("\n" + "="*80)
print("问题诊断:")
print("="*80)

# 检查ID格式问题
print("\n【检查ID字段】")
for seg_name in sorted(segment_info.keys(), key=lambda x: int(x.replace('Segment', '')))[:5]:
    if 'ID' in segment_info[seg_name]:
        id_value = segment_info[seg_name]['ID']
        label_value = segment_info[seg_name].get('LabelValue', 'N/A')
        
        # 3D Slicer期望的格式
        print(f"\n{seg_name}:")
        print(f"  当前ID:        {repr(id_value)}")
        print(f"  LabelValue:    {repr(label_value)}")
        
        # 检查是否是正确的格式
        if isinstance(id_value, str):
            if id_value.startswith('Segment_'):
                print(f"  ✓ ID格式正确: {id_value}")
            else:
                print(f"  ❌ ID格式错误! 应该是 'Segment_X' 格式,而不是 '{id_value}'")
                print(f"     3D Slicer需要ID字段为 'Segment_数字' 格式")
        else:
            print(f"  ❌ ID类型错误! 应该是字符串,而不是 {type(id_value).__name__}")

print("\n" + "="*80)
print("解决方案:")
print("="*80)
print("ID字段应该是 'Segment_1', 'Segment_2' 等格式")
print("当前可能设置成了纯数字,导致3D Slicer只识别第一个")
print("="*80)
