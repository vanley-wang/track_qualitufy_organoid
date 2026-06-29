"""
详细诊断为什么3D Slicer只显示一个segment
"""
import nrrd
import numpy as np

print("="*80)
print("3D Slicer 显示问题诊断")
print("="*80)

filepath = 'results/day2_tracked_only.seg.nrrd'
print(f"\n检查文件: {filepath}")

data, header = nrrd.read(filepath)

# 1. 检查数据
print("\n【1. 数据检查】")
unique_labels = sorted(np.unique(data)[1:])  # 排除0
print(f"  实际label数量: {len(unique_labels)}")
print(f"  Label值: {[int(x) for x in unique_labels]}")

for label in unique_labels[:5]:
    count = np.sum(data == label)
    print(f"    Label {int(label):2d}: {count:8,} 体素")

# 2. 检查Segment元数据完整性
print("\n【2. Segment元数据检查】")
segment_dict = {}
for key in header.keys():
    if key.startswith('Segment') and '_' in key:
        parts = key.split('_', 1)
        seg_name = parts[0]
        prop = parts[1]
        if seg_name not in segment_dict:
            segment_dict[seg_name] = {}
        segment_dict[seg_name][prop] = header[key]

print(f"  找到 {len(segment_dict)} 个Segment定义")

# 检查前5个segment的完整性
for seg_name in sorted(segment_dict.keys(), key=lambda x: int(x.replace('Segment', '')))[:5]:
    seg_info = segment_dict[seg_name]
    print(f"\n  {seg_name}:")
    required_fields = ['ID', 'Name', 'LabelValue', 'Color', 'Layer', 'Extent']
    for field in required_fields:
        if field in seg_info:
            value = seg_info[field]
            if field == 'Extent':
                print(f"    ✓ {field:15s} = {value}")
            elif field == 'Color':
                print(f"    ✓ {field:15s} = {value}")
            else:
                print(f"    ✓ {field:15s} = {value}")
        else:
            print(f"    ❌ {field:15s} = 缺失!")

# 3. 检查可能的问题
print("\n【3. 可能导致只显示一个segment的问题】")

issues = []

# 检查LabelValue是否正确
print("\n  检查LabelValue匹配:")
for seg_name in sorted(segment_dict.keys(), key=lambda x: int(x.replace('Segment', '')))[:3]:
    seg_info = segment_dict[seg_name]
    label_value = int(seg_info['LabelValue'])
    if label_value in unique_labels:
        print(f"    ✓ {seg_name} LabelValue={label_value} 在数据中存在")
    else:
        print(f"    ❌ {seg_name} LabelValue={label_value} 在数据中不存在!")
        issues.append(f"LabelValue {label_value} 不匹配")

# 检查Extent是否全为0
print("\n  检查Extent:")
extents_ok = 0
extents_zero = 0
for seg_name, seg_info in segment_dict.items():
    if 'Extent' in seg_info:
        extent = seg_info['Extent']
        if isinstance(extent, str):
            parts = extent.split()
            if len(parts) == 6 and not all(p == '0' for p in parts):
                extents_ok += 1
            else:
                extents_zero += 1

print(f"    ✓ {extents_ok} 个segment的Extent有效")
if extents_zero > 0:
    print(f"    ❌ {extents_zero} 个segment的Extent全为0")
    issues.append("某些Extent全为0")

# 检查Color是否重复
print("\n  检查Color:")
colors = {}
for seg_name, seg_info in segment_dict.items():
    if 'Color' in seg_info:
        color = seg_info['Color']
        if color not in colors:
            colors[color] = []
        colors[color].append(seg_name)

duplicate_colors = {c: segs for c, segs in colors.items() if len(segs) > 1}
if duplicate_colors:
    print(f"    ⚠️  发现重复颜色:")
    for color, segs in list(duplicate_colors.items())[:3]:
        print(f"      颜色 {color}: {len(segs)} 个segment使用")
else:
    print(f"    ✓ 所有颜色唯一")

# 4. 检查与Day1的对应关系
print("\n【4. 与Day1参考的对应检查】")
data1, header1 = nrrd.read('results/day1_reference.seg.nrrd')

# 提取Day1的颜色映射
day1_colors = {}
for key in header1.keys():
    if key.endswith('_ID'):
        seg_name = key.replace('_ID', '')
        seg_id = header1[key]
        color_key = f"{seg_name}_Color"
        if color_key in header1:
            day1_colors[seg_id] = header1[color_key]

print(f"  Day1有 {len(day1_colors)} 个segment定义")

# 检查Day2的颜色是否来自Day1
matching_colors = 0
for seg_name, seg_info in segment_dict.items():
    if 'ID' in seg_info and 'Color' in seg_info:
        seg_id = seg_info['ID']
        seg_color = seg_info['Color']
        if seg_id in day1_colors:
            if day1_colors[seg_id] == seg_color:
                matching_colors += 1

print(f"  ✓ {matching_colors}/{len(segment_dict)} 个segment颜色与Day1匹配")

# 5. 生成NRRD文本查看前100行
print("\n【5. 生成NRRD文本header供人工检查】")
print("\n保存header到 debug_header.txt ...")
with open('debug_header.txt', 'w') as f:
    f.write("NRRD Header for day2_tracked_only.seg.nrrd\n")
    f.write("="*80 + "\n\n")
    
    # 基本信息
    f.write("Basic Information:\n")
    f.write("-"*80 + "\n")
    for key in ['type', 'dimension', 'space', 'sizes', 'space directions', 
                'space origin', 'encoding']:
        if key in header:
            f.write(f"{key}: {header[key]}\n")
    
    f.write("\n\nSegment Metadata:\n")
    f.write("-"*80 + "\n")
    
    # 按segment排序输出
    for seg_name in sorted(segment_dict.keys(), key=lambda x: int(x.replace('Segment', ''))):
        seg_info = segment_dict[seg_name]
        f.write(f"\n{seg_name}:\n")
        for prop in ['ID', 'Name', 'LabelValue', 'Color', 'Layer', 'Extent']:
            if prop in seg_info:
                f.write(f"  {prop}: {seg_info[prop]}\n")

print("  ✓ 已保存到 debug_header.txt")

# 6. 生成测试用的简化文件
print("\n【6. 生成测试用简化文件】")
print("  创建只包含前3个label的测试文件...")

# 复制数据,只保留前3个label
test_data = np.zeros_like(data)
test_labels = sorted(unique_labels)[:3]
for label in test_labels:
    test_data[data == label] = label

# 创建简化的header
test_header = header.copy()

# 删除所有segment信息
keys_to_delete = [k for k in test_header.keys() if k.startswith('Segment')]
for key in keys_to_delete:
    del test_header[key]

# 只添加前3个segment
for idx, label in enumerate(test_labels):
    seg_prefix = f"Segment{idx}"
    label_int = int(label)
    
    # 从原header中找到这个label的信息
    original_seg = None
    for seg_name, seg_info in segment_dict.items():
        if 'LabelValue' in seg_info and int(seg_info['LabelValue']) == label_int:
            original_seg = seg_info
            break
    
    if original_seg:
        test_header[f"{seg_prefix}_ID"] = original_seg.get('ID', f'Segment_{label_int}')
        test_header[f"{seg_prefix}_Name"] = original_seg.get('Name', f'Segment_{label_int}')
        test_header[f"{seg_prefix}_LabelValue"] = str(label_int)
        test_header[f"{seg_prefix}_Color"] = original_seg.get('Color', '0.9 0.2 0.2')
        test_header[f"{seg_prefix}_Layer"] = original_seg.get('Layer', '0')
        test_header[f"{seg_prefix}_Extent"] = original_seg.get('Extent', '0 799 0 511 0 799')

# 保存测试文件
nrrd.write('test_3_segments.seg.nrrd', test_data.astype(np.int16), test_header)
print("  ✓ 已保存: test_3_segments.seg.nrrd")
print("    包含label:", [int(x) for x in test_labels])
print("    请在3D Slicer中打开这个测试文件")
print("    如果这个文件能正常显示3个segment,说明原文件segment太多")

print("\n" + "="*80)
print("诊断完成!")
print("="*80)
print("\n请检查:")
print("  1. debug_header.txt 文件中的header信息")
print("  2. 在3D Slicer中打开 test_3_segments.seg.nrrd")
print("  3. 如果测试文件正常,可能是segment数量太多导致显示问题")
print("="*80)
