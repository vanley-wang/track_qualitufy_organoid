"""
检查NRRD文件中的Segment名称和label ID的对应关系
"""
import nrrd
import numpy as np

def print_segment_info(filepath, day_name):
    """打印文件中的segment信息"""
    data, header = nrrd.read(filepath)
    
    print(f"\n{'='*70}")
    print(f"{day_name}: {filepath}")
    print(f"{'='*70}")
    
    # 获取实际存在的label
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景0
    
    print(f"\n实际数据中的label ID: {sorted(unique_labels)}")
    print(f"总共: {len(unique_labels)} 个类器官")
    
    # 检查header中的Segment信息
    print(f"\nHeader中的Segment信息:")
    print("-" * 70)
    
    segment_count = 0
    segment_info = []
    
    for key in sorted(header.keys()):
        if key.startswith('Segment') and '_ID' in key:
            seg_name = key.replace('_ID', '')
            seg_id = header[key]
            
            # 获取其他信息
            color_key = f"{seg_name}_Color"
            name_key = f"{seg_name}_Name"
            
            color = header.get(color_key, 'N/A')
            name = header.get(name_key, 'N/A')
            
            # 检查这个ID是否在实际数据中存在
            exists = seg_id in unique_labels
            
            segment_info.append({
                'segment': seg_name,
                'id': seg_id,
                'name': name,
                'exists': exists
            })
            
            segment_count += 1
    
    # 按ID排序并显示
    segment_info.sort(key=lambda x: int(x['segment'].replace('Segment', '').replace('_', '')))
    
    print(f"{'Segment名称':<15} {'Label ID':<10} {'自定义名称':<20} {'数据中存在'}")
    print("-" * 70)
    
    for info in segment_info:
        exists_mark = '✓' if info['exists'] else '✗'
        print(f"{info['segment']:<15} {info['id']:<10} {info['name']:<20} {exists_mark}")
    
    print(f"\nHeader中定义了 {segment_count} 个Segment")
    
    # 检查是否有数据中的label在header里没有定义
    header_ids = set([info['id'] for info in segment_info])
    missing_in_header = set(unique_labels) - header_ids
    
    if missing_in_header:
        print(f"\n⚠️ 警告: 以下label在数据中存在但header中没有定义:")
        print(f"   {sorted(missing_in_header)}")
    
    return segment_info, unique_labels


# 检查所有文件
print("="*70)
print("检查Segment名称和Label ID的对应关系")
print("="*70)

files = [
    ('results/day1_reference.nrrd', 'Day1 参考'),
    ('results/day2_trackpy_matched.nrrd', 'Day2 追踪后'),
    ('results/day3_trackpy_matched.nrrd', 'Day3 追踪后')
]

all_info = {}
for filepath, day_name in files:
    try:
        info, labels = print_segment_info(filepath, day_name)
        all_info[day_name] = {'info': info, 'labels': labels}
    except Exception as e:
        print(f"\n错误: 无法读取 {filepath}")
        print(f"   {e}")

# 特别检查你提到的label
print("\n" + "="*70)
print("特别检查你提到的例子")
print("="*70)

check_labels = [
    ('Day1 参考', 2, 'Day1的label 2'),
    ('Day2 追踪后', 26, 'Day2的label 26 (应该对应Day1的label 2)'),
    ('Day3 追踪后', 2, 'Day3的label 2'),
    ('Day1 参考', 14, 'Day1的label 14'),
    ('Day2 追踪后', 29, 'Day2的label 29 (应该对应Day1的label 14)'),
    ('Day3 追踪后', 14, 'Day3的label 14'),
]

print("\n检查特定label是否存在:")
print("-" * 70)
for day_name, label_id, description in check_labels:
    if day_name in all_info:
        exists = label_id in all_info[day_name]['labels']
        status = '✓ 存在' if exists else '✗ 不存在'
        print(f"{description:<40} → {status}")
        
        if exists:
            # 找到对应的Segment名称
            for info in all_info[day_name]['info']:
                if info['id'] == label_id:
                    print(f"  → 在3D Slicer中显示为: {info['segment']}")
                    if info['name'] != 'N/A':
                        print(f"  → 自定义名称: {info['name']}")
                    break

print("\n" + "="*70)
print("总结")
print("="*70)
print("✓ 在3D Slicer中,左侧显示的是 'Segment_X' 名称")
print("✓ 但实际的label ID可能和Segment编号不一致")
print("✓ 例如: Segment_5 的实际label ID可能是 26")
print("✓ 要找label 26,需要在列表中逐个展开Segment,查看其ID")
print("="*70)
