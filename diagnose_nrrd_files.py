"""
详细检查NRRD文件的所有信息
"""
import nrrd
import numpy as np
import json

files = [
    'results/day1_reference.seg.nrrd',
    'results/day2_tracked_only.seg.nrrd',
    'results/day3_tracked_only.seg.nrrd'
]

print("="*80)
print("NRRD文件详细信息检查")
print("="*80)

for filepath in files:
    print("\n" + "="*80)
    print(f"文件: {filepath}")
    print("="*80)
    
    try:
        data, header = nrrd.read(filepath)
        
        print("\n【数据信息】")
        print(f"  数据形状 (shape):        {data.shape}")
        print(f"  数据类型 (dtype):        {data.dtype}")
        print(f"  数值范围 (min-max):      [{data.min()}, {data.max()}]")
        print(f"  非零体素数:              {np.count_nonzero(data):,}")
        print(f"  总体素数:                {data.size:,}")
        print(f"  非零占比:                {np.count_nonzero(data)/data.size*100:.2f}%")
        
        # 统计每个label的体素数
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]
        print(f"\n  标签数量:                {len(unique_labels)}")
        print(f"  标签ID列表:              {sorted(unique_labels.tolist())}")
        
        if len(unique_labels) > 0:
            print(f"\n  各标签体素统计:")
            for label in sorted(unique_labels)[:10]:  # 只显示前10个
                count = np.sum(data == label)
                print(f"    Label {int(label):4d}: {count:8,} 体素")
            if len(unique_labels) > 10:
                print(f"    ... (还有 {len(unique_labels)-10} 个标签)")
        
        print("\n【Header完整信息】")
        print(f"  Header包含 {len(header.keys())} 个key")
        
        # 按类别显示header
        basic_keys = ['type', 'dimension', 'space', 'sizes', 'space directions', 
                      'kinds', 'endian', 'encoding', 'space origin']
        segment_keys = [k for k in header.keys() if k.startswith('Segment')]
        other_keys = [k for k in header.keys() if k not in basic_keys and not k.startswith('Segment')]
        
        print("\n  --- 基本信息 ---")
        for key in basic_keys:
            if key in header:
                value = header[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                print(f"    {key:25s} = {value}")
        
        if other_keys:
            print("\n  --- 其他元数据 ---")
            for key in other_keys:
                value = header[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"    {key:40s} = {value_str}")
        
        print(f"\n  --- Segment元数据 ({len(segment_keys)} 个key) ---")
        # 按Segment编号分组
        segment_dict = {}
        for key in segment_keys:
            parts = key.split('_', 1)
            if len(parts) == 2:
                seg_id = parts[0]
                prop = parts[1]
                if seg_id not in segment_dict:
                    segment_dict[seg_id] = {}
                segment_dict[seg_id][prop] = header[key]
        
        # 显示每个Segment的信息
        for seg_id in sorted(segment_dict.keys(), key=lambda x: int(x.replace('Segment', '')))[:5]:
            print(f"\n    {seg_id}:")
            seg_info = segment_dict[seg_id]
            for prop in ['ID', 'Name', 'LabelValue', 'Color', 'Layer', 'Extent']:
                if prop in seg_info:
                    value = seg_info[prop]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    print(f"      {prop:15s} = {value}")
        
        if len(segment_dict) > 5:
            print(f"\n    ... (还有 {len(segment_dict)-5} 个Segment)")
        
        # 检查关键问题
        print("\n【诊断信息】")
        
        # 1. 检查space origin
        if 'space origin' in header:
            origin = header['space origin']
            print(f"  ✓ Space origin:          {origin}")
            if np.all(origin == 0):
                print(f"    ⚠️  注意: space origin为(0,0,0),可能影响3D Slicer显示")
        
        # 2. 检查space directions
        if 'space directions' in header:
            dirs = header['space directions']
            print(f"  ✓ Space directions:      形状 {dirs.shape}")
            print(f"    值: {dirs.tolist()}")
        
        # 3. 检查Segment Extent
        extents_all_zero = True
        if segment_dict:
            first_seg = list(segment_dict.values())[0]
            if 'Extent' in first_seg:
                extent = first_seg['Extent']
                if not np.all(np.array(extent) == 0):
                    extents_all_zero = False
        
        if extents_all_zero and len(segment_dict) > 0:
            print(f"  ❌ 问题: 所有Segment的Extent都是 [0 0 0 0 0 0]")
            print(f"     这会导致3D Slicer认为segment没有空间范围!")
        else:
            print(f"  ✓ Segment Extent正常")
        
        # 4. 检查数据是否实际有内容
        if np.count_nonzero(data) == 0:
            print(f"  ❌ 致命问题: 数据全为0,没有任何标签!")
        elif len(unique_labels) == 0:
            print(f"  ❌ 致命问题: 没有找到任何非零标签!")
        else:
            print(f"  ✓ 数据包含 {len(unique_labels)} 个标签")
        
        # 5. 检查Segment元数据数量
        expected_segments = len(unique_labels)
        actual_segments = len(segment_dict)
        if actual_segments != expected_segments:
            print(f"  ⚠️  Segment元数据数量({actual_segments})与实际标签数量({expected_segments})不匹配")
        else:
            print(f"  ✓ Segment元数据数量与标签数量匹配")
        
    except Exception as e:
        print(f"\n  ❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("检查完成")
print("="*80)
