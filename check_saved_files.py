"""
检查保存的nrrd文件是否有数据
"""
import nrrd
import numpy as np

files = [
    'results/day1_reference.nrrd',
    'results/day2_tracked_only.nrrd',
    'results/day2_trackpy_matched.nrrd',
    'results/day3_tracked_only.nrrd',
    'results/day3_trackpy_matched.nrrd'
]

print("="*70)
print("检查保存的文件")
print("="*70)

for filepath in files:
    try:
        data, header = nrrd.read(filepath)
        
        print(f"\n【{filepath}】")
        print(f"  数据形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数值范围: [{data.min()}, {data.max()}]")
        print(f"  非零体素数: {np.count_nonzero(data)}")
        
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]
        print(f"  标签数量: {len(unique_labels)}")
        print(f"  标签ID: {sorted(unique_labels.tolist())}")
        
        # 检查header关键信息
        print(f"  Header keys: {len(header.keys())} 个")
        if 'space' in header:
            print(f"  Space: {header['space']}")
        if 'space origin' in header:
            print(f"  Space origin: {header['space origin']}")
        if 'sizes' in header:
            print(f"  Sizes: {header['sizes']}")
            
        # 检查是否有Segment信息
        segment_keys = [k for k in header.keys() if k.startswith('Segment')]
        print(f"  Segment元数据: {len(segment_keys)} 个key")
        
        if len(unique_labels) == 0:
            print(f"  ❌ 警告: 文件没有任何标签数据!")
        else:
            print(f"  ✓ 文件有效")
            
    except Exception as e:
        print(f"\n【{filepath}】")
        print(f"  ❌ 读取失败: {e}")

print("\n" + "="*70)
