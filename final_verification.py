"""
最终验证 - 确认文件可以在3D Slicer中正确加载
"""
import nrrd
import numpy as np

print("="*80)
print("最终验证: 3D Slicer加载检查")
print("="*80)

files = [
    ('Day1参考', 'results/day1_reference.seg.nrrd'),
    ('Day2纯追踪', 'results/day2_tracked_only.seg.nrrd'),
    ('Day3纯追踪', 'results/day3_tracked_only.seg.nrrd'),
]

all_good = True

for name, filepath in files:
    print(f"\n【{name}: {filepath}】")
    
    try:
        data, header = nrrd.read(filepath)
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]
        
        # 检查项
        checks = []
        
        # 1. 数据非空
        if np.count_nonzero(data) > 0:
            checks.append(("✓", f"数据非空: {np.count_nonzero(data):,} 体素"))
        else:
            checks.append(("❌", "数据为空!"))
            all_good = False
        
        # 2. 有标签
        if len(unique_labels) > 0:
            checks.append(("✓", f"包含标签: {len(unique_labels)} 个"))
        else:
            checks.append(("❌", "没有标签!"))
            all_good = False
        
        # 3. 空间信息完整
        if all(k in header for k in ['space', 'space origin', 'space directions']):
            checks.append(("✓", f"空间信息完整"))
        else:
            checks.append(("❌", "缺少空间信息!"))
            all_good = False
        
        # 4. Segment元数据
        segment_keys = [k for k in header.keys() if k.startswith('Segment') and k.endswith('_Extent')]
        if len(segment_keys) > 0:
            checks.append(("✓", f"Segment元数据: {len(segment_keys)} 个"))
            
            # 检查Extent是否有效
            first_extent_key = segment_keys[0]
            extent_value = header[first_extent_key]
            if isinstance(extent_value, str):
                extent_parts = extent_value.strip().split()
                if len(extent_parts) == 6 and not all(p == '0' for p in extent_parts):
                    checks.append(("✓", f"Extent有效: {extent_value}"))
                else:
                    checks.append(("❌", f"Extent全为0: {extent_value}"))
                    all_good = False
            else:
                checks.append(("❌", f"Extent格式错误"))
                all_good = False
        else:
            checks.append(("❌", "没有Segment元数据!"))
            all_good = False
        
        # 5. 文件扩展名
        if filepath.endswith('.seg.nrrd'):
            checks.append(("✓", "文件扩展名: .seg.nrrd"))
        else:
            checks.append(("⚠️", "文件扩展名不是.seg.nrrd"))
        
        # 输出检查结果
        for status, message in checks:
            print(f"  {status} {message}")
        
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        all_good = False

print("\n" + "="*80)
if all_good:
    print("✅ 所有文件通过验证! 可以在3D Slicer中加载了")
    print("\n【在3D Slicer中的操作步骤】")
    print("  1. 打开3D Slicer")
    print("  2. 将这3个文件拖拽到Slicer窗口:")
    print("     - results/day1_reference.seg.nrrd")
    print("     - results/day2_tracked_only.seg.nrrd")
    print("     - results/day3_tracked_only.seg.nrrd")
    print("  3. 文件会自动识别为Segmentation(有颜色的)")
    print("  4. 在左侧Data模块,展开每个Segmentation节点")
    print("  5. 相同编号的Segment(如Segment_2)会显示相同颜色")
    print("  6. 点击眼睛图标切换可见性")
    print("  7. 点击'3D'按钮查看3D渲染")
else:
    print("❌ 发现问题! 请检查上面的错误信息")
print("="*80)
