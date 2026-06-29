"""分析匹配距离,建议合适的阈值"""
import re

log_file = "id_tracking_correct.log"

with open(log_file, 'r') as f:
    content = f.read()

# 提取Day2的距离
day2_section = re.search(r'匹配 Day2.*?匹配结果:', content, re.DOTALL)
if day2_section:
    distances_day2 = re.findall(r'(\d+)\s+.*?\s+([\d.]+)\s+.*?(Matched|Too far)', day2_section.group())
    
    print("="*60)
    print("Day2匹配距离分析")
    print("="*60)
    
    matched = []
    too_far = []
    
    for target_id, dist, status in distances_day2:
        dist_val = float(dist)
        if 'Matched' in status:
            matched.append(dist_val)
        else:
            too_far.append(dist_val)
    
    print(f"\n当前阈值: 50 pixels (250 μm)")
    print(f"匹配成功: {len(matched)} 个")
    if matched:
        print(f"  距离范围: {min(matched):.2f} - {max(matched):.2f} pixels")
    
    print(f"\n未匹配 (Too far): {len(too_far)} 个")
    if too_far:
        too_far_sorted = sorted(too_far)
        print(f"  距离范围: {min(too_far):.2f} - {max(too_far):.2f} pixels")
        print(f"  距离分布:")
        print(f"    < 60:  {sum(1 for d in too_far if d < 60)} 个")
        print(f"    60-70: {sum(1 for d in too_far if 60 <= d < 70)} 个")
        print(f"    70-80: {sum(1 for d in too_far if 70 <= d < 80)} 个")
        print(f"    80-90: {sum(1 for d in too_far if 80 <= d < 90)} 个")
        print(f"    > 90:  {sum(1 for d in too_far if d >= 90)} 个")
        
        print(f"\n推荐阈值:")
        for threshold in [60, 70, 80, 90, 100]:
            would_match = sum(1 for d in too_far if d <= threshold)
            total_match = len(matched) + would_match
            print(f"  {threshold} pixels ({threshold*5} μm): 总共匹配 {total_match}/{len(distances_day2)} 个")

# Day3分析
day3_section = re.search(r'匹配 Day3.*?匹配结果:', content, re.DOTALL)
if day3_section:
    distances_day3 = re.findall(r'(\d+)\s+.*?\s+([\d.]+)\s+.*?(Matched|Too far)', day3_section.group())
    
    print("\n" + "="*60)
    print("Day3匹配距离分析")
    print("="*60)
    
    matched = []
    too_far = []
    
    for target_id, dist, status in distances_day3:
        dist_val = float(dist)
        if 'Matched' in status:
            matched.append(dist_val)
        else:
            too_far.append(dist_val)
    
    print(f"\n当前阈值: 50 pixels (250 μm)")
    print(f"匹配成功: {len(matched)} 个")
    if matched:
        print(f"  距离范围: {min(matched):.2f} - {max(matched):.2f} pixels")
    
    print(f"\n未匹配 (Too far): {len(too_far)} 个")
    if too_far:
        print(f"  距离范围: {min(too_far):.2f} - {max(too_far):.2f} pixels")
