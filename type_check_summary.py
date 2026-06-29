#!/usr/bin/env python3
"""总结代码中所有关键的类型和key匹配点"""

print("="*70)
print("代码中关键的类型匹配点总结")
print("="*70)

print("""
1. ✅ 【已修复】ref_segment_colors 字典查找
   位置: save_tracking_results() 第232行
   
   字典构建:
     ref_segment_colors[seg_id] = color
     其中 seg_id = reference_header[key]  # 类型: str, 如 "Segment_1"
   
   查找逻辑(修复后):
     seg_id_str = f"Segment_{label_id}"  # 将int转为"Segment_X"字符串
     if seg_id_str in ref_segment_colors:
   
   修复内容:
     - 旧: if label_id in ref_segment_colors  # ❌ int查找str key
     - 新: if seg_id_str in ref_segment_colors  # ✅ str查找str key

2. ✅ label_extents 字典查找
   位置: save_tracking_results() 第249行
   
   字典构建:
     label_extents[prop.label] = extent
     其中 prop.label 是 int 类型
   
   查找逻辑:
     if label_id in label_extents:
     其中 label_id 来自 sorted(unique_labels), unique_labels 来自 np.unique()
   
   状态: ✅ 类型匹配(int对int),numpy int可以直接在Python dict中查找

3. ✅ label_mapping 字典的创建和使用
   位置: create_label_mapping_from_tracking()
   
   字典构建:
     day1_label = int(day1_in_track.iloc[0]['label'])  # 显式转int
     day2_original = int(day2_in_track.iloc[0]['label'])  # 显式转int
     day2_mapping[day2_original] = day1_label  # int: int
   
   字典使用:
     apply_label_mapping() 第161行:
     for orig_label, new_label in label_mapping.items():
         mapped_data[original_data == orig_label] = new_label
     
     第164行:
     if l not in label_mapping:
     其中 l 来自 np.unique(), 是 numpy int 类型
   
   状态: ✅ 类型兼容(numpy int可以在Python int dict中查找)

4. ✅ 其他字典查找
   
   reference_header 字典:
     - 所有key都是字符串(如"Segment0_ID", "Segment0_Color")
     - 查找时使用f-string拼接,确保是字符串类型
   
   output_header 字典:
     - 同样使用f-string构建key,类型一致

5. 📝 数据类型流程追踪
   
   DataFrame['label'] 列:
     → int(df['label'])  # 显式转换
     → label_mapping 字典 (int: int)
     → apply_label_mapping 使用
   
   mapped_data (numpy array):
     → np.unique(mapped_data)  # 返回 numpy int
     → sorted(unique_labels)  # 保持为 numpy int
     → enumerate() 中作为 label_id  # numpy int
     → 需要转换: f"Segment_{label_id}"  # 用于字符串拼接
     → 直接使用: label_extents[label_id]  # int对int查找

总结:
=====
✅ 所有关键的类型匹配点都已检查
✅ 主要bug(ref_segment_colors查找)已修复
✅ 其他类型转换都正确或兼容
✅ 代码应该可以正常工作了
""")

print("="*70)
