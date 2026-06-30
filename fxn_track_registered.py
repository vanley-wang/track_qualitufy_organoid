"""
FXN 类器官追踪与可视化（配准后版本）
对配准后的 FXN_0701 (Day3) 和 registered_seg (Day5) 进行追踪

运行方式:
    python fxn_track_registered.py
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fxn_track_and_visualize import (
    process_well, generate_summary_plots
)
import fxn_config as cfg


def main():
    print("=" * 70)
    print("FXN 类器官追踪与可视化 (配准后)")
    print("Day3 (FXN_0701_seg) → Day5 (registered_seg)")
    print("=" * 70)
    cfg.print_config()

    registered_dir = os.path.join(cfg.OUTPUT_DIR, 'registered_seg')
    output_dir = os.path.join(cfg.OUTPUT_DIR, 'tracked_registered')
    os.makedirs(output_dir, exist_ok=True)

    # 获取 well 列表
    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 Day3 seg 文件")
    print(f"配准后 Day5 seg 目录: {registered_dir}")
    print(f"追踪输出目录: {output_dir}")

    all_quant_records = []
    all_log_records = []
    well_stats = []

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, seg_file)
        day5_path = os.path.join(registered_dir, f'{well_name}_day5_registered.nii.gz')

        if not os.path.exists(day5_path):
            print(f"\n⚠️  跳过 {well_name}: 配准后的 Day5 文件不存在 ({day5_path})")
            print(f"    请先运行 fxn_register.py 进行配准！")
            continue

        df_day3, df_day5, logs = process_well(well_name, day3_path, day5_path, output_dir, generate_plots=False)

        if df_day3 is not None and df_day5 is not None:
            df_day3['Well'] = well_name
            df_day5['Well'] = well_name
            all_quant_records.append(df_day3)
            all_quant_records.append(df_day5)

            n_matched = sum(1 for l in logs if l['Status'] == 'Matched')
            n_new = sum(1 for l in logs if l['Status'] == 'New')
            well_stats.append({
                'Well': well_name,
                'Day3_Count': len(df_day3),
                'Day5_Count': len(df_day5),
                'Matched': n_matched,
                'New': n_new,
            })

        all_log_records.extend(logs)

    # 保存全局 CSV
    if all_quant_records:
        df_all_quant = pd.concat(all_quant_records, ignore_index=True)
        df_all_quant.to_csv(os.path.join(output_dir, 'fxn_quantification_registered.csv'), index=False)
        print(f"\n✓ 全局量化数据: {os.path.join(output_dir, 'fxn_quantification_registered.csv')}")

    if all_log_records:
        df_all_log = pd.DataFrame(all_log_records)
        df_all_log.to_csv(os.path.join(output_dir, 'fxn_matching_log_registered.csv'), index=False)
        print(f"✓ 全局匹配日志: {os.path.join(output_dir, 'fxn_matching_log_registered.csv')}")

    if well_stats:
        df_stats = pd.DataFrame(well_stats)
        df_stats.to_csv(os.path.join(output_dir, 'fxn_well_summary_registered.csv'), index=False)
        print(f"✓ Well 汇总: {os.path.join(output_dir, 'fxn_well_summary_registered.csv')}")

    # 全局可视化
    if all_quant_records and all_log_records:
        print("\n生成全局汇总图表...")
        generate_summary_plots(df_all_quant, df_all_log, output_dir)

    print("\n" + "=" * 70)
    print("配准后追踪完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
