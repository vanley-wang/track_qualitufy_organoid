"""快速测试：只处理 B2_1 一个 well，验证流水线"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fxn_track_and_visualize import process_well
import fxn_config as cfg
import os

well_name = 'B2_1'
day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, f'{well_name}.nii.gz')
day5_path = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

print(f"Testing well: {well_name}")
print(f"Day3 path: {day3_path}")
print(f"Day5 path: {day5_path}")

df_day3, df_day5, logs = process_well(well_name, day3_path, day5_path, cfg.OUTPUT_DIR)

if df_day3 is not None:
    print(f"\nDay3 quantified: {len(df_day3)} organoids")
    print(df_day3.head())
if df_day5 is not None:
    print(f"\nDay5 quantified: {len(df_day5)} organoids")
    print(df_day5.head())
print(f"\nLog entries: {len(logs)}")
