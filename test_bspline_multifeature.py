"""快速测试 B-spline 配准 + 多特征追踪"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fxn_register_bspline import process_well_bspline
from fxn_track_multifeature import process_well_multifeature
import fxn_config as cfg
import os

def test_well(well_name):
    bspline_dir = os.path.join(cfg.OUTPUT_DIR, 'bspline_seg')
    tracking_dir = os.path.join(cfg.OUTPUT_DIR, 'multifeature_tracking')

    print(f"\n{'='*70}")
    print(f"TEST WELL: {well_name}")
    print(f"{'='*70}")

    # Step 1: B-spline registration
    print("\n--- Step 1: B-spline Registration ---")
    success = process_well_bspline(well_name, bspline_dir)
    if not success:
        print("Registration failed!")
        return

    # Step 2: Multi-feature tracking
    print("\n--- Step 2: Multi-feature Tracking ---")
    day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, f'{well_name}.nii.gz')
    day5_path = os.path.join(bspline_dir, f'{well_name}_day5_bspline.nii.gz')
    df_day3, df_day5, logs = process_well_multifeature(well_name, day3_path, day5_path, tracking_dir)

    if df_day3 is not None:
        n_matched = sum(1 for l in logs if l['Status'] == 'Matched')
        n_new = sum(1 for l in logs if l['Status'] == 'New')
        total = n_matched + n_new
        rate = n_matched / total * 100 if total > 0 else 0
        print(f"\n>>> RESULT for {well_name}:")
        print(f"    Day3 count: {len(df_day3)}, Day5 count: {len(df_day5)}")
        print(f"    Matched: {n_matched}, New: {n_new}")
        print(f"    Match rate: {rate:.1f}%")

if __name__ == '__main__':
    # Test Control group
    test_well('F2_1')
    # Test drug group (40uM)
    test_well('B5_1')
