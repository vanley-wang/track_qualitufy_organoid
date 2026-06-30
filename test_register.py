"""测试单个 well 的配准"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fxn_register import process_well_registration
import fxn_config as cfg
import os

well_name = 'B2_1'
registered_dir = os.path.join(cfg.OUTPUT_DIR, 'registered_seg')

print(f"Testing registration for well: {well_name}")
success = process_well_registration(well_name, registered_dir, transform_type='rigid')
print(f"\nResult: {'Success' if success else 'Failed'}")
