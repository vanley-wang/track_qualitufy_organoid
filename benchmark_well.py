"""Benchmark 单个 well 各步骤耗时（优化后）"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import os
import numpy as np
import nibabel as nib
import nrrd
from skimage.measure import regionprops

import fxn_config as cfg
from fxn_track_and_visualize import load_nifti, instance_segmentation, match_organoids_hungarian

well_name = 'B10_1'
day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, f'{well_name}.nii.gz')
day5_path = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

print(f"Benchmarking well: {well_name}")

t0 = time.time()
data3, affine3 = load_nifti(day3_path)
data5, affine5 = load_nifti(day5_path)
print(f"Load NIfTI: {time.time()-t0:.2f}s")

t0 = time.time()
label3, n3 = instance_segmentation(data3)
label5, n5 = instance_segmentation(data5)
print(f"Instance segmentation: {time.time()-t0:.2f}s (n3={n3}, n5={n5})")

t0 = time.time()
props3 = regionprops(label3)
props5 = regionprops(label5)
print(f"Regionprops: {time.time()-t0:.2f}s")

t0 = time.time()
matches, unmatched = match_organoids_hungarian(props3, props5, 100)
print(f"Hungarian matching: {time.time()-t0:.2f}s")

# 向量化 apply matching
t0 = time.time()
max_label5 = max([p.label for p in props5]) if props5 else 0
next_new_id = max([p.label for p in props3]) + 1 if props3 else 1
mapping = np.zeros(max_label5 + 1, dtype=np.int32)
for target_label, ref_label in matches.items():
    mapping[target_label] = ref_label
for t_idx in unmatched:
    mapping[props5[t_idx].label] = next_new_id
    next_new_id += 1
label5_matched = mapping[label5]
print(f"Apply matching (vectorized): {time.time()-t0:.2f}s")

t0 = time.time()
for p in regionprops(label3):
    _ = p.area, p.centroid, p.bbox, p.equivalent_diameter
for p in regionprops(label5_matched):
    _ = p.area, p.centroid, p.bbox, p.equivalent_diameter
print(f"Quantify (no surface): {time.time()-t0:.2f}s")

# Test save NIfTI
t0 = time.time()
nii = nib.Nifti1Image(label3.astype(np.int32), affine3)
nib.save(nii, f'results_fxn/{well_name}_day3_test.nii.gz')
print(f"Save NIfTI: {time.time()-t0:.2f}s")

# Cleanup
os.remove(f'results_fxn/{well_name}_day3_test.nii.gz')
print("Done")
