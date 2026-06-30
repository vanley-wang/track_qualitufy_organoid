"""Benchmark instance_segmentation 各子步骤"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import nibabel as nib
from scipy import ndimage

f = 'E:/student/Private/student13/track_qualitify_organoid/nnUNet_FXN_2023/FXN_0701_seg/B10_1.nii.gz'
data = nib.load(f).get_fdata().astype(np.uint8)
mask = (data > 0)

print(f"Data shape: {data.shape}")

t0 = time.time()
labeled, num_features = ndimage.label(mask)
elapsed = time.time()-t0
print(f"1. ndimage.label: {elapsed:.2f}s, features={num_features}")

t0 = time.time()
component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_features + 1))
print(f"2. ndimage.sum: {time.time()-t0:.2f}s")

t0 = time.time()
valid_mask = component_sizes >= 500
valid_indices = np.where(valid_mask)[0] + 1
print(f"3. Filter: {time.time()-t0:.2f}s, valid={len(valid_indices)}")

t0 = time.time()
new_labeled = np.zeros_like(labeled)
for new_id, old_id in enumerate(valid_indices, start=1):
    new_labeled[labeled == old_id] = new_id
print(f"4. Relabel loop: {time.time()-t0:.2f}s")

# Optimized relabeling using np.where or searchsorted
t0 = time.time()
# Create mapping array
mapping = np.zeros(num_features + 1, dtype=np.int32)
for new_id, old_id in enumerate(valid_indices, start=1):
    mapping[old_id] = new_id
new_labeled2 = mapping[labeled]
print(f"5. Vectorized relabel (mapping array): {time.time()-t0:.2f}s")

print(f"Match check: {np.array_equal(new_labeled, new_labeled2)}")
