"""Benchmark 连通域标记算法"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import label as sk_label

f = 'E:/student/Private/student13/track_qualitify_organoid/nnUNet_FXN_2023/FXN_0701_seg/B10_1.nii.gz'
data = nib.load(f).get_fdata().astype(np.uint8)
mask = (data > 0)

print(f"Data shape: {data.shape}")
print(f"Foreground voxels: {mask.sum()}")

t0 = time.time()
l1, n1 = ndimage.label(mask)
print(f"scipy.ndimage.label: {time.time()-t0:.2f}s, n={n1}")

t0 = time.time()
l2, n2 = sk_label(mask, return_num=True, connectivity=1)
print(f"skimage.measure.label (connectivity=1): {time.time()-t0:.2f}s, n={n2}")

t0 = time.time()
l3, n3 = sk_label(mask, return_num=True, connectivity=2)
print(f"skimage.measure.label (connectivity=2): {time.time()-t0:.2f}s, n={n3}")

# Test if cc3d is available
try:
    import cc3d
    t0 = time.time()
    l4 = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
    n4 = l4.max()
    print(f"cc3d (6-connectivity): {time.time()-t0:.2f}s, n={n4}")
    t0 = time.time()
    l5 = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    n5 = l5.max()
    print(f"cc3d (26-connectivity): {time.time()-t0:.2f}s, n={n5}")
except ImportError:
    print("cc3d not available")
