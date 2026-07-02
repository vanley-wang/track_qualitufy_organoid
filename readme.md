# 版本 A：一对一（默认）

python fxn_track_robust.py --min-voxels 4000 --max-distance 200 --w-volume 15

# 版本 B：允许合并/分裂

python fxn_track_robust.py --min-voxels 4000 --max-distance 200 --w-volume 15 --allow-merge --allow-split
