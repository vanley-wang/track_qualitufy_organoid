"""
FXN 类器官追踪脚本 —— 使用 laptrack 算法
对 FXN_0701 (Day3) 和 FXN_0703 (Day5) 的每个 well 进行：
  1. 连通域实例分割，过滤小类器官（默认 >= 4000 体素，可通过命令行修改）
  2. 用 laptrack 进行 ID 匹配（支持坐标模式 / 重叠模式）
  3. 形态量化、保存 NIfTI / NRRD、生成统计图表

安装依赖（如未安装）:
    pip install laptrack
    # 或从本地源码安装：
    # pip install -e ./laptrack-main

运行方式:
    python fxn_track_laptrack.py
    python fxn_track_laptrack.py --min-voxels 4000 --max-distance 100 --mode centroid
    python fxn_track_laptrack.py --mode overlap   # 需要先配准，否则匹配率会很低
"""

import os
import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 laptrack
try:
    from laptrack import LapTrack, OverLapTrack
except ImportError:
    print("错误：未安装 laptrack。请运行:  pip install laptrack")
    print("或从本地源码安装:  pip install -e ./laptrack-main")
    sys.exit(1)

import fxn_config as cfg

# =============================================================================
# 参数解析
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FXN organoid tracking with laptrack")
    parser.add_argument(
        "--min-voxels", type=int, default=4000,
        help="最小连通域体素数，只追踪大于此值的类器官 (默认: 4000)"
    )
    parser.add_argument(
        "--max-distance", type=float, default=cfg.MAX_DISTANCE_PIXELS,
        help=f"最大匹配距离 (像素，默认: {cfg.MAX_DISTANCE_PIXELS})"
    )
    parser.add_argument(
        "--mode", type=str, choices=["centroid", "overlap"], default="centroid",
        help="追踪模式: centroid=质心距离(无需配准), overlap=掩码重叠(需配准) (默认: centroid)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=None,
        help="laptrack cutoff 阈值。centroid 模式下默认等于 --max-distance；overlap 模式下默认 0.9"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: results_fxn_laptrack)"
    )
    parser.add_argument(
        "--allow-split", action="store_true",
        help="允许检测分裂事件 (splitting)"
    )
    parser.add_argument(
        "--allow-merge", action="store_true",
        help="允许检测合并事件 (merging)"
    )
    return parser.parse_args()


ARGS = parse_args()
MIN_VOXEL_COUNT = ARGS.min_voxels
MAX_DISTANCE = ARGS.max_distance
TRACK_MODE = ARGS.mode
OUTPUT_DIR = ARGS.output_dir or os.path.join(cfg.BASE_DIR, "results_fxn_laptrack")

# 自动设置 cutoff
if TRACK_MODE == "centroid":
    # centroid 模式使用 euclidean 距离，cutoff 直接就是距离阈值
    CUTOFF = ARGS.cutoff if ARGS.cutoff is not None else MAX_DISTANCE
else:
    # overlap 模式默认 0.9（允许 iou >= 0.1 的匹配）
    CUTOFF = ARGS.cutoff if ARGS.cutoff is not None else 0.9


# =============================================================================
# IO 工具函数（复用原有逻辑）
# =============================================================================
def load_nifti(file_path):
    """加载 NIfTI 文件，返回数据和 affine"""
    nii = nib.load(file_path)
    data = np.asanyarray(nii.dataobj).astype(np.uint8)
    return data, nii.affine


def save_nifti(data, affine, file_path):
    """保存 NIfTI 文件"""
    nii = nib.Nifti1Image(data.astype(np.int32), affine)
    nib.save(nii, file_path)


def instance_segmentation(binary_mask, min_voxel_count=MIN_VOXEL_COUNT):
    """
    对二值 mask 进行连通域标记，生成实例分割标签
    过滤小于 min_voxel_count 的连通域
    """
    labeled, num_features = ndimage.label(binary_mask > 0)
    if num_features == 0:
        return labeled, 0

    component_sizes = np.bincount(labeled.ravel())[1:]
    valid_mask = component_sizes >= min_voxel_count
    if not np.any(valid_mask):
        return np.zeros_like(labeled), 0

    valid_indices = np.where(valid_mask)[0] + 1
    mapping = np.zeros(num_features + 1, dtype=np.int32)
    for new_id, old_id in enumerate(valid_indices, start=1):
        mapping[old_id] = new_id
    new_labeled = mapping[labeled]

    return new_labeled, len(valid_indices)


def quantify_organoid(region, label_data, day_name, calculate_surface=False):
    """量化单个类器官"""
    label_id = region.label
    voxel_count = region.area
    volume_um3 = cfg.voxels_to_volume_um3(voxel_count)
    volume_mm3 = cfg.voxels_to_volume_mm3(voxel_count)

    centroid_z, centroid_y, centroid_x = region.centroid
    centroid_x_um = centroid_x * cfg.VOXEL_SIZE_X
    centroid_y_um = centroid_y * cfg.VOXEL_SIZE_Y
    centroid_z_um = centroid_z * cfg.VOXEL_SIZE_Z

    bbox = region.bbox
    bbox_x = (bbox[5] - bbox[2]) * cfg.VOXEL_SIZE_X
    bbox_y = (bbox[4] - bbox[1]) * cfg.VOXEL_SIZE_Y
    bbox_z = (bbox[3] - bbox[0]) * cfg.VOXEL_SIZE_Z

    equiv_diameter_um = region.equivalent_diameter * cfg.VOXEL_SIZE_X

    if calculate_surface:
        from skimage.measure import marching_cubes
        try:
            mask = (label_data == label_id)
            verts, faces, _, _ = marching_cubes(
                mask.astype(np.uint8),
                level=0.5,
                spacing=(cfg.VOXEL_SIZE_Z, cfg.VOXEL_SIZE_Y, cfg.VOXEL_SIZE_X)
            )
            triangles = verts[faces]
            v0 = triangles[:, 0, :]
            v1 = triangles[:, 1, :]
            v2 = triangles[:, 2, :]
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            surface_area = np.sum(areas)
            if surface_area > 0:
                ideal_surface = 4 * np.pi * (3 * volume_um3 / (4 * np.pi)) ** (2 / 3)
                sphericity = ideal_surface / surface_area
                sphericity = min(sphericity, 1.0)
                compactness = (volume_um3 ** 2) / (surface_area ** 3)
            else:
                surface_area = sphericity = compactness = np.nan
        except Exception:
            surface_area = sphericity = compactness = np.nan
    else:
        surface_area = sphericity = compactness = np.nan

    return {
        'Organoid_ID': label_id,
        'Day': day_name,
        'Voxel_Count': voxel_count,
        'Volume_um3': volume_um3,
        'Volume_mm3': volume_mm3,
        'Equivalent_Diameter_um': equiv_diameter_um,
        'Surface_Area_um2': surface_area,
        'Sphericity': sphericity,
        'Compactness': compactness,
        'Centroid_X_pixel': centroid_x,
        'Centroid_Y_pixel': centroid_y,
        'Centroid_Z_pixel': centroid_z,
        'Centroid_X_um': centroid_x_um,
        'Centroid_Y_um': centroid_y_um,
        'Centroid_Z_um': centroid_z_um,
        'BBox_X_um': bbox_x,
        'BBox_Y_um': bbox_y,
        'BBox_Z_um': bbox_z,
    }


def save_seg_nrrd(data, output_path, colors=None):
    """保存为 3D Slicer 兼容的 .seg.nrrd 格式"""
    header = {
        'type': 'short',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': np.array(data.shape),
        'space directions': np.eye(3) * [cfg.VOXEL_SIZE_X, cfg.VOXEL_SIZE_Y, cfg.VOXEL_SIZE_Z],
        'kinds': ['domain', 'domain', 'domain'],
        'space origin': np.zeros(3),
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]

    if colors is None:
        colors = cfg.SLICER_COLORS

    for seg_idx, label_value in enumerate(sorted(unique_labels)):
        label_value = int(label_value)
        color = colors[seg_idx % len(colors)]
        header[f'Segment{seg_idx}_Color'] = color
        header[f'Segment{seg_idx}_ColorAutoGenerated'] = '0'
        header[f'Segment{seg_idx}_Extent'] = f'0 {data.shape[0]-1} 0 {data.shape[1]-1} 0 {data.shape[2]-1}'
        header[f'Segment{seg_idx}_ID'] = f'Organoid_{label_value}'
        header[f'Segment{seg_idx}_LabelValue'] = str(label_value)
        header[f'Segment{seg_idx}_Layer'] = '0'
        header[f'Segment{seg_idx}_Name'] = f'Organoid_{label_value}'
        header[f'Segment{seg_idx}_NameAutoGenerated'] = '0'
        header[f'Segment{seg_idx}_Tags'] = ''

    nrrd.write(output_path, data.astype(np.int16), header)


# =============================================================================
# 核心追踪函数：使用 laptrack
# =============================================================================
def track_well_laptrack_centroid(label3, label5, max_distance):
    """
    使用 LapTrack (坐标模式) 对两帧标签图进行匹配
    返回: (label3_relabeled, label5_matched, track_info_dict)
    """
    props3 = regionprops(label3)
    props5 = regionprops(label5)

    if len(props3) == 0:
        return label3, label5, {}, []

    # 构建坐标 DataFrame (frame=0: Day3, frame=1: Day5)
    records = []
    for p in props3:
        records.append({
            'frame': 0,
            'label': p.label,
            'centroid-0': p.centroid[0],
            'centroid-1': p.centroid[1],
            'centroid-2': p.centroid[2],
        })
    for p in props5:
        records.append({
            'frame': 1,
            'label': p.label,
            'centroid-0': p.centroid[0],
            'centroid-1': p.centroid[1],
            'centroid-2': p.centroid[2],
        })
    df = pd.DataFrame(records)

    # 初始化 LapTrack
    lt = LapTrack(
        metric="euclidean",
        cutoff=max_distance,
        gap_closing_cutoff=False,
        splitting_cutoff=(max_distance if ARGS.allow_split else False),
        merging_cutoff=(max_distance if ARGS.allow_merge else False),
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        df,
        coordinate_cols=["centroid-0", "centroid-1", "centroid-2"],
        frame_col="frame",
    )

    # track_df 的 index 默认 0..N-1，与 df 对齐
    # 建立 (frame, label) -> tree_id 的映射
    frame_label_to_tree = {}
    for _, row in track_df.iterrows():
        frame = int(row["frame"])
        label = int(row["label"])
        tree_id = int(row["tree_id"])
        frame_label_to_tree[(frame, label)] = tree_id

    # 重新映射标签图 (tree_id 从 1 开始避免背景为 0)
    # 先收集所有 tree_id，排序后连续编号
    all_tree_ids = sorted(set(track_df["tree_id"].values))
    tree_id_mapping = {tid: (i + 1) for i, tid in enumerate(all_tree_ids)}

    label3_out = np.zeros_like(label3)
    label5_out = np.zeros_like(label5)

    for p in props3:
        key = (0, p.label)
        if key in frame_label_to_tree:
            new_id = tree_id_mapping[frame_label_to_tree[key]]
            label3_out[label3 == p.label] = new_id

    for p in props5:
        key = (1, p.label)
        if key in frame_label_to_tree:
            new_id = tree_id_mapping[frame_label_to_tree[key]]
            label5_out[label5 == p.label] = new_id

    # 收集匹配信息用于日志
    matches = {}  # Day5 original label -> Day3 original label (通过 tree_id 关联)
    tree_to_day3_label = {}
    for p in props3:
        key = (0, p.label)
        if key in frame_label_to_tree:
            tree_to_day3_label[frame_label_to_tree[key]] = p.label

    for p in props5:
        key = (1, p.label)
        if key in frame_label_to_tree:
            tree_id = frame_label_to_tree[key]
            if tree_id in tree_to_day3_label:
                matches[p.label] = tree_to_day3_label[tree_id]

    return label3_out, label5_out, matches, split_df, merge_df


def track_well_laptrack_overlap(label3, label5, cutoff):
    """
    使用 OverLapTrack (掩码重叠模式) 对两帧标签图进行匹配
    注意: 需要两帧已经空间对齐（配准），否则重叠率低、匹配失败多
    返回: (label3_relabeled, label5_matched, matches_dict)
    """
    # 构建 (2, Z, Y, X) 标签序列
    labels = np.stack([label3, label5], axis=0)

    olt = OverLapTrack(
        cutoff=cutoff,
        metric_coefs=(1.0, 0.0, -1.0, 0.0, 0.0),   # cost = 1 - iou
        gap_closing_metric_coefs=(1.0, 0.0, -1.0, 0.0, 0.0),
        gap_closing_max_frame_count=1,
        splitting_cutoff=(cutoff if ARGS.allow_split else False),
        splitting_metric_coefs=(1.0, 0.0, 0.0, 0.0, -1.0),  # cost = 1 - ratio_2 (day5 overlap ratio)
        merging_cutoff=(cutoff if ARGS.allow_merge else False),
        merging_metric_coefs=(1.0, 0.0, 0.0, -1.0, 0.0),    # cost = 1 - ratio_1 (day3 overlap ratio)
    )

    track_df, split_df, merge_df = olt.predict_overlap_dataframe(labels)

    # track_df 的 index 是 (frame, label)
    frame_label_to_tree = {}
    for (frame, label), row in track_df.iterrows():
        frame_label_to_tree[(int(frame), int(label))] = int(row["tree_id"])

    all_tree_ids = sorted(set(int(v) for v in track_df["tree_id"].values))
    tree_id_mapping = {tid: (i + 1) for i, tid in enumerate(all_tree_ids)}

    label3_out = np.zeros_like(label3)
    label5_out = np.zeros_like(label5)

    props3 = regionprops(label3)
    props5 = regionprops(label5)

    for p in props3:
        key = (0, p.label)
        if key in frame_label_to_tree:
            new_id = tree_id_mapping[frame_label_to_tree[key]]
            label3_out[label3 == p.label] = new_id

    for p in props5:
        key = (1, p.label)
        if key in frame_label_to_tree:
            new_id = tree_id_mapping[frame_label_to_tree[key]]
            label5_out[label5 == p.label] = new_id

    matches = {}
    tree_to_day3_label = {}
    for p in props3:
        key = (0, p.label)
        if key in frame_label_to_tree:
            tree_to_day3_label[frame_label_to_tree[key]] = p.label
    for p in props5:
        key = (1, p.label)
        if key in frame_label_to_tree:
            tree_id = frame_label_to_tree[key]
            if tree_id in tree_to_day3_label:
                matches[p.label] = tree_to_day3_label[tree_id]

    return label3_out, label5_out, matches, split_df, merge_df


# =============================================================================
# Well 处理流水线
# =============================================================================
def process_well(well_name, day3_seg_path, day5_seg_path, output_dir):
    """
    处理单个 well 的完整流水线
    返回: (quant_df_day3, quant_df_day5, log_records)
    """
    start_time = __import__('time').time()
    print(f"\n{'='*60}")
    print(f"处理 Well: {well_name}  [模式={TRACK_MODE}, min_voxels={MIN_VOXEL_COUNT}]")
    print(f"{'='*60}")

    # 1. 加载数据
    data3, affine3 = load_nifti(day3_seg_path)
    data5, affine5 = load_nifti(day5_seg_path)
    print(f"  Day3 shape: {data3.shape}, unique: {np.unique(data3)}")
    print(f"  Day5 shape: {data5.shape}, unique: {np.unique(data5)}")

    # 2. 实例分割（二值 -> 实例标签）
    unique3 = np.unique(data3)
    unique5 = np.unique(data5)

    if len(unique3) <= 2 and np.array_equal(unique3, [0, 1]):
        label3, n3 = instance_segmentation(data3)
        print(f"  Day3 连通域标记: {n3} 个类器官")
    else:
        label3 = data3.astype(np.int32)
        n3 = len(unique3) - 1
        print(f"  Day3 已是实例分割: {n3} 个类器官")

    if len(unique5) <= 2 and np.array_equal(unique5, [0, 1]):
        label5, n5 = instance_segmentation(data5)
        print(f"  Day5 连通域标记: {n5} 个类器官")
    else:
        label5 = data5.astype(np.int32)
        n5 = len(unique5) - 1
        print(f"  Day5 已是实例分割: {n5} 个类器官")

    if n3 == 0:
        print(f"  ⚠️  Day3 无类器官，跳过")
        return None, None, []

    # 3. laptrack 匹配
    print(f"\n  使用 laptrack ({TRACK_MODE} 模式) 匹配 Day5 → Day3 ...")
    if TRACK_MODE == "centroid":
        label3_out, label5_out, matches, split_df, merge_df = track_well_laptrack_centroid(
            label3, label5, MAX_DISTANCE
        )
    else:
        label3_out, label5_out, matches, split_df, merge_df = track_well_laptrack_overlap(
            label3, label5, CUTOFF
        )

    # 统计匹配结果
    n_matched = len(matches)
    day5_original_labels = set(p.label for p in regionprops(label5))
    n_new = len(day5_original_labels) - n_matched

    print(f"    匹配成功: {n_matched} 对")
    print(f"    Day5 新增/未匹配: {n_new} 个")
    if not split_df.empty:
        print(f"    检测到分裂事件: {len(split_df)} 个")
    if not merge_df.empty:
        print(f"    检测到合并事件: {len(merge_df)} 个")

    # 4. 量化分析
    print(f"\n  量化 Day3...")
    results_day3 = []
    for p in regionprops(label3_out):
        results_day3.append(quantify_organoid(p, label3_out, 'Day3', calculate_surface=False))

    print(f"  量化 Day5...")
    results_day5 = []
    for p in regionprops(label5_out):
        results_day5.append(quantify_organoid(p, label5_out, 'Day5', calculate_surface=False))

    df_day3 = pd.DataFrame(results_day3)
    df_day5 = pd.DataFrame(results_day5)

    # 5. 保存输出
    well_output_dir = os.path.join(output_dir, well_name)
    os.makedirs(well_output_dir, exist_ok=True)

    save_nifti(label3_out, affine3, os.path.join(well_output_dir, f'{well_name}_day3_matched.nii.gz'))
    save_nifti(label5_out, affine5, os.path.join(well_output_dir, f'{well_name}_day5_matched.nii.gz'))

    save_seg_nrrd(label3_out, os.path.join(well_output_dir, f'{well_name}_day3.seg.nrrd'))
    save_seg_nrrd(label5_out, os.path.join(well_output_dir, f'{well_name}_day5.seg.nrrd'))

    df_all = pd.concat([df_day3, df_day5], ignore_index=True)
    df_all.to_csv(os.path.join(well_output_dir, f'{well_name}_quantification.csv'), index=False)

    # 6. 日志记录
    log_records = []
    prop3_dict = {p.label: p for p in regionprops(label3)}
    prop5_dict = {p.label: p for p in regionprops(label5)}

    for target_label, ref_label in matches.items():
        t_prop = prop5_dict.get(target_label)
        r_prop = prop3_dict.get(ref_label)
        if t_prop is not None and r_prop is not None:
            dist = np.linalg.norm(np.array(t_prop.centroid) - np.array(r_prop.centroid))
            log_records.append({
                'Well': well_name,
                'Day': 'Day5',
                'Original_ID': target_label,
                'New_ID': ref_label,
                'Status': 'Matched',
                'Distance_pixels': round(dist, 2),
                'Centroid_Day3': f"({r_prop.centroid[0]:.1f}, {r_prop.centroid[1]:.1f}, {r_prop.centroid[2]:.1f})",
                'Centroid_Day5': f"({t_prop.centroid[0]:.1f}, {t_prop.centroid[1]:.1f}, {t_prop.centroid[2]:.1f})",
            })

    day5_matched_labels = set(matches.keys())
    for p in regionprops(label5):
        if p.label not in day5_matched_labels:
            log_records.append({
                'Well': well_name,
                'Day': 'Day5',
                'Original_ID': p.label,
                'New_ID': -1,
                'Status': 'New',
                'Distance_pixels': np.nan,
                'Centroid_Day3': '',
                'Centroid_Day5': f"({p.centroid[0]:.1f}, {p.centroid[1]:.1f}, {p.centroid[2]:.1f})",
            })

    elapsed = __import__('time').time() - start_time
    print(f"  ✓ Well {well_name} 完成 ({elapsed:.1f}s)，输出至: {well_output_dir}")
    return df_day3, df_day5, log_records


# =============================================================================
# 汇总统计与可视化
# =============================================================================
def generate_well_plots(df, well_name, output_dir):
    """为单个 well 生成统计图表"""
    df_day3 = df[df['Day'] == 'Day3']
    df_day5 = df[df['Day'] == 'Day5']

    if len(df_day3) == 0 or len(df_day5) == 0:
        return

    common_ids = sorted(set(df_day3['Organoid_ID']) & set(df_day5['Organoid_ID']))
    if len(common_ids) == 0:
        return

    changes = []
    for oid in common_ids:
        v3 = df_day3[df_day3['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        v5 = df_day5[df_day5['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        change_rate = (v5 - v3) / v3 if v3 > 0 else 0
        changes.append({'Organoid_ID': oid, 'Change_Rate': change_rate})
    df_change = pd.DataFrame(changes).sort_values('Change_Rate')

    def classify(rate):
        if rate <= cfg.THRESHOLD_COMPLETE_RESPONSE:
            return 'CR'
        elif rate <= cfg.THRESHOLD_PARTIAL_RESPONSE:
            return 'PR'
        elif cfg.THRESHOLD_STABLE_DISEASE_MIN <= rate <= cfg.THRESHOLD_STABLE_DISEASE_MAX:
            return 'SD'
        elif rate >= cfg.THRESHOLD_PROGRESSIVE_DISEASE:
            return 'PD'
        else:
            return 'SD'

    df_change['Response'] = df_change['Change_Rate'].apply(classify)
    response_counts = df_change['Response'].value_counts()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Well {well_name} - Organoid Tracking (laptrack)', fontsize=14, fontweight='bold')

    # 1. 瀑布图
    ax = axes[0, 0]
    colors_bar = [cfg.COLOR_CR if r == 'CR' else cfg.COLOR_PR if r == 'PR'
                  else cfg.COLOR_SD if r == 'SD' else cfg.COLOR_PD for r in df_change['Response']]
    ax.bar(range(len(df_change)), df_change['Change_Rate'] * 100, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Organoid (sorted by volume change)')
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('Waterfall Plot')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 药物反应饼图
    ax = axes[0, 1]
    if len(response_counts) > 0:
        pie_colors = [cfg.COLOR_CR if k == 'CR' else cfg.COLOR_PR if k == 'PR'
                      else cfg.COLOR_SD if k == 'SD' else cfg.COLOR_PD for k in response_counts.index]
        ax.pie(response_counts, labels=[f'{k}\n({v})' for k, v in response_counts.items()],
               colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Drug Response Classification')

    # 3. 个体体积轨迹
    ax = axes[1, 0]
    for oid in common_ids[:min(15, len(common_ids))]:
        v3 = df_day3[df_day3['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        v5 = df_day5[df_day5['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        ax.plot([3, 5], [v3, v5], 'o-', alpha=0.6, linewidth=1.5, markersize=6)
    ax.set_xlabel('Day')
    ax.set_ylabel('Volume (mm³)')
    ax.set_title('Individual Volume Trajectories')
    ax.set_xticks([3, 5])
    ax.grid(True, alpha=0.3)

    # 4. 体积 vs 球形度散点
    ax = axes[1, 1]
    ax.scatter(df_day3['Volume_mm3'], df_day3['Sphericity'],
               c=cfg.COLOR_DAY3, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, label='Day3')
    ax.scatter(df_day5['Volume_mm3'], df_day5['Sphericity'],
               c=cfg.COLOR_DAY5, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, label='Day5', marker='s')
    ax.set_xlabel('Volume (mm³)')
    ax.set_ylabel('Sphericity')
    ax.set_title('Volume vs Sphericity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f'{well_name}_analysis.png')
    plt.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def generate_summary_plots(all_quant_df, all_log_df, output_dir):
    """生成全局汇总图表"""
    if all_quant_df is None or len(all_quant_df) == 0:
        return

    df_day3 = all_quant_df[all_quant_df['Day'] == 'Day3']
    df_day5 = all_quant_df[all_quant_df['Day'] == 'Day5']

    common = df_day3.merge(df_day5, on=['Well', 'Organoid_ID'], suffixes=('_d3', '_d5'))
    if len(common) == 0:
        print("  无可匹配的共同类器官，跳过全局图表")
        return

    common['Volume_Change_Rate'] = (common['Volume_mm3_d5'] - common['Volume_mm3_d3']) / common['Volume_mm3_d3']
    common = common.sort_values('Volume_Change_Rate')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FXN Day3 → Day5 Global Summary (laptrack)', fontsize=14, fontweight='bold')

    # 1. 全局瀑布图
    ax = axes[0, 0]
    def classify(rate):
        if rate <= cfg.THRESHOLD_COMPLETE_RESPONSE: return 'CR'
        elif rate <= cfg.THRESHOLD_PARTIAL_RESPONSE: return 'PR'
        elif cfg.THRESHOLD_STABLE_DISEASE_MIN <= rate <= cfg.THRESHOLD_STABLE_DISEASE_MAX: return 'SD'
        elif rate >= cfg.THRESHOLD_PROGRESSIVE_DISEASE: return 'PD'
        else: return 'SD'
    common['Response'] = common['Volume_Change_Rate'].apply(classify)
    colors_bar = [cfg.COLOR_CR if r == 'CR' else cfg.COLOR_PR if r == 'PR'
                  else cfg.COLOR_SD if r == 'SD' else cfg.COLOR_PD for r in common['Response']]
    ax.bar(range(len(common)), common['Volume_Change_Rate'] * 100, color=colors_bar, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Organoid (sorted)')
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('Global Waterfall Plot')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 全局饼图
    ax = axes[0, 1]
    rc = common['Response'].value_counts()
    pie_colors = [cfg.COLOR_CR if k == 'CR' else cfg.COLOR_PR if k == 'PR'
                  else cfg.COLOR_SD if k == 'SD' else cfg.COLOR_PD for k in rc.index]
    ax.pie(rc, labels=[f'{k}\n({v})' for k, v in rc.items()], colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Global Drug Response')

    # 3. 体积分布箱线图
    ax = axes[1, 0]
    data_to_plot = [df_day3['Volume_mm3'].dropna(), df_day5['Volume_mm3'].dropna()]
    bp = ax.boxplot(data_to_plot, labels=['Day3', 'Day5'], patch_artist=True)
    bp['boxes'][0].set_facecolor(cfg.COLOR_DAY3)
    bp['boxes'][1].set_facecolor(cfg.COLOR_DAY5)
    ax.set_ylabel('Volume (mm³)')
    ax.set_title('Volume Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 匹配距离分布
    ax = axes[1, 1]
    matched_log = all_log_df[all_log_df['Status'] == 'Matched']
    if len(matched_log) > 0:
        ax.hist(matched_log['Distance_pixels'].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Centroid Distance (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Matching Distance Distribution')
        ax.axvline(MAX_DISTANCE, color='red', linestyle='--', label=f'Max threshold ({MAX_DISTANCE})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No matched organoids', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Matching Distance Distribution')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'global_summary.png'), dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 全局汇总图: {os.path.join(output_dir, 'global_summary.png')}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("FXN 类器官追踪与可视化 —— laptrack 版")
    print("Day3 (FXN_0701_seg) → Day5 (FXN_0703_seg)")
    print("=" * 70)
    cfg.print_config()
    print(f"\n[Run Parameters]")
    print(f"  Tracking mode: {TRACK_MODE}")
    print(f"  Min voxel count: {MIN_VOXEL_COUNT}")
    if TRACK_MODE == "centroid":
        print(f"  Max distance: {MAX_DISTANCE} pixels")
    print(f"  Cutoff: {CUTOFF}")
    print(f"  Allow split: {ARGS.allow_split}")
    print(f"  Allow merge: {ARGS.allow_merge}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 Day3 seg 文件")

    all_quant_records = []
    all_log_records = []
    well_stats = []

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, seg_file)
        day5_path = os.path.join(cfg.FXN_0703_SEG_DIR, seg_file)

        if not os.path.exists(day5_path):
            print(f"\n⚠️  跳过 {well_name}: Day5 文件不存在 ({day5_path})")
            continue

        df_day3, df_day5, logs = process_well(well_name, day3_path, day5_path, OUTPUT_DIR)

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

            # 生成单个 well 的图
            well_output_dir = os.path.join(OUTPUT_DIR, well_name)
            df_all = pd.concat([df_day3, df_day5], ignore_index=True)
            generate_well_plots(df_all, well_name, well_output_dir)

        all_log_records.extend(logs)

    # 保存全局 CSV
    if all_quant_records:
        df_all_quant = pd.concat(all_quant_records, ignore_index=True)
        df_all_quant.to_csv(os.path.join(OUTPUT_DIR, 'fxn_quantification.csv'), index=False)
        print(f"\n✓ 全局量化数据: {os.path.join(OUTPUT_DIR, 'fxn_quantification.csv')}")

    if all_log_records:
        df_all_log = pd.DataFrame(all_log_records)
        df_all_log.to_csv(os.path.join(OUTPUT_DIR, 'fxn_matching_log.csv'), index=False)
        print(f"✓ 全局匹配日志: {os.path.join(OUTPUT_DIR, 'fxn_matching_log.csv')}")

    if well_stats:
        df_stats = pd.DataFrame(well_stats)
        df_stats.to_csv(os.path.join(OUTPUT_DIR, 'fxn_well_summary.csv'), index=False)
        print(f"✓ Well 汇总: {os.path.join(OUTPUT_DIR, 'fxn_well_summary.csv')}")

    if all_quant_records and all_log_records:
        print("\n生成全局汇总图表...")
        generate_summary_plots(df_all_quant, df_all_log, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n可在 3D Slicer 中加载每个 well 的 .seg.nrrd 文件查看追踪结果")
    print("=" * 70)


if __name__ == '__main__':
    main()
