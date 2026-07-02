"""
FXN 类器官鲁棒追踪脚本 —— 专为癌症类器官设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心特点：
  • 无需配准！直接对原始分割做追踪
  • LapTrack + 自定义多特征代价（位置 + 体积 + 形状）
  • 自动检测合并(Merge)、分裂(Split)、消失(Gone)、新生(New)事件
  • 过滤小类器官（默认 >= 4000 体素）

安装依赖:
    pip install laptrack

运行方式:
    python fxn_track_robust.py
    python fxn_track_robust.py --min-voxels 4000 --max-distance 150 --w-volume 20
    python fxn_track_robust.py --allow-merge --allow-split   # 启用合并/分裂检测

作者: Claude
"""

import os
import sys
import argparse
import warnings
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy import ndimage
from scipy.spatial.distance import cdist
import networkx as nx

# laptrack
try:
    from laptrack import LapTrack
except ImportError:
    print("错误：未安装 laptrack。请运行:  pip install laptrack")
    sys.exit(1)

import fxn_config as cfg
from fxn_track_and_visualize import (
    load_nifti, save_nifti, save_seg_nrrd, build_consistent_color_map,
    quantify_organoid, generate_well_plots, generate_summary_plots
)

# =============================================================================
# 命令行参数
# =============================================================================
parser = argparse.ArgumentParser(description="Robust FXN organoid tracking")
parser.add_argument("--min-voxels", type=int, default=4000,
                    help="最小连通域体素数 (默认: 4000)")
parser.add_argument("--max-distance", type=float, default=150,
                    help="最大质心匹配距离 (像素, 默认: 150)")
parser.add_argument("--w-volume", type=float, default=20.0,
                    help="体积差异权重 (默认: 20.0)")
parser.add_argument("--w-shape", type=float, default=10.0,
                    help="形状差异权重 (默认: 10.0)")
parser.add_argument("--allow-merge", action="store_true",
                    help="允许 laptrack 检测合并事件")
parser.add_argument("--allow-split", action="store_true",
                    help="允许 laptrack 检测分裂事件")
parser.add_argument("--output-dir", type=str, default=None,
                    help="输出目录 (默认: results_fxn_robust)")
ARGS = parser.parse_args()

MIN_VOXELS = ARGS.min_voxels
MAX_DIST_PIX = ARGS.max_distance
OUTPUT_DIR = ARGS.output_dir or os.path.join(cfg.BASE_DIR, "results_fxn_robust_new")

# 计算 cutoff（物理距离 + 体积/形状余量）
# 空间距离上限 + 体积差异(log10)上限 + 形状差异上限
CUT_OFF = (MAX_DIST_PIX * cfg.VOXEL_SIZE_X +
           ARGS.w_volume * np.log(10) +
           ARGS.w_shape * 1.0) * 1.2


# =============================================================================
# 工具函数
# =============================================================================
def instance_segmentation(binary_mask, min_voxel_count=MIN_VOXELS):
    """连通域标记，过滤小目标"""
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
    return mapping[labeled], len(valid_indices)


def extract_features(label_img):
    """提取每个类器官的特征 DataFrame 和 regionprops"""
    props = regionprops(label_img)
    records = []
    for p in props:
        vol = p.area
        # 快速估算球形度（基于等效直径 vs 面积，无需 marching cubes）
        eq_diam = p.equivalent_diameter
        ideal_surface = 4 * np.pi * (eq_diam / 2) ** 2
        actual_surface = max(vol, 1)
        sphericity = ideal_surface / actual_surface
        sphericity = min(sphericity, 1.0)

        records.append({
            'label': p.label,
            'centroid-0': p.centroid[0],
            'centroid-1': p.centroid[1],
            'centroid-2': p.centroid[2],
            'log_volume': np.log(max(vol, 1)),
            'sphericity': sphericity,
            'area': vol,
        })
    df = pd.DataFrame(records)
    return df, props


# =============================================================================
# LapTrack 自定义多特征距离
# =============================================================================
def make_metric(w_centroid=1.0, w_vol=ARGS.w_volume, w_shape=ARGS.w_shape,
                voxel_size=(cfg.VOXEL_SIZE_Z, cfg.VOXEL_SIZE_Y, cfg.VOXEL_SIZE_X)):
    """返回自定义距离函数，输入为 [z, y, x, log_volume, sphericity]"""
    def metric(c1, c2):
        dz = (c1[0] - c2[0]) * voxel_size[0]
        dy = (c1[1] - c2[1]) * voxel_size[1]
        dx = (c1[2] - c2[2]) * voxel_size[2]
        spatial = np.sqrt(dx**2 + dy**2 + dz**2)
        vol_dist = abs(c1[3] - c2[3])
        shape_dist = abs(c1[4] - c2[4])
        return w_centroid * spatial + w_vol * vol_dist + w_shape * shape_dist
    return metric


# =============================================================================
# 后处理：合并/分裂/消失/新生事件检测
# =============================================================================
def detect_events(props3, props5, max_dist_pix=MAX_DIST_PIX):
    """
    启发式检测合并和分裂事件
    无需配准，基于质心距离 + 体积约束
    """
    merges, splits = [], []
    if len(props3) == 0 or len(props5) == 0:
        return merges, splits

    c3 = np.array([p.centroid for p in props3])
    c5 = np.array([p.centroid for p in props5])
    l3 = [p.label for p in props3]
    l5 = [p.label for p in props5]
    a3 = [p.area for p in props3]
    a5 = [p.area for p in props5]

    dist_mat = cdist(c5, c3, metric='euclidean')

    # 合并：一个 Day5 附近多个 Day3，且 Day5 体积 >= 多个 Day3 体积之和的 40%
    for i5 in range(len(c5)):
        nearby = np.where(dist_mat[i5] <= max_dist_pix)[0]
        if len(nearby) >= 2:
            sum_vol = sum(a3[j] for j in nearby)
            if a5[i5] >= sum_vol * 0.4:
                merges.append({
                    'Well': None,  # 稍后填充
                    'Type': 'Merge',
                    'Day3_Labels': [int(l3[j]) for j in nearby],
                    'Day5_Label': int(l5[i5]),
                    'Day3_Volumes': [int(a3[j]) for j in nearby],
                    'Day5_Volume': int(a5[i5]),
                    'Distance_Pixels': float(np.min(dist_mat[i5, nearby])),
                })

    # 分裂：一个 Day3 附近多个 Day5，且 Day3 体积 >= 多个 Day5 体积之和的 40%
    for i3 in range(len(c3)):
        nearby = np.where(dist_mat[:, i3] <= max_dist_pix)[0]
        if len(nearby) >= 2:
            sum_vol = sum(a5[j] for j in nearby)
            if a3[i3] >= sum_vol * 0.4:
                splits.append({
                    'Well': None,
                    'Type': 'Split',
                    'Day3_Label': int(l3[i3]),
                    'Day5_Labels': [int(l5[j]) for j in nearby],
                    'Day3_Volume': int(a3[i3]),
                    'Day5_Volumes': [int(a5[j]) for j in nearby],
                    'Distance_Pixels': float(np.min(dist_mat[nearby, i3])),
                })

    return merges, splits


# =============================================================================
# Well 处理
# =============================================================================
def process_well(well_name, day3_path, day5_path, out_dir):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"鲁棒追踪 Well: {well_name}  [min_voxels={MIN_VOXELS}, max_dist={MAX_DIST_PIX}]")
    print(f"{'='*60}")

    # 加载
    data3, aff3 = load_nifti(day3_path)
    data5, aff5 = load_nifti(day5_path)

    # 实例分割（过滤小类器官）
    unique3 = np.unique(data3)
    unique5 = np.unique(data5)
    if len(unique3) <= 2 and np.array_equal(unique3, [0, 1]):
        label3, n3 = instance_segmentation(data3)
    else:
        label3 = data3.astype(np.int32)
        n3 = len(unique3) - 1
    if len(unique5) <= 2 and np.array_equal(unique5, [0, 1]):
        label5, n5 = instance_segmentation(data5)
    else:
        label5 = data5.astype(np.int32)
        n5 = len(unique5) - 1

    print(f"  Day3: {n3} 个类器官 (>= {MIN_VOXELS} voxels)")
    print(f"  Day5: {n5} 个类器官 (>= {MIN_VOXELS} voxels)")

    if n3 == 0:
        print("  ⚠️ Day3 无类器官，跳过")
        return None, None, [], [], []

    # 提取特征
    df3, props3 = extract_features(label3)
    df5, props5 = extract_features(label5)

    # 构建 laptrack 输入
    df3['frame'] = 0
    df5['frame'] = 1
    df_input = pd.concat([df3, df5], ignore_index=True)

    coord_cols = ['centroid-0', 'centroid-1', 'centroid-2', 'log_volume', 'sphericity']

    lt = LapTrack(
        metric=make_metric(),
        cutoff=CUT_OFF,
        gap_closing_cutoff=False,
        splitting_cutoff=(CUT_OFF if ARGS.allow_split else False),
        merging_cutoff=(CUT_OFF if ARGS.allow_merge else False),
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        df_input, coordinate_cols=coord_cols, frame_col='frame'
    )

    # 建立 (frame, label) -> tree_id 映射
    frame_label_to_tree = {}
    for _, row in track_df.iterrows():
        frame_label_to_tree[(int(row['frame']), int(row['label']))] = int(row['tree_id'])

    # tree_id 重新编号为连续正整数（用于 label 图像）
    all_trees = sorted(set(int(v) for v in track_df['tree_id'].values))
    tree_map = {tid: (i + 1) for i, tid in enumerate(all_trees)}

    # 应用重标记到 label 图像
    label3_out = np.zeros_like(label3)
    label5_out = np.zeros_like(label5)
    for _, row in df3.iterrows():
        key = (0, int(row['label']))
        if key in frame_label_to_tree:
            label3_out[label3 == row['label']] = tree_map[frame_label_to_tree[key]]
    for _, row in df5.iterrows():
        key = (1, int(row['label']))
        if key in frame_label_to_tree:
            label5_out[label5 == row['label']] = tree_map[frame_label_to_tree[key]]

    # 统计匹配/新生/消失
    day3_trees = set()
    for _, r in df3.iterrows():
        k = (0, int(r['label']))
        if k in frame_label_to_tree:
            day3_trees.add(frame_label_to_tree[k])
    day5_trees = set()
    for _, r in df5.iterrows():
        k = (1, int(r['label']))
        if k in frame_label_to_tree:
            day5_trees.add(frame_label_to_tree[k])

    common_trees = day3_trees & day5_trees
    n_matched = len(common_trees)
    n_gone = len(day3_trees - day5_trees)
    n_new = len(day5_trees - day3_trees)

    print(f"  匹配: {n_matched} 对 | 消失: {n_gone} | 新生: {n_new}")
    if not merge_df.empty:
        print(f"  LapTrack 合并事件: {len(merge_df)}")
    if not split_df.empty:
        print(f"  LapTrack 分裂事件: {len(split_df)}")

    # 启发式合并/分裂检测
    merges, splits = detect_events(props3, props5)
    if merges:
        print(f"  启发式合并检测: {len(merges)} 个")
    if splits:
        print(f"  启发式分裂检测: {len(splits)} 个")
    for m in merges + splits:
        m['Well'] = well_name

    # 量化
    print("  量化...")
    results_day3 = [quantify_organoid(p, label3_out, 'Day3', calculate_surface=False)
                    for p in regionprops(label3_out)]
    results_day5 = [quantify_organoid(p, label5_out, 'Day5', calculate_surface=False)
                    for p in regionprops(label5_out)]
    df_q3 = pd.DataFrame(results_day3)
    df_q5 = pd.DataFrame(results_day5)

    # 日志记录
    logs = []
    prop3_dict = {p.label: p for p in props3}
    prop5_dict = {p.label: p for p in props5}

    # Matched
    for _, r5 in df5.iterrows():
        key5 = (1, int(r5['label']))
        if key5 not in frame_label_to_tree:
            continue
        tree_id = frame_label_to_tree[key5]
        matched_day3 = None
        for _, r3 in df3.iterrows():
            if frame_label_to_tree.get((0, int(r3['label']))) == tree_id:
                matched_day3 = r3
                break
        if matched_day3 is not None:
            t_prop = prop5_dict.get(int(r5['label']))
            r_prop = prop3_dict.get(int(matched_day3['label']))
            dist = np.linalg.norm(np.array(t_prop.centroid) - np.array(r_prop.centroid))
            logs.append({
                'Well': well_name, 'Day': 'Day5',
                'Original_ID': int(r5['label']), 'New_ID': int(matched_day3['label']),
                'Status': 'Matched', 'Distance_pixels': round(dist, 2),
                'Centroid_Day3': f"({r_prop.centroid[0]:.1f}, {r_prop.centroid[1]:.1f}, {r_prop.centroid[2]:.1f})",
                'Centroid_Day5': f"({t_prop.centroid[0]:.1f}, {t_prop.centroid[1]:.1f}, {t_prop.centroid[2]:.1f})",
            })

    # New (unmatched Day5)
    for _, r5 in df5.iterrows():
        key5 = (1, int(r5['label']))
        if key5 not in frame_label_to_tree:
            continue
        tree_id = frame_label_to_tree[key5]
        if tree_id not in day3_trees:
            t_prop = prop5_dict.get(int(r5['label']))
            logs.append({
                'Well': well_name, 'Day': 'Day5',
                'Original_ID': int(r5['label']), 'New_ID': -1,
                'Status': 'New', 'Distance_pixels': np.nan,
                'Centroid_Day3': '',
                'Centroid_Day5': f"({t_prop.centroid[0]:.1f}, {t_prop.centroid[1]:.1f}, {t_prop.centroid[2]:.1f})",
            })

    # Gone (unmatched Day3)
    for _, r3 in df3.iterrows():
        key3 = (0, int(r3['label']))
        if key3 not in frame_label_to_tree:
            continue
        tree_id = frame_label_to_tree[key3]
        if tree_id not in day5_trees:
            r_prop = prop3_dict.get(int(r3['label']))
            logs.append({
                'Well': well_name, 'Day': 'Day3',
                'Original_ID': int(r3['label']), 'New_ID': -1,
                'Status': 'Gone', 'Distance_pixels': np.nan,
                'Centroid_Day3': f"({r_prop.centroid[0]:.1f}, {r_prop.centroid[1]:.1f}, {r_prop.centroid[2]:.1f})",
                'Centroid_Day5': '',
            })

    # 保存
    wdir = os.path.join(out_dir, well_name)
    os.makedirs(wdir, exist_ok=True)

    save_nifti(label3_out, aff3, os.path.join(wdir, f'{well_name}_day3.nii.gz'))
    save_nifti(label5_out, aff5, os.path.join(wdir, f'{well_name}_day5.nii.gz'))
    consistent_color_map = build_consistent_color_map(label3_out, label5_out)
    save_seg_nrrd(label3_out, os.path.join(wdir, f'{well_name}_day3.seg.nrrd'),
                  id_to_color_map=consistent_color_map)
    save_seg_nrrd(label5_out, os.path.join(wdir, f'{well_name}_day5.seg.nrrd'),
                  id_to_color_map=consistent_color_map)

    df_all = pd.concat([df_q3, df_q5], ignore_index=True)
    df_all.to_csv(os.path.join(wdir, f'{well_name}_quantification.csv'), index=False)

    # 单 well 图表
    generate_well_plots(df_all, well_name, wdir)

    elapsed = time.time() - start
    print(f"  ✓ 完成 ({elapsed:.1f}s)")
    return df_q3, df_q5, logs, merges, splits


# =============================================================================
# 全局汇总（按浓度组）
# =============================================================================
def get_group(well_name):
    prefix = well_name.split('_')[0]
    return cfg.WELL_GROUPS.get(prefix, 'Unknown')


def generate_robust_summary(all_quant, all_log, all_merges, all_splits, out_dir):
    if all_quant is None or len(all_quant) == 0:
        return

    df_day3 = all_quant[all_quant['Day'] == 'Day3']
    df_day5 = all_quant[all_quant['Day'] == 'Day5']
    common = df_day3.merge(df_day5, on=['Well', 'Organoid_ID'], suffixes=('_d3', '_d5'))
    if len(common) == 0:
        print("  无共同追踪对象，跳过汇总图")
        return

    common['Volume_Change_Rate'] = (common['Volume_mm3_d5'] - common['Volume_mm3_d3']) / common['Volume_mm3_d3']
    common['Group'] = common['Well_d3'].apply(get_group)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('FXN Robust Tracking Summary (No Registration)', fontsize=16, fontweight='bold')

    # 1. 各组平均体积变化
    ax = axes[0, 0]
    grp = common.groupby('Group')['Volume_Change_Rate'].agg(['mean', 'std', 'count'])
    groups = grp.index.tolist()
    means = grp['mean'].values * 100
    stds = grp['std'].values * 100
    colors = [cfg.GROUP_COLORS.get(g, '#333') for g in groups]
    bars = ax.bar(groups, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean Volume Change (%)')
    ax.set_title('Mean Change by Group')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 药物反应堆叠柱状图
    ax = axes[0, 1]
    def classify(rate):
        if rate <= cfg.THRESHOLD_COMPLETE_RESPONSE: return 'CR'
        elif rate <= cfg.THRESHOLD_PARTIAL_RESPONSE: return 'PR'
        elif cfg.THRESHOLD_STABLE_DISEASE_MIN <= rate <= cfg.THRESHOLD_STABLE_DISEASE_MAX: return 'SD'
        elif rate >= cfg.THRESHOLD_PROGRESSIVE_DISEASE: return 'PD'
        else: return 'SD'
    common['Response'] = common['Volume_Change_Rate'].apply(classify)
    resp = common.groupby(['Group', 'Response']).size().unstack(fill_value=0)
    resp = resp[['CR', 'PR', 'SD', 'PD']] if not resp.empty else resp
    resp.plot(kind='bar', stacked=True, ax=ax,
              color=[cfg.COLOR_CR, cfg.COLOR_PR, cfg.COLOR_SD, cfg.COLOR_PD],
              edgecolor='black', linewidth=0.5)
    ax.set_title('Response by Group')
    ax.legend(title='Response')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 体积变化箱线图
    ax = axes[0, 2]
    gdata = [common[common['Group'] == g]['Volume_Change_Rate'].values * 100 for g in groups]
    bp = ax.boxplot(gdata, labels=groups, patch_artist=True)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(cfg.GROUP_COLORS.get(g, '#333'))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('Change Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 匹配/新生/消失统计
    ax = axes[1, 0]
    if all_log is not None and len(all_log) > 0:
        log_df = pd.DataFrame(all_log)
        status_counts = log_df['Status'].value_counts()
        colors_pie = {'Matched': '#3498db', 'New': '#2ecc71', 'Gone': '#e74c3c'}
        sc = status_counts.reindex(['Matched', 'New', 'Gone']).fillna(0)
        ax.pie(sc, labels=[f'{k}\n({int(v)})' for k, v in sc.items()],
               colors=[colors_pie.get(k, '#95a5a6') for k in sc.index],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Tracking Events')

    # 5. 合并事件统计
    ax = axes[1, 1]
    if all_merges:
        merge_df = pd.DataFrame(all_merges)
        merge_by_well = merge_df.groupby('Well').size()
        if len(merge_by_well) > 0:
            merge_by_well.plot(kind='bar', ax=ax, color='#e67e22', edgecolor='black')
            ax.set_title(f'Merge Events (total={len(all_merges)})')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No merge events', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Merge Events')

    # 6. 分裂事件统计
    ax = axes[1, 2]
    if all_splits:
        split_df = pd.DataFrame(all_splits)
        split_by_well = split_df.groupby('Well').size()
        if len(split_by_well) > 0:
            split_by_well.plot(kind='bar', ax=ax, color='#9b59b6', edgecolor='black')
            ax.set_title(f'Split Events (total={len(all_splits)})')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No split events', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Split Events')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, 'robust_summary.png'), dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 全局汇总图: {os.path.join(out_dir, 'robust_summary.png')}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("FXN 类器官鲁棒追踪 (laptrack + 多特征)")
    print("无需配准，支持合并/分裂/消失/新生检测")
    print("=" * 70)
    cfg.print_config()
    print(f"\n[Run Parameters]")
    print(f"  Min voxels: {MIN_VOXELS}")
    print(f"  Max distance: {MAX_DIST_PIX} px = {MAX_DIST_PIX * cfg.VOXEL_SIZE_X:.0f} μm")
    print(f"  Weights: centroid=1.0, volume={ARGS.w_volume}, shape={ARGS.w_shape}")
    print(f"  LapTrack cutoff: {CUT_OFF:.1f}")
    print(f"  Allow merge: {ARGS.allow_merge}")
    print(f"  Allow split: {ARGS.allow_split}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 well")

    all_quant, all_log = [], []
    all_merges, all_splits = [], []
    well_stats = []

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        d3 = os.path.join(cfg.FXN_0701_SEG_DIR, seg_file)
        d5 = os.path.join(cfg.FXN_0703_SEG_DIR, seg_file)
        if not os.path.exists(d5):
            print(f"\n⚠️ 跳过 {well_name}: Day5 不存在")
            continue

        df3, df5, logs, merges, splits = process_well(well_name, d3, d5, OUTPUT_DIR)

        if df3 is not None and df5 is not None:
            df3['Well'] = well_name
            df5['Well'] = well_name
            all_quant.extend([df3, df5])

            n_m = sum(1 for l in logs if l['Status'] == 'Matched')
            n_new = sum(1 for l in logs if l['Status'] == 'New')
            n_gone = sum(1 for l in logs if l['Status'] == 'Gone')
            well_stats.append({
                'Well': well_name, 'Group': get_group(well_name),
                'Day3_Count': len(df3), 'Day5_Count': len(df5),
                'Matched': n_m, 'New': n_new, 'Gone': n_gone,
                'Merge_Events': len(merges), 'Split_Events': len(splits),
            })

        all_log.extend(logs)
        all_merges.extend(merges)
        all_splits.extend(splits)

    # 保存全局 CSV
    if all_quant:
        df_all_q = pd.concat(all_quant, ignore_index=True)
        df_all_q.to_csv(os.path.join(OUTPUT_DIR, 'fxn_quantification_robust.csv'), index=False)
        print(f"\n✓ 全局量化: {os.path.join(OUTPUT_DIR, 'fxn_quantification_robust.csv')}")

    if all_log:
        df_all_l = pd.DataFrame(all_log)
        df_all_l.to_csv(os.path.join(OUTPUT_DIR, 'fxn_matching_log_robust.csv'), index=False)
        print(f"✓ 匹配日志: {os.path.join(OUTPUT_DIR, 'fxn_matching_log_robust.csv')}")

    if all_merges:
        pd.DataFrame(all_merges).to_csv(os.path.join(OUTPUT_DIR, 'fxn_merge_events.csv'), index=False)
        print(f"✓ 合并事件: {os.path.join(OUTPUT_DIR, 'fxn_merge_events.csv')}")

    if all_splits:
        pd.DataFrame(all_splits).to_csv(os.path.join(OUTPUT_DIR, 'fxn_split_events.csv'), index=False)
        print(f"✓ 分裂事件: {os.path.join(OUTPUT_DIR, 'fxn_split_events.csv')}")

    if well_stats:
        pd.DataFrame(well_stats).to_csv(os.path.join(OUTPUT_DIR, 'fxn_well_summary_robust.csv'), index=False)
        print(f"✓ Well 汇总: {os.path.join(OUTPUT_DIR, 'fxn_well_summary_robust.csv')}")

    if all_quant and all_log:
        print("\n生成全局汇总图表...")
        generate_robust_summary(df_all_q, all_log, all_merges, all_splits, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
