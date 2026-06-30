"""
FXN 多特征加权匹配追踪脚本
对 B-spline 配准后的数据进行追踪
特征: 质心距离 + 体积相似度 + 球形度相似度

运行方式:
    python fxn_track_multifeature.py
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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg
from fxn_track_and_visualize import (
    load_nifti, instance_segmentation, save_seg_nrrd,
    quantify_organoid, generate_well_plots, save_nifti
)


def get_group(well_name):
    """获取 well 所属浓度组"""
    prefix = well_name.split('_')[0]
    return cfg.WELL_GROUPS.get(prefix, 'Unknown')


def compute_multifeature_distance(props_ref, props_target, w_centroid=cfg.MF_WEIGHT_CENTROID,
                                   w_volume=cfg.MF_WEIGHT_VOLUME, w_sphericity=cfg.MF_WEIGHT_SPHERICITY):
    """
    计算多特征加权距离矩阵
    返回: distance_matrix (n_target, n_ref)
    """
    n_ref = len(props_ref)
    n_target = len(props_target)

    if n_ref == 0 or n_target == 0:
        return np.zeros((n_target, n_ref))

    # 提取特征
    ref_centroids = np.array([p.centroid for p in props_ref])
    target_centroids = np.array([p.centroid for p in props_target])

    ref_volumes = np.array([p.area for p in props_ref], dtype=float)
    target_volumes = np.array([p.area for p in props_target], dtype=float)

    ref_sphericity = np.array([
        min(1.0, (np.pi ** (1/3)) * ((6 * cfg.voxels_to_volume_um3(p.area)) ** (2/3)) /
            max(1.0, p.equivalent_diameter ** 2 * np.pi)) for p in props_ref
    ])
    target_sphericity = np.array([
        min(1.0, (np.pi ** (1/3)) * ((6 * cfg.voxels_to_volume_um3(p.area)) ** (2/3)) /
            max(1.0, p.equivalent_diameter ** 2 * np.pi)) for p in props_target
    ])

    # 1. 质心距离矩阵
    dist_centroid = cdist(target_centroids, ref_centroids, metric='euclidean')

    # 2. 体积差异矩阵 (log ratio)
    # 避免除零: 体积都至少 MIN_VOXEL_COUNT
    vol_ratio = np.log(target_volumes[:, None] / ref_volumes[None, :])
    dist_volume = np.abs(vol_ratio)

    # 3. 球形度差异矩阵
    dist_sphericity = np.abs(target_sphericity[:, None] - ref_sphericity[None, :])

    # 加权总距离
    distance_matrix = (w_centroid * dist_centroid +
                       w_volume * dist_volume +
                       w_sphericity * dist_sphericity)

    return distance_matrix, dist_centroid, dist_volume, dist_sphericity


def match_organoids_multifeature(props_ref, props_target, max_distance=cfg.MF_MAX_DISTANCE):
    """
    使用多特征加权距离 + 匈牙利算法进行一对一匹配
    返回: matches dict {target_label: ref_label}, unmatched_target list
    """
    if len(props_ref) == 0 or len(props_target) == 0:
        return {}, list(range(len(props_target)))

    distance_matrix, dist_centroid, _, _ = compute_multifeature_distance(props_ref, props_target)

    ref_labels = np.array([p.label for p in props_ref])
    target_labels = np.array([p.label for p in props_target])

    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matches = {}
    unmatched_target = []

    for t_idx, s_idx in zip(row_ind, col_ind):
        total_dist = distance_matrix[t_idx, s_idx]
        centroid_dist = dist_centroid[t_idx, s_idx]
        # 以质心距离作为主阈值判断，总距离作为辅助
        if total_dist <= max_distance:
            matches[target_labels[t_idx]] = ref_labels[s_idx]
        else:
            unmatched_target.append(t_idx)

    # 未参与匹配的目标
    matched_target_set = set(row_ind)
    for t_idx in range(len(props_target)):
        if t_idx not in matched_target_set:
            unmatched_target.append(t_idx)

    return matches, unmatched_target


def process_well_multifeature(well_name, day3_path, day5_path, output_dir):
    """
    处理单个 well 的多特征追踪
    """
    print(f"\n{'='*60}")
    print(f"多特征追踪 Well: {well_name} [{get_group(well_name)}]")
    print(f"{'='*60}")

    # 1. 加载数据
    data3, affine3 = load_nifti(day3_path)
    data5, affine5 = load_nifti(day5_path)

    # 2. 实例分割
    label3, n3 = instance_segmentation(data3)
    label5, n5 = instance_segmentation(data5)
    print(f"  Day3: {n3} 个类器官, Day5: {n5} 个类器官")

    if n3 == 0:
        print(f"  ⚠️  Day3 无类器官，跳过")
        return None, None, []

    # 3. 提取 regionprops
    from skimage.measure import regionprops
    props3 = regionprops(label3)
    props5 = regionprops(label5)

    # 4. 多特征匹配
    print(f"  多特征匹配 (max_dist={cfg.MF_MAX_DISTANCE})...")
    matches, unmatched = match_organoids_multifeature(props3, props5)
    print(f"    匹配成功: {len(matches)} 对, 新增: {len(unmatched)} 个")

    # 5. 应用匹配（向量化）
    max_label5 = max([p.label for p in props5]) if props5 else 0
    next_new_id = max([p.label for p in props3]) + 1 if props3 else 1
    mapping = np.zeros(max_label5 + 1, dtype=np.int32)
    for target_label, ref_label in matches.items():
        mapping[target_label] = ref_label
    for t_idx in unmatched:
        mapping[props5[t_idx].label] = next_new_id
        next_new_id += 1
    label5_matched = mapping[label5]

    # 6. 日志
    prop3_dict = {p.label: p for p in props3}
    prop5_dict = {p.label: p for p in props5}
    log_records = []

    for target_label, ref_label in matches.items():
        t_prop = prop5_dict.get(target_label)
        r_prop = prop3_dict.get(ref_label)
        if t_prop is not None and r_prop is not None:
            dist = np.linalg.norm(np.array(t_prop.centroid) - np.array(r_prop.centroid))
            log_records.append({
                'Well': well_name, 'Group': get_group(well_name), 'Day': 'Day5',
                'Original_ID': target_label, 'New_ID': ref_label,
                'Status': 'Matched', 'Distance_pixels': round(dist, 2),
                'Centroid_Day3': f"({r_prop.centroid[0]:.1f}, {r_prop.centroid[1]:.1f}, {r_prop.centroid[2]:.1f})",
                'Centroid_Day5': f"({t_prop.centroid[0]:.1f}, {t_prop.centroid[1]:.1f}, {t_prop.centroid[2]:.1f})",
            })

    for t_idx in unmatched:
        target_label = props5[t_idx].label
        log_records.append({
            'Well': well_name, 'Group': get_group(well_name), 'Day': 'Day5',
            'Original_ID': target_label, 'New_ID': mapping[target_label],
            'Status': 'New', 'Distance_pixels': np.nan,
            'Centroid_Day3': '',
            'Centroid_Day5': f"({props5[t_idx].centroid[0]:.1f}, {props5[t_idx].centroid[1]:.1f}, {props5[t_idx].centroid[2]:.1f})",
        })

    # 7. 量化
    print(f"  量化...")
    results_day3 = [quantify_organoid(p, label3, 'Day3') for p in regionprops(label3)]
    results_day5 = [quantify_organoid(p, label5_matched, 'Day5') for p in regionprops(label5_matched)]
    df_day3 = pd.DataFrame(results_day3)
    df_day5 = pd.DataFrame(results_day5)

    # 8. 保存
    well_output_dir = os.path.join(output_dir, well_name)
    os.makedirs(well_output_dir, exist_ok=True)

    save_nifti(label3, affine3, os.path.join(well_output_dir, f'{well_name}_day3.nii.gz'))
    save_nifti(label5_matched, affine5, os.path.join(well_output_dir, f'{well_name}_day5.nii.gz'))
    save_seg_nrrd(label3, os.path.join(well_output_dir, f'{well_name}_day3.seg.nrrd'))
    save_seg_nrrd(label5_matched, os.path.join(well_output_dir, f'{well_name}_day5.seg.nrrd'))

    df_all = pd.concat([df_day3, df_day5], ignore_index=True)
    df_all.to_csv(os.path.join(well_output_dir, f'{well_name}_quantification.csv'), index=False)

    elapsed = 0  # placeholder
    print(f"  ✓ Well {well_name} 完成，输出至: {well_output_dir}")
    return df_day3, df_day5, log_records


def generate_group_summary(df_all_quant, df_all_log, output_dir):
    """按浓度组生成汇总图表"""
    if df_all_quant is None or len(df_all_quant) == 0:
        return

    # 合并量化数据和分组信息
    df_all_quant['Group'] = df_all_quant['Well'].apply(get_group)

    # 计算跨天体积变化
    df_day3 = df_all_quant[df_all_quant['Day'] == 'Day3']
    df_day5 = df_all_quant[df_all_quant['Day'] == 'Day5']
    common = df_day3.merge(df_day5, on=['Well', 'Organoid_ID'], suffixes=('_d3', '_d5'))
    if len(common) == 0:
        print("  无共同追踪的类器官，跳过分组图表")
        return

    common['Volume_Change_Rate'] = (common['Volume_mm3_d5'] - common['Volume_mm3_d3']) / common['Volume_mm3_d3']
    common['Group'] = common['Well_d3'].apply(get_group)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FXN Drug Response by Concentration Group', fontsize=16, fontweight='bold')

    # 1. 各组平均体积变化率
    ax = axes[0, 0]
    group_means = common.groupby('Group')['Volume_Change_Rate'].agg(['mean', 'std', 'count'])
    groups = group_means.index.tolist()
    means = group_means['mean'].values * 100
    stds = group_means['std'].values * 100
    colors = [cfg.GROUP_COLORS.get(g, '#333') for g in groups]
    bars = ax.bar(groups, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean Volume Change (%)')
    ax.set_title('Mean Volume Change by Group')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m, c in zip(bars, means, group_means['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{m:.1f}%\n(n={c})', ha='center', va='bottom', fontsize=9)

    # 2. 各组药物反应分布
    ax = axes[0, 1]
    def classify(rate):
        if rate <= cfg.THRESHOLD_COMPLETE_RESPONSE: return 'CR'
        elif rate <= cfg.THRESHOLD_PARTIAL_RESPONSE: return 'PR'
        elif cfg.THRESHOLD_STABLE_DISEASE_MIN <= rate <= cfg.THRESHOLD_STABLE_DISEASE_MAX: return 'SD'
        elif rate >= cfg.THRESHOLD_PROGRESSIVE_DISEASE: return 'PD'
        else: return 'SD'

    common['Response'] = common['Volume_Change_Rate'].apply(classify)
    response_by_group = common.groupby(['Group', 'Response']).size().unstack(fill_value=0)
    response_by_group = response_by_group[['CR', 'PR', 'SD', 'PD']]
    response_by_group.plot(kind='bar', stacked=True, ax=ax,
                           color=[cfg.COLOR_CR, cfg.COLOR_PR, cfg.COLOR_SD, cfg.COLOR_PD],
                           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Count')
    ax.set_title('Drug Response Distribution by Group')
    ax.legend(title='Response')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 各组体积变化箱线图
    ax = axes[1, 0]
    group_data = [common[common['Group'] == g]['Volume_Change_Rate'].values * 100 for g in groups]
    bp = ax.boxplot(group_data, labels=groups, patch_artist=True)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(cfg.GROUP_COLORS.get(g, '#333'))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('Volume Change Distribution by Group')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 匹配率按组
    ax = axes[1, 1]
    log_df = df_all_log.copy()
    log_df['Group'] = log_df['Well'].apply(get_group)
    match_rates = log_df.groupby('Group').apply(
        lambda x: (x['Status'] == 'Matched').sum() / len(x) * 100
    )
    colors = [cfg.GROUP_COLORS.get(g, '#333') for g in match_rates.index]
    bars = ax.bar(match_rates.index, match_rates.values, color=colors, edgecolor='black')
    ax.set_ylabel('Match Rate (%)')
    ax.set_title('Tracking Match Rate by Group')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, match_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'group_summary.png'), dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 分组汇总图: {os.path.join(output_dir, 'group_summary.png')}")


def main():
    print("=" * 70)
    print("FXN 多特征追踪 (B-spline 配准后)")
    print("Day3 (FXN_0701_seg) → Day5 (bspline_seg)")
    print("=" * 70)

    bspline_dir = os.path.join(cfg.OUTPUT_DIR, 'bspline_seg')
    output_dir = os.path.join(cfg.OUTPUT_DIR, 'multifeature_tracking')
    os.makedirs(output_dir, exist_ok=True)

    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 well")
    print(f"B-spline seg 目录: {bspline_dir}")
    print(f"追踪输出目录: {output_dir}")

    all_quant = []
    all_log = []
    well_stats = []

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, seg_file)
        day5_path = os.path.join(bspline_dir, f'{well_name}_day5_bspline.nii.gz')

        if not os.path.exists(day5_path):
            print(f"\n⚠️  跳过 {well_name}: B-spline 文件不存在")
            continue

        df_day3, df_day5, logs = process_well_multifeature(well_name, day3_path, day5_path, output_dir)

        if df_day3 is not None and df_day5 is not None:
            df_day3['Well'] = well_name
            df_day5['Well'] = well_name
            all_quant.append(df_day3)
            all_quant.append(df_day5)

            n_matched = sum(1 for l in logs if l['Status'] == 'Matched')
            n_new = sum(1 for l in logs if l['Status'] == 'New')
            well_stats.append({
                'Well': well_name, 'Group': get_group(well_name),
                'Day3_Count': len(df_day3), 'Day5_Count': len(df_day5),
                'Matched': n_matched, 'New': n_new,
            })

        all_log.extend(logs)

    # 保存全局 CSV
    if all_quant:
        df_all_quant = pd.concat(all_quant, ignore_index=True)
        df_all_quant.to_csv(os.path.join(output_dir, 'quantification_multifeature.csv'), index=False)
        print(f"\n✓ 全局量化: {os.path.join(output_dir, 'quantification_multifeature.csv')}")

    if all_log:
        df_all_log = pd.DataFrame(all_log)
        df_all_log.to_csv(os.path.join(output_dir, 'matching_log_multifeature.csv'), index=False)
        print(f"✓ 匹配日志: {os.path.join(output_dir, 'matching_log_multifeature.csv')}")

    if well_stats:
        df_stats = pd.DataFrame(well_stats)
        df_stats.to_csv(os.path.join(output_dir, 'well_summary_multifeature.csv'), index=False)
        print(f"✓ Well 汇总: {os.path.join(output_dir, 'well_summary_multifeature.csv')}")

    # 生成图表
    if all_quant and all_log:
        print("\n生成分组汇总图表...")
        generate_group_summary(df_all_quant, df_all_log, output_dir)

    print("\n" + "=" * 70)
    print("多特征追踪完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
