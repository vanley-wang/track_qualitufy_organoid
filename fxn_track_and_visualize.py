"""
FXN 类器官追踪与可视化主脚本
对 FXN_0701 (Day3) 和 FXN_0703 (Day5) 的每个 well 进行：
  1. 连通域实例分割
  2. 匈牙利算法 ID 匹配
  3. 形态量化
  4. 生成 3D Slicer NRRD + 统计图表

运行方式:
    python fxn_track_and_visualize.py
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import matplotlib
matplotlib.use('Agg')  # 无头环境
import matplotlib.pyplot as plt
from skimage.measure import regionprops, marching_cubes
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg


def load_nifti(file_path):
    """加载 NIfTI 文件，返回数据和 affine"""
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.int32), nii.affine


def save_nifti(data, affine, file_path):
    """保存 NIfTI 文件"""
    nii = nib.Nifti1Image(data.astype(np.int32), affine)
    nib.save(nii, file_path)


def instance_segmentation(binary_mask, min_voxel_count=cfg.MIN_VOXEL_COUNT):
    """
    对二值 mask 进行连通域标记，生成实例分割标签
    过滤小于 min_voxel_count 的连通域
    """
    labeled, num_features = ndimage.label(binary_mask > 0)
    if num_features == 0:
        return labeled, 0

    # 获取每个连通域的体积
    component_sizes = ndimage.sum(binary_mask > 0, labeled, index=np.arange(1, num_features + 1))

    # 过滤小连通域
    valid_mask = component_sizes >= min_voxel_count
    if not np.any(valid_mask):
        return np.zeros_like(labeled), 0

    # 重新标记为连续的整数 ID
    valid_indices = np.where(valid_mask)[0] + 1  # +1 因为 label 从 1 开始
    new_labeled = np.zeros_like(labeled)
    for new_id, old_id in enumerate(valid_indices, start=1):
        new_labeled[labeled == old_id] = new_id

    return new_labeled, len(valid_indices)


def match_organoids_hungarian(props_ref, props_target, max_distance):
    """
    使用匈牙利算法进行一对一匹配
    返回: matches dict {target_label: ref_label}, unmatched_target list
    """
    if len(props_ref) == 0 or len(props_target) == 0:
        return {}, list(range(len(props_target)))

    ref_centroids = np.array([p.centroid for p in props_ref])
    target_centroids = np.array([p.centroid for p in props_target])
    ref_labels = np.array([p.label for p in props_ref])
    target_labels = np.array([p.label for p in props_target])

    # 计算距离矩阵
    dist_matrix = cdist(target_centroids, ref_centroids, metric='euclidean')

    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = {}
    unmatched_target = []

    for t_idx, s_idx in zip(row_ind, col_ind):
        distance = dist_matrix[t_idx, s_idx]
        if distance <= max_distance:
            matches[target_labels[t_idx]] = ref_labels[s_idx]
        else:
            unmatched_target.append(t_idx)

    # 未参与匈牙利算法的目标也视为未匹配
    matched_target_set = set(row_ind)
    for t_idx in range(len(props_target)):
        if t_idx not in matched_target_set:
            unmatched_target.append(t_idx)

    return matches, unmatched_target


def calculate_surface_area(mask):
    """使用 Marching Cubes 计算表面积 (μm²)"""
    try:
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
        return np.sum(areas)
    except Exception:
        return np.nan


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

    # 表面积和球形度（可选，较耗时）
    if calculate_surface:
        mask = (label_data == label_id)
        surface_area = calculate_surface_area(mask)
        if surface_area > 0 and not np.isnan(surface_area):
            ideal_surface = 4 * np.pi * (3 * volume_um3 / (4 * np.pi)) ** (2 / 3)
            sphericity = ideal_surface / surface_area
            sphericity = min(sphericity, 1.0)
            compactness = (volume_um3 ** 2) / (surface_area ** 3)
        else:
            sphericity = np.nan
            compactness = np.nan
            surface_area = np.nan
    else:
        surface_area = np.nan
        sphericity = np.nan
        compactness = np.nan

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
    """
    保存为 3D Slicer 兼容的 .seg.nrrd 格式
    为每个 Segment 设置颜色和名称
    """
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


def generate_well_plots(df, well_name, output_dir):
    """为单个 well 生成统计图表"""
    df_day3 = df[df['Day'] == 'Day3']
    df_day5 = df[df['Day'] == 'Day5']

    if len(df_day3) == 0 or len(df_day5) == 0:
        return

    # 找到共同 ID
    common_ids = sorted(set(df_day3['Organoid_ID']) & set(df_day5['Organoid_ID']))
    if len(common_ids) == 0:
        return

    # 计算体积变化率
    changes = []
    for oid in common_ids:
        v3 = df_day3[df_day3['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        v5 = df_day5[df_day5['Organoid_ID'] == oid]['Volume_mm3'].values[0]
        change_rate = (v5 - v3) / v3 if v3 > 0 else 0
        changes.append({'Organoid_ID': oid, 'Change_Rate': change_rate})
    df_change = pd.DataFrame(changes).sort_values('Change_Rate')

    # 药物反应分类
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
    fig.suptitle(f'Well {well_name} - Organoid Tracking Analysis', fontsize=14, fontweight='bold')

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


def process_well(well_name, day3_seg_path, day5_seg_path, output_dir, max_distance=cfg.MAX_DISTANCE_PIXELS, generate_plots=True):
    """
    处理单个 well 的完整流水线
    返回: (quant_df_day3, quant_df_day5, log_records)
    """
    start_time = __import__('time').time()
    print(f"\n{'='*60}")
    print(f"处理 Well: {well_name}")
    print(f"{'='*60}")

    # 1. 加载数据
    data3, affine3 = load_nifti(day3_seg_path)
    data5, affine5 = load_nifti(day5_seg_path)
    print(f"  Day3 shape: {data3.shape}, unique: {np.unique(data3)}")
    print(f"  Day5 shape: {data5.shape}, unique: {np.unique(data5)}")

    # 2. 实例分割（如果已经是多标签，跳过；如果是二值，进行连通域标记）
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

    # 3. 提取 regionprops
    props3 = regionprops(label3)
    props5 = regionprops(label5)

    # 4. 匈牙利算法匹配
    print(f"\n  匹配 Day5 → Day3 (max_distance={max_distance}px)...")
    matches, unmatched = match_organoids_hungarian(props3, props5, max_distance)
    print(f"    匹配成功: {len(matches)} 对")
    print(f"    Day5 新增: {len(unmatched)} 个")

    # 5. 应用匹配到 label5
    label5_matched = np.zeros_like(label5)
    next_new_id = max([p.label for p in props3]) + 1 if props3 else 1

    log_records = []

    # 匹配成功的
    for target_label, ref_label in matches.items():
        label5_matched[label5 == target_label] = ref_label
        # 找到对应的 regionprops 以记录质心
        t_prop = next((p for p in props5 if p.label == target_label), None)
        r_prop = next((p for p in props3 if p.label == ref_label), None)
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

    # 未匹配的（新增）
    for t_idx in unmatched:
        target_label = props5[t_idx].label
        label5_matched[label5 == target_label] = next_new_id
        log_records.append({
            'Well': well_name,
            'Day': 'Day5',
            'Original_ID': target_label,
            'New_ID': next_new_id,
            'Status': 'New',
            'Distance_pixels': np.nan,
            'Centroid_Day3': '',
            'Centroid_Day5': f"({props5[t_idx].centroid[0]:.1f}, {props5[t_idx].centroid[1]:.1f}, {props5[t_idx].centroid[2]:.1f})",
        })
        next_new_id += 1

    # 6. 量化分析
    print(f"\n  量化 Day3...")
    results_day3 = []
    for p in regionprops(label3):
        results_day3.append(quantify_organoid(p, label3, 'Day3', calculate_surface=False))

    print(f"  量化 Day5...")
    results_day5 = []
    for p in regionprops(label5_matched):
        results_day5.append(quantify_organoid(p, label5_matched, 'Day5', calculate_surface=False))

    df_day3 = pd.DataFrame(results_day3)
    df_day5 = pd.DataFrame(results_day5)

    # 7. 保存匹配后的标签
    well_output_dir = os.path.join(output_dir, well_name)
    os.makedirs(well_output_dir, exist_ok=True)

    # NIfTI 格式
    save_nifti(label3, affine3, os.path.join(well_output_dir, f'{well_name}_day3_matched.nii.gz'))
    save_nifti(label5_matched, affine5, os.path.join(well_output_dir, f'{well_name}_day5_matched.nii.gz'))

    # NRRD 格式（用于 3D Slicer）
    save_seg_nrrd(label3, os.path.join(well_output_dir, f'{well_name}_day3.seg.nrrd'))
    save_seg_nrrd(label5_matched, os.path.join(well_output_dir, f'{well_name}_day5.seg.nrrd'))

    # 合并数据
    df_all = pd.concat([df_day3, df_day5], ignore_index=True)

    # 8. 生成图表（可选）
    if generate_plots:
        generate_well_plots(df_all, well_name, well_output_dir)

    # 保存量化 CSV
    df_all.to_csv(os.path.join(well_output_dir, f'{well_name}_quantification.csv'), index=False)

    elapsed = __import__('time').time() - start_time
    print(f"  ✓ Well {well_name} 完成 ({elapsed:.1f}s)，输出至: {well_output_dir}")
    return df_day3, df_day5, log_records


def generate_summary_plots(all_quant_df, all_log_df, output_dir):
    """生成全局汇总图表"""
    if all_quant_df is None or len(all_quant_df) == 0:
        return

    df_day3 = all_quant_df[all_quant_df['Day'] == 'Day3']
    df_day5 = all_quant_df[all_quant_df['Day'] == 'Day5']

    # 跨 well 计算体积变化
    common = df_day3.merge(df_day5, on=['Well', 'Organoid_ID'], suffixes=('_d3', '_d5'))
    if len(common) == 0:
        print("  无可匹配的共同类器官，跳过全局图表")
        return

    common['Volume_Change_Rate'] = (common['Volume_mm3_d5'] - common['Volume_mm3_d3']) / common['Volume_mm3_d3']
    common = common.sort_values('Volume_Change_Rate')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FXN Day3 → Day5 Global Summary', fontsize=14, fontweight='bold')

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
        ax.axvline(cfg.MAX_DISTANCE_PIXELS, color='red', linestyle='--', label=f'Max threshold ({cfg.MAX_DISTANCE_PIXELS})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No matched organoids', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Matching Distance Distribution')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'global_summary.png'), dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 全局汇总图: {os.path.join(output_dir, 'global_summary.png')}")


def main():
    print("=" * 70)
    print("FXN 类器官追踪与可视化")
    print("Day3 (FXN_0701_seg) → Day5 (FXN_0703_seg)")
    print("=" * 70)
    cfg.print_config()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 获取 well 列表（以 Day3 seg 目录为准）
    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 Day3 seg 文件")

    all_quant_records = []
    all_log_records = []
    well_stats = []

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')  # e.g. B2_1
        day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, seg_file)
        day5_path = os.path.join(cfg.FXN_0703_SEG_DIR, seg_file)

        if not os.path.exists(day5_path):
            print(f"\n⚠️  跳过 {well_name}: Day5 文件不存在 ({day5_path})")
            continue

        df_day3, df_day5, logs = process_well(well_name, day3_path, day5_path, cfg.OUTPUT_DIR, generate_plots=False)

        if df_day3 is not None and df_day5 is not None:
            # 添加 Well 列
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

        all_log_records.extend(logs)

    # 保存全局 CSV
    if all_quant_records:
        df_all_quant = pd.concat(all_quant_records, ignore_index=True)
        df_all_quant.to_csv(os.path.join(cfg.OUTPUT_DIR, 'fxn_quantification.csv'), index=False)
        print(f"\n✓ 全局量化数据: {os.path.join(cfg.OUTPUT_DIR, 'fxn_quantification.csv')}")

    if all_log_records:
        df_all_log = pd.DataFrame(all_log_records)
        df_all_log.to_csv(os.path.join(cfg.OUTPUT_DIR, 'fxn_matching_log.csv'), index=False)
        print(f"✓ 全局匹配日志: {os.path.join(cfg.OUTPUT_DIR, 'fxn_matching_log.csv')}")

    if well_stats:
        df_stats = pd.DataFrame(well_stats)
        df_stats.to_csv(os.path.join(cfg.OUTPUT_DIR, 'fxn_well_summary.csv'), index=False)
        print(f"✓ Well 汇总: {os.path.join(cfg.OUTPUT_DIR, 'fxn_well_summary.csv')}")

    # 全局可视化
    if all_quant_records and all_log_records:
        print("\n生成全局汇总图表...")
        generate_summary_plots(df_all_quant, df_all_log, cfg.OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)
    print(f"输出目录: {cfg.OUTPUT_DIR}")
    print("\n可在 3D Slicer 中加载每个 well 的 .seg.nrrd 文件查看追踪结果")
    print("=" * 70)


if __name__ == '__main__':
    main()
