"""
FXN 分割质量验证脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对每个 well 的分割 mask 计算质量指标，输出报告和可视化

运行方式:
    python fxn_validate_segmentation.py
    python fxn_validate_segmentation.py --well B9_1 F10_1   # 只验证指定 well

输出:
    results_validation/
    ├── validation_summary.csv      # 全局汇总表
    ├── validation_report.html      # 可视化报告
    └── B2_1/
        ├── B2_1_validation.png     # 质量指标图
        └── B2_1_issues.csv         # 疑似问题对象列表
"""

import os
import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import regionprops
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg


# =============================================================================
# 命令行参数
# =============================================================================
parser = argparse.ArgumentParser(description="Validate organoid segmentation quality")
parser.add_argument("--well", nargs='+', default=None,
                    help="指定验证的 well，默认全部")
parser.add_argument("--min-voxels", type=int, default=4000,
                    help="最小连通域体素数 (默认: 4000)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="输出目录 (默认: results_validation)")
ARGS = parser.parse_args()

OUTPUT_DIR = ARGS.output_dir or os.path.join(cfg.BASE_DIR, "results_validation")
MIN_VOXELS = ARGS.min_voxels


# =============================================================================
# 核心验证指标
# =============================================================================
def validate_segmentation(seg_path, day_name, well_name, min_voxels=MIN_VOXELS):
    """
    计算单个分割文件的质量指标
    返回: (summary_dict, issue_records, per_object_df)
    """
    data = np.asanyarray(nib.load(seg_path).dataobj).astype(np.uint8)

    # 如果是二值图，做连通域标记
    uniques = np.unique(data)
    if len(uniques) <= 2 and np.array_equal(uniques, [0, 1]):
        labeled, n_total = ndimage.label(data > 0)
    else:
        labeled = data.astype(np.int32)
        n_total = len(uniques) - 1

    if n_total == 0:
        return {
            'Well': well_name, 'Day': day_name,
            'Total_Objects': 0, 'Big_Objects': 0,
            'Small_Objects': 0, 'Total_Voxels': 0,
            'Mean_Size': 0, 'Median_Size': 0, 'Std_Size': 0,
            'CV_Size': 0, 'Max_Size': 0, 'Min_Size': 0,
            'Mean_Sphericity': 0, 'Mean_Aspect_Ratio': 0,
            'Fragmentation_Index': 0, 'Giant_Object_Ratio': 0,
            'Merge_Suspect_Count': 0, 'Touching_Border_Ratio': 0,
        }, [], pd.DataFrame()

    props = regionprops(labeled)
    sizes = np.array([p.area for p in props])
    big_mask = sizes >= min_voxels
    big_sizes = sizes[big_mask]
    small_sizes = sizes[~big_mask]

    # 逐个对象计算指标
    records = []
    issue_records = []
    for p in props:
        sz = p.area
        bbox = p.bbox  # (z0, y0, x0, z1, y1, x1)
        bbox_dims = np.array([bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]])
        max_dim = max(bbox_dims) if max(bbox_dims) > 0 else 1
        min_dim = min(bbox_dims) if min(bbox_dims) > 0 else 1
        aspect_ratio = max_dim / min_dim

        # 等效球形度（快速估算，无需 marching cubes）
        eq_diam = p.equivalent_diameter
        ideal_vol = (4/3) * np.pi * (eq_diam/2)**3
        actual_vol = sz
        # 用体积比估算 compactness
        compactness = (ideal_vol / actual_vol) if actual_vol > 0 else 0
        compactness = min(compactness, 1.0)

        # 接触边界的比例（边界对象可能欠分割）
        img_shape = np.array(labeled.shape)
        touches_border = (
            bbox[0] == 0 or bbox[1] == 0 or bbox[2] == 0 or
            bbox[3] >= img_shape[0] or bbox[4] >= img_shape[1] or bbox[5] >= img_shape[2]
        )

        records.append({
            'Well': well_name, 'Day': day_name,
            'Label': p.label, 'Size': sz,
            'Centroid_Z': p.centroid[0], 'Centroid_Y': p.centroid[1], 'Centroid_X': p.centroid[2],
            'BBox_Z': bbox_dims[0], 'BBox_Y': bbox_dims[1], 'BBox_X': bbox_dims[2],
            'Aspect_Ratio': aspect_ratio,
            'Compactness': compactness,
            'Touches_Border': touches_border,
            'Is_Big': sz >= min_voxels,
        })

        # 问题检测规则
        issues = []
        # 1. 超巨大对象（疑似欠分割/合并）
        if sz >= min_voxels * 20:
            issues.append('Giant_Undersegmented')
        # 2. 极度扁平（aspect ratio > 10）
        if aspect_ratio > 10:
            issues.append('Extremely_Flat')
        # 3. 极低 compactness（< 0.1，可能空洞严重或边缘破碎）
        if compactness < 0.1:
            issues.append('Low_Compactness')
        # 4. 边界接触 + 大尺寸（可能图像边缘截断）
        if touches_border and sz >= min_voxels * 5:
            issues.append('Border_Truncated')
        # 5. 极小碎片（< 100 体素，可能是噪声）
        if sz < 100:
            issues.append('Tiny_Noise')

        if issues:
            issue_records.append({
                'Well': well_name, 'Day': day_name,
                'Label': p.label, 'Size': sz,
                'Issues': '|'.join(issues),
                'Aspect_Ratio': round(aspect_ratio, 2),
                'Compactness': round(compactness, 4),
                'Touches_Border': touches_border,
            })

    df_obj = pd.DataFrame(records)

    # 汇总指标
    total_voxels = int(np.sum(sizes))
    mean_size = float(np.mean(sizes)) if len(sizes) > 0 else 0
    median_size = float(np.median(sizes)) if len(sizes) > 0 else 0
    std_size = float(np.std(sizes)) if len(sizes) > 0 else 0
    cv_size = std_size / mean_size if mean_size > 0 else 0

    # 碎片化指数 = 小对象数 / 大对象数（越高说明过分割越严重）
    frag_index = len(small_sizes) / max(len(big_sizes), 1)

    # 巨对象比例 = 最大对象体积 / 总前景体积
    giant_ratio = float(np.max(sizes)) / total_voxels if total_voxels > 0 else 0

    # 疑似合并对象数（超巨大的）
    merge_suspects = sum(1 for s in sizes if s >= min_voxels * 20)

    # 边界接触比例
    border_ratio = sum(1 for r in records if r['Touches_Border']) / max(len(records), 1)

    # 大对象的平均 sphericality / compactness
    big_df = df_obj[df_obj['Is_Big'] == True]
    mean_compact = float(big_df['Compactness'].mean()) if len(big_df) > 0 else 0
    mean_aspect = float(big_df['Aspect_Ratio'].mean()) if len(big_df) > 0 else 0

    summary = {
        'Well': well_name,
        'Day': day_name,
        'Total_Objects': n_total,
        'Big_Objects': int(np.sum(big_mask)),
        'Small_Objects': int(np.sum(~big_mask)),
        'Total_Voxels': total_voxels,
        'Mean_Size': round(mean_size, 1),
        'Median_Size': round(median_size, 1),
        'Std_Size': round(std_size, 1),
        'CV_Size': round(cv_size, 3),
        'Max_Size': int(np.max(sizes)) if len(sizes) > 0 else 0,
        'Min_Size': int(np.min(sizes)) if len(sizes) > 0 else 0,
        'Mean_Compactness': round(mean_compact, 4),
        'Mean_Aspect_Ratio': round(mean_aspect, 2),
        'Fragmentation_Index': round(frag_index, 2),
        'Giant_Object_Ratio': round(giant_ratio, 4),
        'Merge_Suspect_Count': merge_suspects,
        'Touching_Border_Ratio': round(border_ratio, 3),
    }

    return summary, issue_records, df_obj


# =============================================================================
# 可视化
# =============================================================================
def plot_validation(df_obj, summary, well_name, day_name, output_dir):
    """为单个 well 生成分割质量验证图"""
    if len(df_obj) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{well_name} {day_name} - Segmentation Validation', fontsize=14, fontweight='bold')

    big_df = df_obj[df_obj['Is_Big'] == True]
    small_df = df_obj[df_obj['Is_Big'] == False]

    # 1. 体积分布直方图
    ax = axes[0, 0]
    bins = np.logspace(1, np.log10(max(df_obj['Size'].max(), 100)), 30)
    ax.hist(small_df['Size'], bins=bins, color='gray', alpha=0.6, label=f'Small (<{MIN_VOXELS})')
    ax.hist(big_df['Size'], bins=bins, color='steelblue', alpha=0.8, label=f'Big (≥{MIN_VOXELS})')
    ax.axvline(MIN_VOXELS, color='red', linestyle='--', label=f'Threshold={MIN_VOXELS}')
    ax.set_xscale('log')
    ax.set_xlabel('Size (voxels)')
    ax.set_ylabel('Count')
    ax.set_title(f'Size Distribution (n={len(df_obj)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 体积排序图（瀑布图）
    ax = axes[0, 1]
    sorted_sizes = df_obj.sort_values('Size', ascending=False)
    colors_bar = ['steelblue' if s else 'gray' for s in sorted_sizes['Is_Big']]
    ax.bar(range(len(sorted_sizes)), sorted_sizes['Size'], color=colors_bar, edgecolor='black', linewidth=0.3)
    ax.axhline(MIN_VOXELS, color='red', linestyle='--')
    ax.set_xlabel('Object Index (sorted by size)')
    ax.set_ylabel('Size (voxels)')
    ax.set_title('Size Waterfall')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Aspect Ratio 分布
    ax = axes[0, 2]
    if len(big_df) > 0:
        ax.hist(big_df['Aspect_Ratio'], bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(10, color='red', linestyle='--', label='Suspicious >10')
    ax.set_xlabel('Aspect Ratio (max/min bbox dim)')
    ax.set_ylabel('Count')
    ax.set_title(f'Aspect Ratio (Big objects)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Compactness vs Size
    ax = axes[1, 0]
    ax.scatter(big_df['Size'], big_df['Compactness'], c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(0.1, color='red', linestyle='--', label='Low compactness <0.1')
    ax.set_xlabel('Size (voxels)')
    ax.set_ylabel('Compactness')
    ax.set_title('Compactness vs Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 空间分布（Centroid YX 投影）
    ax = axes[1, 1]
    if len(big_df) > 0:
        sc = ax.scatter(big_df['Centroid_X'], big_df['Centroid_Y'],
                        c=big_df['Size'], cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(sc, ax=ax, label='Size (voxels)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Spatial Distribution (YX projection)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 6. 质量评分面板
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = (
        f"Objects: {summary['Total_Objects']}\n"
        f"Big (≥{MIN_VOXELS}): {summary['Big_Objects']}\n"
        f"Small: {summary['Small_Objects']}\n"
        f"Total Voxels: {summary['Total_Voxels']:,}\n"
        f"Mean Size: {summary['Mean_Size']:.0f}\n"
        f"CV Size: {summary['CV_Size']:.3f}\n"
        f"Fragmentation Idx: {summary['Fragmentation_Index']:.2f}\n"
        f"Giant Ratio: {summary['Giant_Object_Ratio']:.4f}\n"
        f"Merge Suspects: {summary['Merge_Suspect_Count']}\n"
        f"Border Touch: {summary['Touching_Border_Ratio']:.1%}\n"
        f"Mean Compactness: {summary['Mean_Compactness']:.4f}\n"
        f"Mean Aspect Ratio: {summary['Mean_Aspect_Ratio']:.2f}"
    )
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Quality Metrics')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f'{well_name}_{day_name}_validation.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 验证图: {out_path}")


# =============================================================================
# 全局汇总报告
# =============================================================================
def generate_global_report(all_summary, all_issues, output_dir):
    """生成全局 HTML 报告和汇总 CSV"""
    df_sum = pd.DataFrame(all_summary)
    df_sum.to_csv(os.path.join(output_dir, 'validation_summary.csv'), index=False)
    print(f"\n✓ 全局汇总: {os.path.join(output_dir, 'validation_summary.csv')}")

    if all_issues:
        df_issues = pd.DataFrame(all_issues)
        df_issues.to_csv(os.path.join(output_dir, 'all_issues.csv'), index=False)
        print(f"✓ 问题列表: {os.path.join(output_dir, 'all_issues.csv')}")

    # 生成对比图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Global Segmentation Quality Summary', fontsize=14, fontweight='bold')

    # 按 Group 分组
    def get_group(well):
        prefix = well.split('_')[0]
        return cfg.WELL_GROUPS.get(prefix, 'Unknown')
    df_sum['Group'] = df_sum['Well'].apply(get_group)

    groups = ['Control', '20uM', '40uM', '80uM']

    # 1. 平均对象数
    ax = axes[0, 0]
    grp_counts = df_sum.groupby(['Group', 'Day'])['Big_Objects'].mean().unstack()
    grp_counts = grp_counts.reindex(groups)
    grp_counts.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71'], edgecolor='black')
    ax.set_ylabel('Mean Big Objects')
    ax.set_title('Mean Organoid Count by Group')
    ax.legend(title='Day')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 碎片化指数
    ax = axes[0, 1]
    grp_frag = df_sum.groupby(['Group', 'Day'])['Fragmentation_Index'].mean().unstack()
    grp_frag = grp_frag.reindex(groups)
    grp_frag.plot(kind='bar', ax=ax, color=['#e74c3c', '#f39c12'], edgecolor='black')
    ax.set_ylabel('Fragmentation Index')
    ax.set_title('Fragmentation (Small/Big ratio)')
    ax.legend(title='Day')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 体积变异系数 CV
    ax = axes[0, 2]
    grp_cv = df_sum.groupby(['Group', 'Day'])['CV_Size'].mean().unstack()
    grp_cv = grp_cv.reindex(groups)
    grp_cv.plot(kind='bar', ax=ax, color=['#9b59b6', '#1abc9c'], edgecolor='black')
    ax.set_ylabel('CV of Size')
    ax.set_title('Size Variability (CV)')
    ax.legend(title='Day')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 疑似合并对象数
    ax = axes[1, 0]
    grp_merge = df_sum.groupby(['Group', 'Day'])['Merge_Suspect_Count'].mean().unstack()
    grp_merge = grp_merge.reindex(groups)
    grp_merge.plot(kind='bar', ax=ax, color=['#e67e22', '#34495e'], edgecolor='black')
    ax.set_ylabel('Merge Suspects')
    ax.set_title('Undersegmentation Suspects')
    ax.legend(title='Day')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 巨对象比例
    ax = axes[1, 1]
    grp_giant = df_sum.groupby(['Group', 'Day'])['Giant_Object_Ratio'].mean().unstack()
    grp_giant = grp_giant.reindex(groups)
    grp_giant.plot(kind='bar', ax=ax, color=['#c0392b', '#16a085'], edgecolor='black')
    ax.set_ylabel('Giant Object Ratio')
    ax.set_title('Dominant Object Size')
    ax.legend(title='Day')
    ax.grid(True, alpha=0.3, axis='y')

    # 6. 问题统计
    ax = axes[1, 2]
    if all_issues:
        issue_counts = df_issues.groupby(['Day', 'Issues']).size().unstack(fill_value=0)
        issue_counts.plot(kind='bar', stacked=True, ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Issue Count')
        ax.set_title('Detected Issues')
        ax.legend(title='Issue Type', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No issues detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Detected Issues')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'global_validation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 全局图: {os.path.join(output_dir, 'global_validation_summary.png')}")

    # 生成简单的 HTML 报告
    html = f"""<!DOCTYPE html>
<html>
<head><title>FXN Segmentation Validation Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #3498db; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    .warning {{ color: #e74c3c; font-weight: bold; }}
    .ok {{ color: #27ae60; }}
</style>
</head>
<body>
<h1>FXN 分割质量验证报告</h1>
<p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>全局汇总</h2>
<img src="global_validation_summary.png" style="max-width:100%;">
<h2>详细数据</h2>
<table>
<tr><th>Well</th><th>Day</th><th>Big Objects</th><th>Fragmentation</th><th>CV Size</th><th>Merge Suspects</th><th>Issues</th></tr>
"""
    for _, row in df_sum.iterrows():
        issues_for_well = [i for i in all_issues if i['Well'] == row['Well'] and i['Day'] == row['Day']]
        issue_str = f"{len(issues_for_well)} issues" if issues_for_well else "OK"
        issue_class = "warning" if issues_for_well else "ok"
        html += f"""
<tr>
    <td>{row['Well']}</td>
    <td>{row['Day']}</td>
    <td>{row['Big_Objects']}</td>
    <td>{row['Fragmentation_Index']:.2f}</td>
    <td>{row['CV_Size']:.3f}</td>
    <td>{row['Merge_Suspect_Count']}</td>
    <td class="{issue_class}">{issue_str}</td>
</tr>
"""
    html += "</table></body></html>"

    with open(os.path.join(output_dir, 'validation_report.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ HTML 报告: {os.path.join(output_dir, 'validation_report.html')}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("FXN 分割质量验证")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wells = ARGS.well
    if wells is None:
        day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
        wells = [f.replace('.nii.gz', '') for f in day3_files]

    print(f"验证 {len(wells)} 个 well，输出至: {OUTPUT_DIR}\n")

    all_summary = []
    all_issues = []

    for well_name in wells:
        day3_path = os.path.join(cfg.FXN_0701_SEG_DIR, f'{well_name}.nii.gz')
        day5_path = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

        if not os.path.exists(day3_path) or not os.path.exists(day5_path):
            print(f"⚠️ 跳过 {well_name}: 文件缺失")
            continue

        print(f"验证 {well_name} ...")
        well_dir = os.path.join(OUTPUT_DIR, well_name)
        os.makedirs(well_dir, exist_ok=True)

        # Day3
        s3, i3, d3 = validate_segmentation(day3_path, 'Day3', well_name)
        all_summary.append(s3)
        all_issues.extend(i3)
        plot_validation(d3, s3, well_name, 'Day3', well_dir)

        if i3:
            pd.DataFrame(i3).to_csv(os.path.join(well_dir, f'{well_name}_issues_day3.csv'), index=False)

        # Day5
        s5, i5, d5 = validate_segmentation(day5_path, 'Day5', well_name)
        all_summary.append(s5)
        all_issues.extend(i5)
        plot_validation(d5, s5, well_name, 'Day5', well_dir)

        if i5:
            pd.DataFrame(i5).to_csv(os.path.join(well_dir, f'{well_name}_issues_day5.csv'), index=False)

    # 全局报告
    print("\n" + "=" * 70)
    print("生成全局报告...")
    generate_global_report(all_summary, all_issues, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("验证完成！")
    print(f"用浏览器打开: {os.path.join(OUTPUT_DIR, 'validation_report.html')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
