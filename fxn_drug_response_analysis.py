"""
FXN 药效响应分析脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于追踪结果（Day3 → Day5）计算个体和群体级别的药效响应

运行方式:
    python fxn_drug_response_analysis.py

输出:
    results_drug_response/
    ├── organoid_response.csv           # 每个匹配对的响应
    ├── well_response_summary.csv       # 每孔汇总
    ├── group_response_comparison.csv   # 组间对比
    ├── waterfall_by_group.png          # 分组瀑布图
    └── group_boxplot.png               # 组间箱线图
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg

OUTPUT_DIR = os.path.join(cfg.BASE_DIR, "results_drug_response")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_group(well_name):
    prefix = well_name.split('_')[0]
    return cfg.WELL_GROUPS.get(prefix, 'Unknown')


def classify_response(rate):
    """按体积变化率分类药效响应"""
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


def main():
    print("=" * 70)
    print("FXN 药效响应分析")
    print("=" * 70)

    # 1. 读取追踪后的量化数据
    robust_dir = os.path.join(cfg.BASE_DIR, "results_fxn_robust")
    if not os.path.exists(robust_dir):
        print(f"错误: 追踪结果目录不存在: {robust_dir}")
        print("请先运行 fxn_track_robust.py")
        sys.exit(1)

    # 收集所有 well 的匹配对数据
    organoid_records = []
    well_records = []

    well_dirs = [d for d in os.listdir(robust_dir) if os.path.isdir(os.path.join(robust_dir, d))]

    for well_name in sorted(well_dirs):
        csv_path = os.path.join(robust_dir, well_name, f'{well_name}_quantification.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df_day3 = df[df['Day'] == 'Day3']
        df_day5 = df[df['Day'] == 'Day5']

        if len(df_day3) == 0 or len(df_day5) == 0:
            continue

        group = get_group(well_name)

        # 找到共同 ID（即追踪匹配上的）
        common_ids = sorted(set(df_day3['Organoid_ID']) & set(df_day5['Organoid_ID']))

        well_cr = well_pr = well_sd = well_pd = 0
        well_changes = []

        for oid in common_ids:
            v3 = df_day3[df_day3['Organoid_ID'] == oid]['Volume_mm3'].values[0]
            v5 = df_day5[df_day5['Organoid_ID'] == oid]['Volume_mm3'].values[0]
            voxel3 = df_day3[df_day3['Organoid_ID'] == oid]['Voxel_Count'].values[0]
            voxel5 = df_day5[df_day5['Organoid_ID'] == oid]['Voxel_Count'].values[0]

            change_rate = (v5 - v3) / v3 if v3 > 0 else 0
            response = classify_response(change_rate)

            if response == 'CR':
                well_cr += 1
            elif response == 'PR':
                well_pr += 1
            elif response == 'SD':
                well_sd += 1
            else:
                well_pd += 1

            well_changes.append(change_rate)

            organoid_records.append({
                'Well': well_name,
                'Group': group,
                'Organoid_ID': oid,
                'Volume_mm3_Day3': v3,
                'Volume_mm3_Day5': v5,
                'Voxel_Count_Day3': voxel3,
                'Voxel_Count_Day5': voxel5,
                'Volume_Change_Rate': change_rate,
                'Response': response,
            })

        n_total = len(common_ids)
        well_records.append({
            'Well': well_name,
            'Group': group,
            'Matched_Count': n_total,
            'Mean_Volume_Change_Rate': np.mean(well_changes) if well_changes else np.nan,
            'Median_Volume_Change_Rate': np.median(well_changes) if well_changes else np.nan,
            'Std_Volume_Change_Rate': np.std(well_changes) if well_changes else np.nan,
            'CR_Count': well_cr,
            'PR_Count': well_pr,
            'SD_Count': well_sd,
            'PD_Count': well_pd,
            'CR_Ratio': well_cr / n_total if n_total > 0 else 0,
            'PR_Ratio': well_pr / n_total if n_total > 0 else 0,
            'SD_Ratio': well_sd / n_total if n_total > 0 else 0,
            'PD_Ratio': well_pd / n_total if n_total > 0 else 0,
        })

    # 保存个体响应表
    df_org = pd.DataFrame(organoid_records)
    df_org.to_csv(os.path.join(OUTPUT_DIR, 'organoid_response.csv'), index=False)
    print(f"✓ 个体响应表: {os.path.join(OUTPUT_DIR, 'organoid_response.csv')} ({len(df_org)} 条)")

    # 保存 well 汇总表
    df_well = pd.DataFrame(well_records)
    df_well.to_csv(os.path.join(OUTPUT_DIR, 'well_response_summary.csv'), index=False)
    print(f"✓ 孔板汇总表: {os.path.join(OUTPUT_DIR, 'well_response_summary.csv')}")

    # 2. 组间对比统计
    group_records = []
    for group in ['Control', '20uM', '40uM', '80uM']:
        sub = df_org[df_org['Group'] == group]
        if len(sub) == 0:
            continue

        rates = sub['Volume_Change_Rate'].values
        group_records.append({
            'Group': group,
            'N_Organoids': len(sub),
            'Mean_Change_Rate': np.mean(rates),
            'Median_Change_Rate': np.median(rates),
            'Std_Change_Rate': np.std(rates),
            'Min_Change_Rate': np.min(rates),
            'Max_Change_Rate': np.max(rates),
            'CR_Ratio': (sub['Response'] == 'CR').mean(),
            'PR_Ratio': (sub['Response'] == 'PR').mean(),
            'SD_Ratio': (sub['Response'] == 'SD').mean(),
            'PD_Ratio': (sub['Response'] == 'PD').mean(),
        })

    df_group = pd.DataFrame(group_records)
    df_group.to_csv(os.path.join(OUTPUT_DIR, 'group_response_comparison.csv'), index=False)
    print(f"✓ 组间对比表: {os.path.join(OUTPUT_DIR, 'group_response_comparison.csv')}")

    # 打印组间统计
    print("\n" + "=" * 70)
    print("组间药效响应对比")
    print("=" * 70)
    print(df_group[['Group', 'N_Organoids', 'Mean_Change_Rate', 'Median_Change_Rate',
                    'CR_Ratio', 'PR_Ratio', 'SD_Ratio', 'PD_Ratio']].to_string(index=False))

    # 3. 统计检验（Control vs 各浓度组）
    print("\n" + "=" * 70)
    print("统计检验 (Mann-Whitney U, vs Control)")
    print("=" * 70)
    control_rates = df_org[df_org['Group'] == 'Control']['Volume_Change_Rate'].values
    for group in ['20uM', '40uM', '80uM']:
        group_rates = df_org[df_org['Group'] == group]['Volume_Change_Rate'].values
        if len(group_rates) > 0 and len(control_rates) > 0:
            statistic, pvalue = stats.mannwhitneyu(control_rates, group_rates, alternative='two-sided')
            print(f"  Control vs {group:4s}: p = {pvalue:.6f} {'***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else 'ns'}")

    # 4. 可视化
    print("\n生成可视化...")

    # 图 1: 分组瀑布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FXN Drug Response Waterfall by Group', fontsize=14, fontweight='bold')

    for idx, group in enumerate(['Control', '20uM', '40uM', '80uM']):
        ax = axes[idx // 2, idx % 2]
        sub = df_org[df_org['Group'] == group].sort_values('Volume_Change_Rate')
        colors = [cfg.COLOR_CR if r == 'CR' else cfg.COLOR_PR if r == 'PR'
                  else cfg.COLOR_SD if r == 'SD' else cfg.COLOR_PD for r in sub['Response']]
        ax.bar(range(len(sub)), sub['Volume_Change_Rate'] * 100, color=colors, edgecolor='black', linewidth=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(cfg.THRESHOLD_COMPLETE_RESPONSE * 100, color='green', linestyle='--', alpha=0.5)
        ax.axhline(cfg.THRESHOLD_PARTIAL_RESPONSE * 100, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(cfg.THRESHOLD_PROGRESSIVE_DISEASE * 100, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Organoid (sorted)')
        ax.set_ylabel('Volume Change (%)')
        ax.set_title(f'{group} (n={len(sub)})')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'waterfall_by_group.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 瀑布图: {os.path.join(OUTPUT_DIR, 'waterfall_by_group.png')}")

    # 图 2: 组间箱线图 + 响应堆叠柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('FXN Drug Response Group Comparison', fontsize=14, fontweight='bold')

    ax = axes[0]
    group_data = [df_org[df_org['Group'] == g]['Volume_Change_Rate'].values * 100
                  for g in ['Control', '20uM', '40uM', '80uM']]
    bp = ax.boxplot(group_data, labels=['Control', '20uM', '40uM', '80uM'], patch_artist=True)
    colors = [cfg.GROUP_COLORS[g] for g in ['Control', '20uM', '40uM', '80uM']]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(cfg.THRESHOLD_COMPLETE_RESPONSE * 100, color='green', linestyle='--', alpha=0.5, label='CR threshold')
    ax.axhline(cfg.THRESHOLD_PROGRESSIVE_DISEASE * 100, color='red', linestyle='--', alpha=0.5, label='PD threshold')
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('Volume Change Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    resp = df_org.groupby(['Group', 'Response']).size().unstack(fill_value=0)
    resp = resp.reindex(['Control', '20uM', '40uM', '80uM'])
    resp = resp[['CR', 'PR', 'SD', 'PD']] if all(c in resp.columns for c in ['CR', 'PR', 'SD', 'PD']) else resp
    resp.plot(kind='bar', stacked=True, ax=ax,
              color=[cfg.COLOR_CR, cfg.COLOR_PR, cfg.COLOR_SD, cfg.COLOR_PD],
              edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Count')
    ax.set_title('Response Distribution')
    ax.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'group_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 箱线图: {os.path.join(OUTPUT_DIR, 'group_boxplot.png')}")

    print("\n" + "=" * 70)
    print("药效分析完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
