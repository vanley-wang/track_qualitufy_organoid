# 类器官胆管癌药效评估系统

## 项目概述

这是一个用于分析类器官(Organoid)在药物处理后纵向变化的完整分析系统。通过对三个时间点(Day1, Day2, Day3)的成像数据进行量化分析,评估胆管癌类器官对药物的反应。

## 研究意义

- **精准医疗**: 为患者筛选最有效的药物,实现个体化治疗
- **动态监测**: 通过连续追踪观察药物处理后类器官的生长/死亡变化
- **定量评价**: 客观、定量地评估药效,取代主观判断
- **临床转化**: 为临床治疗方案提供科学依据

## 分析流程

```
原始图像配准 → ID追踪 → 量化分析 → 纵向变化分析 → 可视化 → 药效评估
```

### 1. **ID追踪** (track_3days.py)
- 将三天的类器官进行ID匹配
- 使用质心最近邻算法
- 确保同一个类器官在不同时间点保持相同的ID

### 2. **量化分析** (quantify_organoids.py)
提取每个类器官的参数:

**体积参数:**
- 体素数量
- 实际体积 (μm³, mm³)
- 等效直径

**形态参数:**
- 表面积
- 球形度 (Sphericity)
- 长短轴比
- 紧凑度

**密度参数:**
- 平均灰度值
- 灰度标准差
- 灰度变异系数

**空间参数:**
- 质心坐标 (像素和物理坐标)
- 边界框大小

### 3. **纵向变化分析** (longitudinal_analysis.py)
计算时间序列变化:
- 体积变化率 (Growth Rate)
- 球形度变化
- 灰度值变化
- 质心位移距离

**药物反应分类:**
- **CR** (Complete Response 完全缓解): 体积缩小 > 90%
- **PR** (Partial Response 部分缓解): 体积缩小 > 50%
- **SD** (Stable Disease 疾病稳定): 体积变化 -30% ~ +30%
- **PD** (Progressive Disease 疾病进展): 体积增长 > 30%

**药效指标:**
- **ORR** (客观缓解率) = (CR + PR) / 总数
- **DCR** (疾病控制率) = (CR + PR + SD) / 总数

**统计检验:**
- 配对t检验 (Day1 vs Day3)
- Wilcoxon符号秩检验

### 4. **可视化** (visualize_results.py)
生成图表:
- 体积分布箱线图/小提琴图
- 个体类器官轨迹图
- 体积变化率直方图
- 药物反应分类饼图
- 瀑布图 (Waterfall Plot)
- 相关性散点图 (球形度vs体积, 灰度vs体积)
- 汇总统计表

## 安装依赖

```bash
pip install numpy pandas nibabel pynrrd scikit-image scipy matplotlib seaborn
```

或使用提供的requirements文件:
```bash
pip install -r requirements.txt
```

## 配置

在 `config.py` 中配置:

### 重要参数:
```python
# 体素物理尺寸 (微米)
VOXEL_SIZE_X = 5.0  # μm
VOXEL_SIZE_Y = 5.0  # μm
VOXEL_SIZE_Z = 5.5  # μm

# 图像尺寸 (像素)
IMAGE_SIZE_X = 800
IMAGE_SIZE_Y = 800
IMAGE_SIZE_Z = 512

# 数据路径
DATA_DIR = "/path/to/your/data"
OUTPUT_DIR = "/path/to/output"

# 文件名
DAY1_LABEL = "organoid_031_tracked.nii.seg.nrrd"
DAY2_LABEL = "organoid_043_tracked.nii.seg.nrrd"
DAY3_LABEL = "organoid_062_tracked.nii.seg.nrrd"

DAY1_IMAGE = "organoid_031_0000.nii.gz"
DAY2_IMAGE = "organoid_043_0000.nii.gz"
DAY3_IMAGE = "organoid_062_0000.nii.gz"

# 药效评估阈值
THRESHOLD_COMPLETE_RESPONSE = -0.9  # 缩小90%
THRESHOLD_PARTIAL_RESPONSE = -0.5   # 缩小50%
THRESHOLD_STABLE_DISEASE_MIN = -0.3
THRESHOLD_STABLE_DISEASE_MAX = 0.3
```

## 使用方法

### 方法1: 一键运行 (推荐)

```bash
python run_full_analysis.py
```

这会自动执行所有步骤:
1. ID追踪
2. 量化分析
3. 纵向变化分析
4. 可视化

### 方法2: 分步运行

```bash
# 1. ID追踪
python track_3days.py

# 2. 量化分析
python quantify_organoids.py

# 3. 纵向分析
python longitudinal_analysis.py

# 4. 可视化
python visualize_results.py
```

### 方法3: 在代码中调用

```python
import run_full_analysis

success = run_full_analysis.run_full_analysis(
    day1_label_path="/path/to/day1.nrrd",
    day2_label_path="/path/to/day2.nrrd",
    day3_label_path="/path/to/day3.nrrd",
    output_dir="/path/to/output",
    calculate_surface=True,  # 是否计算表面积
    max_distance_pixels=50   # ID匹配最大距离
)
```

## 输出文件

运行完成后,在输出目录会生成:

### CSV数据文件:
- `organoid_quantification.csv` - 所有时间点的量化数据
- `organoid_longitudinal_analysis.csv` - 纵向变化分析结果
- `organoid_summary_statistics.csv` - 汇总统计
- `statistical_tests.csv` - 统计检验结果
- `matching_log_day2.csv`, `matching_log_day3.csv` - ID匹配日志

### 追踪后的标签文件:
- `organoid_043_tracked.nii.gz` (Day2)
- `organoid_062_tracked.nii.gz` (Day3)

### 可视化图表:
- `volume_distribution.png` - 体积分布
- `individual_trajectories.png` - 个体轨迹
- `growth_rate_histogram.png` - 变化率直方图
- `drug_response_pie.png` - 药物反应饼图
- `waterfall_plot.png` - 瀑布图
- `correlation_plots.png` - 相关性分析

## 数据要求

### 输入文件:
1. **标签文件**: 分割后的类器官标签 (NIfTI或NRRD格式)
   - 每个类器官有唯一的整数标签
   - 背景为0
   
2. **原始成像文件**: 用于提取灰度值 (可选但推荐)
   - NIfTI格式 (.nii, .nii.gz)

### 数据要求:
- 三个时间点的数据必须已经配准到同一坐标系
- 标签ID可以不一致 (脚本会自动匹配)
- 推荐分辨率: 5μm × 5μm × 5.5μm

## 量化参数说明

### 为什么要量化这些参数?

1. **体积**: 直接反映肿瘤细胞的增殖或死亡
   - 缩小 → 药物有效
   - 增长 → 药物无效或耐药

2. **球形度**: 反映类器官的结构完整性
   - 下降 → 可能提示细胞凋亡、结构崩解
   - 保持 → 结构稳定

3. **灰度值**: 反映细胞密度
   - 下降 → 可能细胞死亡、空泡化
   - 增加 → 细胞增殖、密度增加

4. **质心位移**: 类器官的移动距离
   - 用于验证ID匹配的准确性
   - 过大位移可能提示匹配错误

## 常见问题

### Q1: 如何确定药效评估阈值?
A: 默认阈值参考RECIST标准,但可根据您的具体研究调整`config.py`中的参数。

### Q2: 表面积计算很慢怎么办?
A: 可以设置`calculate_surface=False`跳过表面积计算,会更快但无法得到球形度。

### Q3: 如果文件名不同怎么办?
A: 修改`config.py`中的文件名配置,或直接传入路径参数。

### Q4: 如何处理新出现或消失的类器官?
A: 脚本会自动检测并标记:
- `Persistent`: 持续存在 (用于药效评估)
- `New`: Day3新出现
- `Lost`: Day1后消失

### Q5: 统计检验的p值如何解释?
A: p < 0.05表示差异显著。配对t检验用于正态分布数据,Wilcoxon检验用于非正态分布。

## 论文撰写建议

### 方法部分:
```
图像采集后,使用[配准方法]将三个时间点的数据配准到同一坐标系。
通过[分割方法]自动分割类器官。使用质心最近邻算法(最大距离阈值250μm)
进行ID追踪。提取形态学参数包括体积(mm³)、球形度、表面积等,以及
灰度值参数。根据Day1到Day3的体积变化率分类药物反应:CR(缩小>90%)、
PR(缩小>50%)、SD(-30%~+30%)、PD(增长>30%)。使用配对t检验
分析Day1与Day3的差异。
```

### 结果部分:
```
共检测到XX个类器官,其中XX个在三天内持续存在。平均体积变化率为XX%。
药物反应分类:CR XX例(XX%)、PR XX例(XX%)、SD XX例(XX%)、
PD XX例(XX%)。客观缓解率(ORR)为XX%,疾病控制率(DCR)为XX%。
Day3相比Day1的体积显著下降(p<0.05)。
```

## 联系方式

如有问题或建议,请联系: [您的邮箱]

## 更新日志

- 2025-12-24: 初始版本
  - 完成ID追踪、量化分析、纵向分析、可视化功能
  - 支持物理尺寸转换
  - 实现药效评估分类

## 许可

[您的许可类型]

---

**祝研究顺利! 🎉**
