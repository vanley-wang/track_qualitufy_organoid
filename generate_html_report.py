"""
生成交互式HTML可视化报告
"""
import pandas as pd
import numpy as np
import json

# 读取追踪日志
df = pd.read_csv('results/trackpy_tracking_log.csv')

# 找到完整追踪的类器官
complete_particles = df.groupby('particle')['frame'].nunique()
complete_particles = complete_particles[complete_particles == 3].index

print("生成HTML可视化报告...")

# 准备数据
tracking_data = []
for pid in sorted(complete_particles):
    track = df[df['particle'] == pid].sort_values('frame')
    
    day1_data = track[track['frame'] == 0].iloc[0]
    day2_data = track[track['frame'] == 1].iloc[0]
    day3_data = track[track['frame'] == 2].iloc[0]
    
    coords1 = [float(day1_data['x']), float(day1_data['y']), float(day1_data['z'])]
    coords3 = [float(day3_data['x']), float(day3_data['y']), float(day3_data['z'])]
    displacement = np.linalg.norm(np.array(coords3) - np.array(coords1))
    
    vol_change = (day3_data['volume'] - day1_data['volume']) / day1_data['volume'] * 100
    
    tracking_data.append({
        'particle': int(pid),
        'day1_label': int(day1_data['label']),
        'day1_volume': int(day1_data['volume']),
        'day2_volume': int(day2_data['volume']),
        'day3_volume': int(day3_data['volume']),
        'volume_change_pct': float(vol_change),
        'displacement': float(displacement),
        'day1_coords': coords1,
        'day2_coords': [float(day2_data['x']), float(day2_data['y']), float(day2_data['z'])],
        'day3_coords': coords3
    })

# 生成HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Organoid Tracking Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            font-size: 18px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .image-card img {{
            width: 100%;
            height: auto;
        }}
        .image-caption {{
            padding: 10px;
            background-color: #f9f9f9;
            font-size: 14px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Organoid Tracking Results (Trackpy)</h1>
        
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <div class="summary-item">✅ Total tracked organoids (Day1→Day2→Day3): <strong>{len(tracking_data)}</strong></div>
            <div class="summary-item">📈 Mean volume change: <strong>{np.mean([d['volume_change_pct'] for d in tracking_data]):.1f}%</strong></div>
            <div class="summary-item">📏 Mean displacement: <strong>{np.mean([d['displacement'] for d in tracking_data]):.1f} pixels</strong></div>
            <div class="summary-item">📏 Max displacement: <strong>{np.max([d['displacement'] for d in tracking_data]):.1f} pixels</strong></div>
        </div>
        
        <h2>📋 Individual Tracking Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Day1<br>Label</th>
                    <th>Volume Day1<br>(voxels)</th>
                    <th>Volume Day2<br>(voxels)</th>
                    <th>Volume Day3<br>(voxels)</th>
                    <th>Volume Change<br>(%)</th>
                    <th>Displacement<br>(pixels)</th>
                </tr>
            </thead>
            <tbody>
"""

for i, data in enumerate(tracking_data, 1):
    vol_change_class = 'positive' if data['volume_change_pct'] > 0 else 'negative'
    html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{data['day1_label']}</strong></td>
                    <td>{data['day1_volume']:,}</td>
                    <td>{data['day2_volume']:,}</td>
                    <td>{data['day3_volume']:,}</td>
                    <td class="{vol_change_class}">{data['volume_change_pct']:+.1f}%</td>
                    <td>{data['displacement']:.1f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>
        
        <h2>📸 Visualization Results</h2>
        <p>The following images show the tracking trajectories and statistical analysis:</p>
        
        <div class="image-gallery">
            <div class="image-card">
                <img src="tracking_trajectories_2d.png" alt="2D Trajectories">
                <div class="image-caption">2D Projection Trajectories (XY, XZ, YZ views)</div>
            </div>
            <div class="image-card">
                <img src="tracking_trajectories_3d.png" alt="3D Trajectories">
                <div class="image-caption">3D Trajectory Visualization</div>
            </div>
            <div class="image-card">
                <img src="tracking_analysis.png" alt="Statistical Analysis">
                <div class="image-caption">Volume Changes and Displacement Analysis</div>
            </div>
            <div class="image-card">
                <img src="tracking_report.png" alt="Detailed Report">
                <div class="image-caption">Detailed Tracking Report</div>
            </div>
        </div>
        
        <h2>📁 Generated Files</h2>
        <ul>
            <li><code>results/day1_reference.seg.nrrd</code> - Day1 reference segmentation</li>
            <li><code>results/day2_tracked_only.seg.nrrd</code> - Day2 tracked organoids only</li>
            <li><code>results/day3_tracked_only.seg.nrrd</code> - Day3 tracked organoids only</li>
            <li><code>results/trackpy_tracking_log.csv</code> - Complete tracking log</li>
        </ul>
        
        <h2>✨ How to View in 3D Slicer</h2>
        <ol>
            <li>Open 3D Slicer</li>
            <li>Drag and drop these 3 files:
                <ul>
                    <li><code>results/day1_reference.seg.nrrd</code></li>
                    <li><code>results/day2_tracked_only.seg.nrrd</code></li>
                    <li><code>results/day3_tracked_only.seg.nrrd</code></li>
                </ul>
            </li>
            <li>In the Segmentations module, expand each segmentation node</li>
            <li>Segments with the same ID (e.g., Segment_2) represent the same organoid across time points</li>
            <li>They will automatically display in the same color!</li>
        </ol>
        
        <div style="margin-top: 40px; padding: 20px; background-color: #f0f0f0; border-radius: 5px; text-align: center;">
            <p style="margin: 0; color: #666;">Generated by trackpy organoid tracking pipeline</p>
        </div>
    </div>
</body>
</html>
"""

# 保存HTML
with open('results/tracking_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"  ✓ 已保存: results/tracking_report.html")
print("\n在浏览器中打开 results/tracking_report.html 查看完整报告!")
