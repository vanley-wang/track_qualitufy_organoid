"""
FXN 数据配准脚本
使用 itk-elastix 对 Day5 原始图像进行刚体配准到 Day3，
然后将同一变换应用到 Day5 分割 mask 上。

输出：registered_seg/ 目录，包含配准后的 Day5 分割文件
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import itk
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg


def load_nifti_itk(file_path):
    """使用 itk 加载 NIfTI 文件"""
    image = itk.imread(file_path)
    return image


def save_nifti_itk(image, file_path):
    """使用 itk 保存 NIfTI 文件"""
    itk.imwrite(image, file_path)


def register_images(fixed_path, moving_path, transform_type='rigid'):
    """
    使用 Elastix 配准 moving 图像到 fixed 图像

    参数:
        fixed_path: 固定图像路径 (Day3 原始图像)
        moving_path: 浮动图像路径 (Day5 原始图像)
        transform_type: 'rigid' 或 'affine'

    返回:
        result_image: 配准后的 moving 图像
        transform_parameters: 变换参数对象
    """
    fixed_image = load_nifti_itk(fixed_path)
    moving_image = load_nifti_itk(moving_path)

    # Elastix 参数对象
    parameter_object = itk.ParameterObject.New()

    # 使用预设参数文件
    if transform_type == 'rigid':
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(default_rigid_parameter_map)
    elif transform_type == 'affine':
        default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
        parameter_object.AddParameterMap(default_affine_parameter_map)
    else:
        raise ValueError("transform_type must be 'rigid' or 'affine'")

    # 修改参数以适应我们的数据
    parameter_object.SetParameter("MaximumNumberOfIterations", "1000")
    parameter_object.SetParameter("NumberOfResolutions", "3")
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("AutomaticTransformInitialization", "true")

    # 运行 Elastix
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False
    )

    return result_image, result_transform_parameters


def transform_mask(fixed_path, moving_mask_path, transform_parameters, output_path):
    """
    将配准变换应用到分割 mask 上
    使用最近邻插值以保持整数标签

    参数:
        fixed_path: 固定图像路径（用于获取参考空间）
        moving_mask_path: Day5 分割 mask 路径
        transform_parameters: Elastix 变换参数
        output_path: 输出路径
    """
    fixed_image = load_nifti_itk(fixed_path)
    moving_mask = load_nifti_itk(moving_mask_path)

    # 创建变换参数对象用于 Transformix
    transformix_parameter_object = itk.ParameterObject.New()
    transformix_parameter_object.AddParameterMap(transform_parameters.GetParameterMap(0))

    # 设置最近邻插值（对分割 mask 至关重要）
    transformix_parameter_object.SetParameter("ResampleInterpolator", "FinalNearestNeighborInterpolator")
    transformix_parameter_object.SetParameter("WriteResultImage", "true")

    # 运行 Transformix
    result_mask = itk.transformix_filter(
        moving_mask,
        transformix_parameter_object
    )

    # 保存结果
    save_nifti_itk(result_mask, output_path)


def process_well_registration(well_name, output_dir, transform_type='rigid'):
    """
    处理单个 well 的配准

    参数:
        well_name: e.g. 'B2_1'
        output_dir: 输出目录
        transform_type: 'rigid' 或 'affine'
    """
    print(f"\n{'='*60}")
    print(f"配准 Well: {well_name} ({transform_type})")
    print(f"{'='*60}")

    # 文件路径
    day3_raw = os.path.join(cfg.FXN_0701_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_raw = os.path.join(cfg.FXN_0703_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_seg = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

    # 检查文件
    for f, name in [(day3_raw, 'Day3 raw'), (day5_raw, 'Day5 raw'), (day5_seg, 'Day5 seg')]:
        if not os.path.exists(f):
            print(f"  ❌ 错误: 未找到 {name}: {f}")
            return False

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{well_name}_day5_registered.nii.gz')

    # 如果已存在，跳过
    if os.path.exists(output_path):
        print(f"  ⚠️  输出已存在，跳过: {output_path}")
        return True

    try:
        # Step 1: 配准原始图像
        print(f"  [1/2] 配准 Day5 raw → Day3 raw...")
        _, transform_parameters = register_images(day3_raw, day5_raw, transform_type)
        print(f"        ✓ 配准完成")

        # Step 2: 应用变换到分割 mask
        print(f"  [2/2] 变换 Day5 seg → Day3 空间...")
        transform_mask(day3_raw, day5_seg, transform_parameters, output_path)
        print(f"        ✓ 已保存: {output_path}")

        return True

    except Exception as e:
        print(f"  ❌ 配准失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("FXN 数据配准: Day5 → Day3")
    print("使用 Elastix 刚体/仿射配准")
    print("=" * 70)

    # 输出目录
    registered_dir = os.path.join(cfg.OUTPUT_DIR, 'registered_seg')
    os.makedirs(registered_dir, exist_ok=True)

    # 获取 well 列表
    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 well 需要配准")
    print(f"输出目录: {registered_dir}")

    success_count = 0
    fail_count = 0

    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        success = process_well_registration(well_name, registered_dir, transform_type='rigid')
        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("配准完成!")
    print(f"成功: {success_count} / {len(day3_files)}")
    print(f"失败: {fail_count}")
    print(f"输出目录: {registered_dir}")
    print("=" * 70)
    print("\n下一步: 运行 fxn_track_and_visualize_registered.py 进行追踪")


if __name__ == '__main__':
    main()
