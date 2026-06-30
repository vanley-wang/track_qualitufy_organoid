"""
FXN 仿射配准脚本 (Affine Registration)
将 Day5 配准到 Day3，使用仿射变换补偿整体形变

运行方式:
    python fxn_register_affine.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import itk
import warnings
warnings.filterwarnings('ignore')

import fxn_config as cfg


def load_nifti_itk(file_path):
    return itk.imread(file_path)


def save_nifti_itk(image, file_path):
    itk.imwrite(image, file_path)


def register_affine(fixed_image, moving_image):
    """仿射配准"""
    parameter_object = itk.ParameterObject.New()
    default_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_affine)
    parameter_object.SetParameter("MaximumNumberOfIterations", "1000")
    parameter_object.SetParameter("WriteResultImage", "false")

    result_image, transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False
    )
    return result_image, transform_parameters


def transform_mask(fixed_path, moving_mask_path, transform_parameters, output_path):
    """将变换应用到分割 mask（最近邻插值）"""
    fixed_image = load_nifti_itk(fixed_path)
    moving_mask = load_nifti_itk(moving_mask_path)

    transformix_parameter_object = itk.ParameterObject.New()
    transformix_parameter_object.AddParameterMap(transform_parameters.GetParameterMap(0))
    transformix_parameter_object.SetParameter("ResampleInterpolator", "FinalNearestNeighborInterpolator")
    transformix_parameter_object.SetParameter("WriteResultImage", "true")

    result_mask = itk.transformix_filter(
        moving_mask,
        transformix_parameter_object
    )

    save_nifti_itk(result_mask, output_path)


def process_well_affine(well_name, output_dir):
    """处理单个 well 的仿射配准"""
    print(f"\n{'='*60}")
    print(f"Affine 配准 Well: {well_name}")
    print(f"{'='*60}")

    day3_raw = os.path.join(cfg.FXN_0701_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_raw = os.path.join(cfg.FXN_0703_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_seg = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

    for f, name in [(day3_raw, 'Day3 raw'), (day5_raw, 'Day5 raw'), (day5_seg, 'Day5 seg')]:
        if not os.path.exists(f):
            print(f"  ❌ 错误: 未找到 {name}: {f}")
            return False

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{well_name}_day5_affine.nii.gz')

    if os.path.exists(output_path):
        print(f"  ⚠️  输出已存在，跳过")
        return True

    try:
        print(f"  [1/2] 配准 Day5 raw → Day3 raw (affine)...")
        fixed_image = load_nifti_itk(day3_raw)
        moving_image = load_nifti_itk(day5_raw)
        _, transform_parameters = register_affine(fixed_image, moving_image)
        print(f"        ✓ 配准完成")

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
    print("FXN 仿射配准")
    print("Day5 raw → Day3 raw (Affine)")
    print("=" * 70)

    affine_dir = os.path.join(cfg.OUTPUT_DIR, 'affine_seg')
    os.makedirs(affine_dir, exist_ok=True)

    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 well")
    print(f"输出目录: {affine_dir}")

    success = 0
    fail = 0
    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        if process_well_affine(well_name, affine_dir):
            success += 1
        else:
            fail += 1

    print("\n" + "=" * 70)
    print("仿射配准完成!")
    print(f"成功: {success} / {len(day3_files)}")
    print(f"失败: {fail}")
    print(f"输出目录: {affine_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
