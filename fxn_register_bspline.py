"""
FXN B-spline 非刚性配准脚本
步骤: 刚体预对齐 → B-spline 精调 → 变换应用到 mask

运行方式:
    python fxn_register_bspline.py
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
    """使用 itk 加载 NIfTI"""
    return itk.imread(file_path)


def save_nifti_itk(image, file_path):
    """使用 itk 保存 NIfTI"""
    itk.imwrite(image, file_path)


def register_rigid(fixed_image, moving_image):
    """刚体预对齐"""
    parameter_object = itk.ParameterObject.New()
    default_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid)
    parameter_object.SetParameter("MaximumNumberOfIterations", "500")
    parameter_object.SetParameter("NumberOfResolutions", "3")
    parameter_object.SetParameter("WriteResultImage", "false")

    result_image, transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False
    )
    return result_image, transform_parameters


def register_bspline(fixed_image, moving_image):
    """
    B-spline 非刚性配准
    输入 moving_image 应该已经经过刚体预对齐
    """
    parameter_object = itk.ParameterObject.New()

    # 先添加刚体参数作为初始化
    default_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid)

    # 再添加 B-spline 参数
    default_bspline = parameter_object.GetDefaultParameterMap('bspline')
    parameter_object.AddParameterMap(default_bspline)

    # 设置 B-spline 参数
    parameter_object.SetParameter("MaximumNumberOfIterations", str(cfg.BSPLINE_MAX_ITERATIONS))
    parameter_object.SetParameter("NumberOfResolutions", str(cfg.BSPLINE_NUM_RESOLUTIONS))
    parameter_object.SetParameter("FinalGridSpacingInVoxels", str(cfg.BSPLINE_GRID_SPACING))
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("Metric", "AdvancedMattesMutualInformation")
    parameter_object.SetParameter("Transform", "BSplineTransform")

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
    # 使用最后一组参数（B-spline 的）
    transformix_parameter_object.AddParameterMap(transform_parameters.GetParameterMap(transform_parameters.GetNumberOfParameterMaps() - 1))
    transformix_parameter_object.SetParameter("ResampleInterpolator", "FinalNearestNeighborInterpolator")
    transformix_parameter_object.SetParameter("WriteResultImage", "true")

    result_mask = itk.transformix_filter(
        moving_mask,
        transformix_parameter_object
    )

    save_nifti_itk(result_mask, output_path)


def process_well_bspline(well_name, output_dir):
    """处理单个 well: 刚体预对齐 + B-spline 精调 + mask 变换"""
    print(f"\n{'='*60}")
    print(f"B-spline 配准 Well: {well_name}")
    print(f"{'='*60}")

    day3_raw = os.path.join(cfg.FXN_0701_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_raw = os.path.join(cfg.FXN_0703_RAW_DIR, f'{well_name}_0000.nii.gz')
    day5_seg = os.path.join(cfg.FXN_0703_SEG_DIR, f'{well_name}.nii.gz')

    for f, name in [(day3_raw, 'Day3 raw'), (day5_raw, 'Day5 raw'), (day5_seg, 'Day5 seg')]:
        if not os.path.exists(f):
            print(f"  ❌ 错误: 未找到 {name}: {f}")
            return False

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{well_name}_day5_bspline.nii.gz')

    if os.path.exists(output_path):
        print(f"  ⚠️  输出已存在，跳过")
        return True

    try:
        print(f"  [1/3] 加载图像...")
        fixed_image = load_nifti_itk(day3_raw)
        moving_image = load_nifti_itk(day5_raw)

        print(f"  [2/3] 刚体预对齐...")
        rigid_aligned, rigid_params = register_rigid(fixed_image, moving_image)
        print(f"        ✓ 刚体预对齐完成")

        print(f"  [3/3] B-spline 精调 + mask 变换...")
        _, bspline_params = register_bspline(fixed_image, rigid_aligned)
        print(f"        ✓ B-spline 配准完成")

        # 将最终变换应用到 mask
        transform_mask(day3_raw, day5_seg, bspline_params, output_path)
        print(f"        ✓ 已保存: {output_path}")
        return True

    except Exception as e:
        print(f"  ❌ 配准失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("FXN B-spline 非刚性配准")
    print("Day5 raw → Day3 raw (Rigid + B-spline)")
    print("=" * 70)

    bspline_dir = os.path.join(cfg.OUTPUT_DIR, 'bspline_seg')
    os.makedirs(bspline_dir, exist_ok=True)

    day3_files = sorted([f for f in os.listdir(cfg.FXN_0701_SEG_DIR) if f.endswith('.nii.gz')])
    print(f"\n发现 {len(day3_files)} 个 well")
    print(f"输出目录: {bspline_dir}")

    success = 0
    fail = 0
    for seg_file in day3_files:
        well_name = seg_file.replace('.nii.gz', '')
        if process_well_bspline(well_name, bspline_dir):
            success += 1
        else:
            fail += 1

    print("\n" + "=" * 70)
    print("B-spline 配准完成!")
    print(f"成功: {success} / {len(day3_files)}")
    print(f"失败: {fail}")
    print(f"输出目录: {bspline_dir}")
    print("=" * 70)
    print("\n下一步: 运行 fxn_track_multifeature.py 进行多特征追踪")


if __name__ == '__main__':
    main()
