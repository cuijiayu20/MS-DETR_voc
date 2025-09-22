import argparse
import time
import torch
from PIL import Image

# 导入项目中的必要模块
from main import get_args_parser
from models import build_model
# --- 修复：从正确的文件导入变换函数 ---
from datasets.foggy_driving import make_foggy_driving_transforms
from util.misc import nested_tensor_from_tensor_list


def get_fps_parser():
    """为FPS计算脚本定义特定的命令行参数"""
    parser = argparse.ArgumentParser('MS-DETR FPS Calculation Script', add_help=False)
    parser.add_argument('--resume', required=True,
                        help='要测试的模型权重文件路径 (例如: exps/ms_detr_foggy_voc/checkpoint0019.pth)')
    parser.add_argument('--image_path', required=True, help='用于测试速度的单张图片路径')
    parser.add_argument('--batch_size', default=1, type=int, help='用于推理的批量大小，通常设置为1来测量单张图片的FPS')
    parser.add_argument('--num_iters', type=int, default=1, help='总共进行推理的次数，次数越多结果越稳定')
    parser.add_argument('--warm_iters', type=int, default=10, help='预热运行次数（不计入最终时间，用于稳定GPU状态）')
    return parser


@torch.no_grad()
def measure_inference_speed(model, sample_input, num_iters, warm_iters):
    """
    精确测量模型的推理速度
    """
    print(f"开始测量速度... 总共运行 {num_iters} 次, 其中前 {warm_iters} 次为预热。")

    # 预热阶段 (Warm-up)
    # 第一次运行会加载CUDA内核，速度较慢，预热可以排除这些干扰
    for _ in range(warm_iters):
        model(sample_input)

    # 正式计时
    torch.cuda.synchronize()  # 确保所有之前的GPU操作都已完成
    start_time = time.perf_counter()

    for _ in range(num_iters):
        model(sample_input)

    torch.cuda.synchronize()  # 确保所有推理操作都已完成
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iters

    return avg_time_per_iter


def main(args):
    # --- 1. 初始化模型 ---
    # 使用 MS-DETR 的主参数解析器来构建一个与训练时结构一致的模型
    main_args_parser = get_args_parser()
    # 传入与训练时相同的模型架构参数，以避免加载权重时出错
    # 这些参数应该与您训练被测模型时使用的参数一致
    model_args_list = [
        '--with_box_refine',
        '--two_stage',
        '--dim_feedforward', '2048',
        '--num_queries', '300',
        '--use_ms_detr',
        '--use_aux_ffn'
    ]
    main_args, _ = main_args_parser.parse_known_args(model_args_list)

    # 更新必要的参数
    main_args.dataset_file = 'foggy_driving'  # 这会影响类别数量的设置
    main_args.resume = args.resume

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型并加载权重
    model, _, _ = build_model(main_args)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.resume, map_location='cpu')
    # 使用我们之前修复过的逻辑来加载权重，可以忽略不匹配的分类头
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items()
                       if k in model_dict and "class_embed" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"成功从 {args.resume} 加载模型权重。")

    # --- 2. 准备输入数据 ---
    # 加载您指定的单张图片
    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {args.image_path}")
        return

    # 应用与验证时相同的图像变换
    transforms = make_foggy_driving_transforms('val')
    tensor_image, _ = transforms(image, target=None)  # 评估时不需要target

    # 组合成一个batch
    input_list = [tensor_image for _ in range(args.batch_size)]
    sample_input = nested_tensor_from_tensor_list(input_list).to(device)

    # --- 3. 计算并打印FPS ---
    avg_time = measure_inference_speed(model, sample_input, args.num_iters, args.warm_iters)

    fps = args.batch_size / avg_time

    print("\n--- FPS 计算结果 ---")
    print(f"模型: {args.resume}")
    print(f"测试图片: {args.image_path}")
    print(f"批量大小 (Batch Size): {args.batch_size}")
    print(f"平均每批次推理时间: {avg_time * 1000:.2f} ms")
    print(f"FPS (每秒帧数): {fps:.2f}")
    print("--------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MS-DETR FPS Calculation', parents=[get_fps_parser()])
    args = parser.parse_args()
    main(args)

