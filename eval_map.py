# eval_map.py

import argparse
import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

from main import get_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list
from util.utils_map import get_map  # 导入我们刚刚复制的模块


def get_eval_args_parser():
    parser = argparse.ArgumentParser('MS-DETR evaluation with custom mAP script', add_help=False)
    parser.add_argument('--resume', required=True, help='Path to the checkpoint file to evaluate.')
    parser.add_argument('--test_dataset', required=True, type=str, choices=['Foggy_Driving_voc', 'RTTStest'],
                        help='Name of the test dataset to use.')
    parser.add_argument('--min_overlap', default=0.5, type=float, help='IoU threshold for mAP calculation.')
    parser.add_argument('--map_out_path', default='map_out',
                        help='Path to save detection results and ground truth files.')
    return parser


# MS-DETR 的类别顺序
VOC_CLASSES = (
    "person", "bicycle", "car", "motorbike", "bus"
)


def generate_prediction_files(model, dataset, device, map_out_path, postprocessors):
    print(f"Generating prediction files for dataset: {dataset.img_folder.parent.name}...")
    det_results_path = os.path.join(map_out_path, 'detection-results')
    if not os.path.exists(det_results_path):
        os.makedirs(det_results_path)

    for i in tqdm(range(len(dataset))):
        sample, target = dataset[i]
        image_id = dataset.image_ids[i]

        samples = nested_tensor_from_tensor_list([sample]).to(device)

        with torch.no_grad():
            outputs = model(samples)

        orig_target_sizes = torch.stack([target["orig_size"]], dim=0).to(device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        scores = results[0]['scores']
        labels = results[0]['labels']
        boxes = results[0]['boxes']

        with open(os.path.join(det_results_path, image_id + ".txt"), "w") as f:
            for score, label, box in zip(scores, labels, boxes):
                class_name = VOC_CLASSES[label]
                left, top, right, bottom = box.cpu().numpy()
                f.write(f"{class_name} {score.item()} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")


def generate_ground_truth_files(dataset, map_out_path):
    print(f"Generating ground truth files for dataset: {dataset.img_folder.parent.name}...")
    gt_path = os.path.join(map_out_path, 'ground-truth')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    for image_id in tqdm(dataset.image_ids):
        xml_path = os.path.join(dataset.ann_folder, image_id + ".xml")

        with open(os.path.join(gt_path, image_id + ".txt"), "w") as new_f:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                if obj_name not in VOC_CLASSES:
                    continue

                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                else:
                    new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")


def main(args):
    # --- 1. 初始化模型和数据 ---
    main_args_parser = get_args_parser()
    main_args, _ = main_args_parser.parse_known_args()

    # 将 eval_map.py 的参数传递给 build_dataset 函数
    main_args.dataset_file = 'foggy_driving'
    main_args.test_dataset = args.test_dataset  # <--- 关键修改

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, postprocessors = build_model(main_args)
    model.to(device)
    model.eval()

    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    # 使用 'test' image_set 来构建测试数据集
    dataset_test = build_dataset(image_set='test', args=main_args)

    # --- 2. 准备评估文件 ---
    map_out_path = f"{args.map_out_path}_{args.test_dataset}"
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    os.makedirs(map_out_path)

    generate_prediction_files(model, dataset_test, device, map_out_path, postprocessors)
    generate_ground_truth_files(dataset_test, map_out_path)

    # --- 3. 计算 mAP ---
    print(f"Calculating mAP for {args.test_dataset}...")
    get_map(args.min_overlap, draw_plot=True, path=map_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MS-DETR custom evaluation script', parents=[get_eval_args_parser()])
    args = parser.parse_args()
    main(args)
