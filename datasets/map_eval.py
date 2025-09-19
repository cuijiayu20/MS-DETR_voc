# datasets/map_eval.py

import os
import torch
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

from util.misc import all_gather
from util.utils_map import get_map

# 确保 MS-DETR 的类别顺序与您的数据一致
VOC_CLASSES = (
    "person", "bicycle", "car", "motorbike", "bus"
)


class MapEvaluator:
    def __init__(self, dataset, output_dir, epoch):
        self.dataset = dataset
        # 为每个 epoch 创建一个独立的评估文件夹
        self.map_out_path = os.path.join(output_dir, f'map_out_epoch_{epoch}')
        self.predictions = []

    def update(self, predictions):
        # predictions 是一个字典 {image_id: {'scores': tensor, 'labels': tensor, 'boxes': tensor}}
        self.predictions.extend(predictions.items())

    def synchronize_between_processes(self):
        all_preds = all_gather(self.predictions)
        merged_preds = []
        for p in all_preds:
            merged_preds.extend(p)
        self.predictions = merged_preds

    def accumulate(self):
        # 准备文件目录
        self.det_results_path = os.path.join(self.map_out_path, 'detection-results')
        self.gt_path = os.path.join(self.map_out_path, 'ground-truth')

        if os.path.exists(self.map_out_path):
            shutil.rmtree(self.map_out_path)
        os.makedirs(self.det_results_path)
        os.makedirs(self.gt_path)

        # 1. 生成预测文件
        print("Generating prediction files for mAP...")
        # self.predictions 是一个列表，元素为 (image_id_int, prediction_dict)
        for image_id_int, prediction in tqdm(self.predictions):
            # 从 dataset.image_ids 中通过整数索引获取文件名
            image_filename_no_ext = self.dataset.image_ids[image_id_int]

            with open(os.path.join(self.det_results_path, image_filename_no_ext + ".txt"), "w") as f:
                for score, label, box in zip(prediction['scores'], prediction['labels'], prediction['boxes']):
                    class_name = VOC_CLASSES[label]
                    left, top, right, bottom = box.cpu().numpy()
                    f.write(f"{class_name} {score.item()} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")

        # 2. 生成 GT 文件
        print("Generating ground truth files for mAP...")
        ann_folder = self.dataset.ann_folder  # 从数据集中获取标注文件夹路径
        for image_id in tqdm(self.dataset.image_ids):
            xml_path = os.path.join(ann_folder, image_id + ".xml")
            if not os.path.exists(xml_path):
                # 打印警告，以便调试
                print(f"Warning: Annotation file not found for image {image_id}, skipping GT generation.")
                continue

            with open(os.path.join(self.gt_path, image_id + ".txt"), "w") as new_f:
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

    def summarize(self):
        print(f"Calculating mAP for epoch, results in: {self.map_out_path}")
        # 注意: get_map 会将结果打印到控制台，并返回一个字典
        results = get_map(min_overlap=0.5, draw_plot=True, path=self.map_out_path)

        # 我们只关心 mAP
        mAP = results['mAP']
        # 返回一个和 CocoEvaluator 兼容的字典结构
        stats = {'mAP': mAP}

        # 在主进程中打印 mAP
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"mAP@0.5: {mAP}")

        return stats