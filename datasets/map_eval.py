import os
import torch
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

from util.misc import all_gather
from util.utils_map import get_map  # 导入我们刚刚复制的模块

# 确保这里的类别与您 datasets/foggy_driving.py 中的 VOC_CLASSES 完全一致
VOC_CLASSES = (
    "person", "bicycle", "car", "motorbike", "bus"
)


class MapEvaluator:
    def __init__(self, dataset, output_dir, epoch):
        self.dataset = dataset
        # 为每个 epoch 创建一个独立的评估文件夹，方便保存
        self.map_out_path = os.path.join(output_dir, f'map_out_epoch_{epoch}')
        self.predictions = []

    def update(self, predictions):
        # predictions 的格式是 {coco_image_id: result_dict}
        # result_dict 包含 'scores', 'labels', 'boxes'
        self.predictions.extend(predictions.items())

    def synchronize_between_processes(self):
        # 在分布式训练中，从所有GPU收集预测结果
        all_preds = all_gather(self.predictions)
        merged_preds = []
        for p in all_preds:
            merged_preds.extend(p)
        self.predictions = merged_preds

    def accumulate(self):
        # 这个方法负责将模型输出转换为 get_map.py 需要的 .txt 文件格式

        # 1. 准备文件目录
        self.det_results_path = os.path.join(self.map_out_path, 'detection-results')
        self.gt_path = os.path.join(self.map_out_path, 'ground-truth')

        if os.path.exists(self.map_out_path):
            shutil.rmtree(self.map_out_path)
        os.makedirs(self.det_results_path)
        os.makedirs(self.gt_path)

        # 2. 生成预测文件 (detection-results)
        print("Generating prediction files for mAP...")
        for image_id_int, prediction in tqdm(self.predictions):
            # 使用整数索引从数据集中获取不带后缀的文件名
            image_filename_no_ext = self.dataset.image_ids[image_id_int]

            with open(os.path.join(self.det_results_path, image_filename_no_ext + ".txt"), "w") as f:
                for score, label, box in zip(prediction['scores'], prediction['labels'], prediction['boxes']):
                    class_name = VOC_CLASSES[label]
                    left, top, right, bottom = box.cpu().numpy()
                    f.write(f"{class_name} {score.item()} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")

        # 3. 生成真实标签文件 (ground-truth)
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
        # --- 修复 ---
        # 根据您提供的函数定义 def get_map(MINOVERLAP, draw_plot, path) 进行调用
        results = get_map(0.5, True, path=self.map_out_path)
        # --- 修复结束 ---

        # 我们只关心 mAP
        mAP = results['mAP']
        # 返回一个和 CocoEvaluator 兼容的字典结构
        stats = {'mAP': mAP}

        # 在主进程中打印 mAP
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"mAP@0.5 IoU: {mAP}")

        return stats

