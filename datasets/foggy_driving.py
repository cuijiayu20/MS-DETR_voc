# datasets/foggy_driving.py

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from pathlib import Path

# 类别名称 -> 类别ID 的映射
# !!! 固定为您的5个类别 !!!
VOC_CLASSES = (
    "person", "bicycle", "car", "motorbike", "bus"
)


def get_class_map():
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    return class_to_ind


class FoggyDrivingDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, image_set_file, transforms, return_masks):
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)  # <--- 确保这行存在
        self._transforms = transforms
        self.return_masks = return_masks
        self.class_to_ind = get_class_map()

        with open(image_set_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]  # <--- 确保这行存在

        self.ids = list(range(len(self.image_ids)))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        ann_path = self.ann_folder / f"{image_id}.xml"
        target = self.parse_voc_xml(ET.parse(ann_path).getroot())

        img_path = self.img_folder / f"{image_id}.jpg"
        img = Image.open(img_path).convert('RGB')

        target['image_id'] = torch.tensor([idx])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, node):
        voc_dict = {}
        size = node.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        objs = node.findall('object')
        boxes = []
        gt_classes = []
        for obj in objs:
            obj_name = obj.find('name').text
            if obj_name not in self.class_to_ind:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) - 1
            ymin = float(bbox.find('ymin').text) - 1
            xmax = float(bbox.find('xmax').text) - 1
            ymax = float(bbox.find('ymax').text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            gt_classes.append(self.class_to_ind[obj_name])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = torch.tensor(gt_classes, dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                    target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)

        return target


def make_foggy_driving_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    if image_set == 'train':
        root_dir = Path('./voc-fog/train')
        dataset_name = args.train_dataset
        image_set_file = root_dir / 'ImageSets' / 'Main' / 'train.txt'
    elif image_set == 'val':
        root_dir = Path('./voc-fog/test')
        dataset_name = args.val_dataset
        image_set_file = root_dir / 'ImageSets' / 'Main' / 'val.txt'
    else:
        # Handling independent test sets
        if args.test_dataset == 'Foggy_Driving_voc':
            img_folder = './Foggy_Driving_voc/JPEGImages'
            ann_folder = './Foggy_Driving_voc/Annotations'
            image_set_file = './Foggy_Driving_voc/ImageSets/Main/test.txt'  # Assume you have a test.txt
        elif args.test_dataset == 'RTTStest':
            img_folder = './RTTStest/Images'
            ann_folder = './RTTStest/Annotations'
            image_set_file = './RTTStest/ImageSets/Main/test.txt'  # Assume you have a test.txt
        else:
            raise ValueError(f'unknown test_dataset {args.test_dataset}')

        dataset = FoggyDrivingDetection(str(img_folder), str(ann_folder), str(image_set_file),
                                        transforms=make_foggy_driving_transforms('test'),
                                        return_masks=args.masks)
        return dataset

    # Construct paths for train/val sets
    img_folder = root_dir / dataset_name
    ann_folder = root_dir / 'Annotations'

    dataset = FoggyDrivingDetection(str(img_folder), str(ann_folder), str(image_set_file),
                                    transforms=make_foggy_driving_transforms(image_set),
                                    return_masks=args.masks)
    return dataset