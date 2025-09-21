import os
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path

from util.misc import get_local_rank, get_local_size
import datasets.transforms as T

# 类别名称 -> 类别ID 的映射 (固定为您的5个类别)
VOC_CLASSES = (
    "person", "bicycle", "car", "motorbike", "bus"
)


def get_class_map():
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    return class_to_ind


class FoggyDrivingDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, image_set_file, transforms, return_masks):
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self._transforms = transforms
        self.return_masks = return_masks
        self.class_to_ind = get_class_map()

        with open(image_set_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.ids = list(range(len(self.image_ids)))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        ann_path = self.ann_folder / f"{image_id}.xml"
        target = self.parse_voc_xml(ET.parse(ann_path).getroot())

        # --- 修复：动态检查图片后缀 ---
        img_path_jpg = self.img_folder / f"{image_id}.jpg"
        img_path_png = self.img_folder / f"{image_id}.png"

        if img_path_jpg.exists():
            img_path = img_path_jpg
        elif img_path_png.exists():
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image not found for {image_id} with .jpg or .png extension in {self.img_folder}")

        img = Image.open(img_path).convert('RGB')
        # --- 修复结束 ---

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

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) - 1
            ymin = float(bndbox.find('ymin').text) - 1
            xmax = float(bndbox.find('xmax').text) - 1
            ymax = float(bndbox.find('ymax').text) - 1
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
    # --- 修复：调整路径以匹配 "data" 子文件夹 ---
    if image_set == 'train':
        root_dir = Path('./data/voc-fog/train')
        dataset_name = args.train_dataset
        image_set_file = root_dir / 'ImageSets' / 'Main' / 'train.txt'
        ann_folder = root_dir / 'Annotations'
        img_folder = root_dir / dataset_name
    elif image_set == 'val':
        dataset_name = args.val_dataset

        if dataset_name in ['VOCtest-FOG', 'RainyImages', 'SnowyImages']:
            root_dir = Path('./data/voc-fog/test')
            image_set_file = root_dir / 'ImageSets' / 'Main' / 'val.txt'
            ann_folder = root_dir / 'Annotations'
            img_folder = root_dir / dataset_name
        elif dataset_name == 'Foggy_Driving_voc':
            root_dir = Path('./data/Foggy_Driving_voc')
            image_set_file = root_dir / 'ImageSets' / 'Main' / 'val.txt'
            ann_folder = root_dir / 'Annotations'
            img_folder = root_dir / 'JPEGImages'
        elif dataset_name == 'RTTStest':
            root_dir = Path('./data/RTTStest')
            image_set_file = root_dir / 'ImageSets' / 'Main' / 'val.txt'
            ann_folder = root_dir / 'Annotations'
            img_folder = root_dir / 'Images'
        else:
            raise ValueError(f'unknown val_dataset {dataset_name}')
    else:
        raise ValueError(f'unknown image_set {image_set}')
    # --- 修复结束 ---

    dataset = FoggyDrivingDetection(str(img_folder), str(ann_folder), str(image_set_file),
                                    transforms=make_foggy_driving_transforms(image_set),
                                    return_masks=args.masks)
    return dataset
