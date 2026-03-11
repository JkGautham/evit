import os
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torch
from torchvision.transforms import functional as F

class COCO10(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transforms = transforms

        self.cat_id_map = {}
        self.cat_ids = []

        all_categories = self.coco.loadCats(self.coco.getCatIds())
        for idx, cat in enumerate(all_categories):
            self.cat_id_map[cat['id']] = idx + 1
            self.cat_ids.append(cat['id'])

        # Filter valid image IDs: with annotations + image file exists
        self.img_ids = []
        for img_id in self.coco.imgs.keys():
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
            if os.path.exists(img_path) and len(ann_ids) > 0:
                self.img_ids.append(img_id)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        try:
            image = Image.open(img_path)
            image.load()  # Explicitly load image to catch truncation early
            image = image.convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted image: {img_path} due to {e}")
            return self.__getitem__((index + 1) % len(self))  # Try next image

        seg_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_label = self.cat_id_map.get(cat_id, 0)
            mask = self.coco.annToMask(ann)
            seg_mask = np.where(mask == 1, cat_label, seg_mask)

        image = F.resize(image, (512, 512), interpolation=Image.BILINEAR)
        seg_mask = F.resize(Image.fromarray(seg_mask), (512, 512), interpolation=Image.NEAREST)

        image = F.to_tensor(image)
        seg_mask = torch.from_numpy(np.array(seg_mask)).long()

        return image, seg_mask



    def __len__(self):
        return len(self.img_ids)
