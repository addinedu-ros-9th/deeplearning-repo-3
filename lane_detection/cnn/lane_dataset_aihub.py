import os
import cv2
import torch
from torch.utils.data import Dataset

class LaneDataset(Dataset):
    def __init__(self, list_path, resized_image_dir, resized_mask_dir):
        with open(list_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        self.image_dir = resized_image_dir
        self.mask_dir = resized_mask_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_file = self.image_list[idx]
        img_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, image_file.replace(".jpg", "_mask.png"))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"[이미지 없음] {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.FloatTensor(img)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"[마스크 없음] {mask_path}")
        mask = torch.LongTensor(mask)

        return img, mask
