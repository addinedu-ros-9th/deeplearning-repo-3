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
        # 원본 경로 예) images/UUID/0001.jpg
        orig_path = self.image_list[idx]
        
        # resized_images에 맞는 파일명으로 변환: images/UUID/0001.jpg → UUID_0001.jpg
        base_name = orig_path.replace("images/", "").replace("/", "_")
        img_path = os.path.join(self.image_dir, base_name)
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")
        
        # 리사이즈된 상태
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0 # HWC to CHW and normalized
        img = torch.FloatTensor(img)

        # 마스크 경로: UUID_0001_mask.png
        mask_name = base_name.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"마스크를 불러올 수 없습니다: {mask_path}")

        # 마스크도 512x256 크기로 저장되어 있음
        mask = torch.LongTensor(mask)

        return img, mask
