import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 경로 설정
train_root = os.path.abspath("SDLane/train")
images_dir = os.path.join(train_root, "images")
labels_dir = os.path.join(train_root, "labels")
resized_images_dir = os.path.join(train_root, "resized_images")
resized_masks_dir = os.path.join(train_root, "resized_masks")
train_list_path = os.path.join(train_root, "train_list.txt")

# 디렉토리 생성
os.makedirs(resized_images_dir, exist_ok=True)
os.makedirs(resized_masks_dir, exist_ok=True)

# 이미지 리스트 로드
with open(train_list_path, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines):
    image_path = line.strip()  # ex) images/UUID/0001.jpg
    full_img_path = os.path.join(train_root, image_path)

    # 이미지 로드
    img = cv2.imread(full_img_path)
    if img is None:
        print(f"이미지 로드 실패: {full_img_path}")
        continue

    # 원본 크기 저장 후 리사이즈
    orig_h, orig_w = img.shape[:2]
    resized_img = cv2.resize(img, (512, 256))

    # 저장 이름 처리: images/UUID/0001.jpg → UUID_0001.jpg
    base_name = image_path.replace("images/", "").replace("/", "_")
    resized_img_path = os.path.join(resized_images_dir, base_name)
    cv2.imwrite(resized_img_path, resized_img)

    # 마스크 초기화 (512x256)
    mask = np.zeros((256, 512), dtype=np.uint8)

    # 라벨 경로 계산
    relative_label_path = image_path.replace("images/", "").replace(".jpg", ".json")
    label_path = os.path.join(labels_dir, relative_label_path)

    if not os.path.exists(label_path):
        print(f"라벨 파일 없음: {label_path}")
        continue

    # JSON 로드
    with open(label_path, 'r') as lf:
        label_data = json.load(lf)

    for idx, line in enumerate(label_data.get("geometry", [])):
        points = np.array(line, dtype=np.float32)

        # 좌표 리사이즈 반영
        points[:, 0] = points[:, 0] * (512 / orig_w)
        points[:, 1] = points[:, 1] * (256 / orig_h)
        points = points.astype(np.int32)

        mean_x = np.mean(points[:, 0])
        color = 1 if mean_x < 512 // 2 else 2
        cv2.polylines(mask, [points], isClosed=False, color=color, thickness=5)

    # 마스크 저장
    mask_name = base_name.replace(".jpg", "_mask.png")
    mask_path = os.path.join(resized_masks_dir, mask_name)
    cv2.imwrite(mask_path, mask)
