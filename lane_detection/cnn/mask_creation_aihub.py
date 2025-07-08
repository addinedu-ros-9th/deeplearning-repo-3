import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# [원천]c_1280_720_daylight_train_1-002.tar
# [원천]c_1280_720_daylight_train_8-001.tar

# [원천]c_1280_720_daylight_train_5-001.tar
# [원천]c_1280_720_daylight_train_6-002.tar

# 경로 설정
train_root = os.path.abspath("ai_hub_dataset/train")
images_dir = os.path.join(train_root, "images")
labels_dir = os.path.join(train_root, "labels")
resized_images_dir = os.path.join(train_root, "resized_images")
resized_masks_dir = os.path.join(train_root, "resized_masks")
train_list_path = os.path.join(train_root, "train_list.txt")

# 디렉토리 생성
os.makedirs(resized_images_dir, exist_ok=True)
os.makedirs(resized_masks_dir, exist_ok=True)

# train_list.txt 자동 생성
if not os.path.exists(train_list_path):
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    with open(train_list_path, 'w') as f:
        for name in all_images:
            f.write(f"{name}\n")
    print(f"[생성 완료] train_list.txt ({len(all_images)}개 항목)")

# 클래스 인덱스 결정 함수
def get_class_index(ann_class, attributes):
    if ann_class == "traffic_lane":
        color = attributes.get("lane_color", "")
        ltype = attributes.get("lane_type", "")
        if color == "white":
            return 1 if ltype == "solid" else 2  # 흰색 실선 vs 점선
        elif color == "yellow" and ltype == "solid":
            return 3  # 중앙선 (노란 실선)
    elif ann_class == "stop_line":
        return 4
    elif ann_class == "crosswalk":
        return 5
    return 0  # 배경


# 이미지 리스트 로드
with open(train_list_path, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines):
    image_file = line.strip()  # 예: 0001.jpg
    img_path = os.path.join(images_dir, image_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[이미지 로드 실패] {img_path}")
        continue

    orig_h, orig_w = img.shape[:2]
    resized_img = cv2.resize(img, (512, 256))
    resized_img_path = os.path.join(resized_images_dir, image_file)
    cv2.imwrite(resized_img_path, resized_img)

    # 마스크 초기화
    mask = np.zeros((256, 512), dtype=np.uint8)

    json_file = image_file.replace(".jpg", ".json")
    label_path = os.path.join(labels_dir, json_file)
    if not os.path.exists(label_path):
        print(f"[라벨 없음] {label_path}")
        continue

    with open(label_path, 'r') as lf:
        label_data = json.load(lf)

    for ann in label_data.get("annotations", []):
        cls = ann["class"]
        attr = {a["code"]: a["value"] for a in ann.get("attributes", [])}
        category = ann.get("category")
        points = np.array([[pt["x"], pt["y"]] for pt in ann["data"]], dtype=np.float32)

        # 좌표 리사이즈
        points[:, 0] *= (512 / orig_w)
        points[:, 1] *= (256 / orig_h)
        points = points.astype(np.int32)

        # 클래스 인덱스 추출
        label_id = get_class_index(cls, attr)

        if category == "polyline":
            cv2.polylines(mask, [points], isClosed=False, color=label_id, thickness=5)
        elif category == "polygon":
            cv2.fillPoly(mask, [points], color=label_id)

    # 마스크 저장
    mask_file = image_file.replace(".jpg", "_mask.png")
    mask_path = os.path.join(resized_masks_dir, mask_file)
    cv2.imwrite(mask_path, mask)

print("[All done!]")