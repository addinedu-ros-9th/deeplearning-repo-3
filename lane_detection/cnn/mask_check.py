import cv2
import numpy as np
import os

# 파일 경로 설정
img_file = "6938798.jpg"  # 확인할 이미지 파일 이름
resized_img_path = os.path.join("ai_hub_dataset/train/resized_images", img_file)
mask_file = img_file.replace(".jpg", "_mask.png")
resized_mask_path = os.path.join("ai_hub_dataset/train/resized_masks", mask_file)

# 이미지, 마스크 불러오기
img = cv2.imread(resized_img_path)
mask = cv2.imread(resized_mask_path, cv2.IMREAD_GRAYSCALE)

# 마스크를 컬러로 변환 (클래스별 색상 맵핑)
def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # 클래스 색상: (B, G, R)
    color_map = {
        0: (0, 0, 0),         # 배경 - 검정
        1: (255, 255, 255),   # 흰 실선 - 흰색
        2: (200, 200, 200),   # 흰 점선 - 회색
        3: (0, 255, 255),     # 노란 실선 - 노랑
        4: (0, 0, 255),       # 정지선 - 빨강
        5: (255, 0, 0),       # 횡단보도 - 파랑
    }

    for k, color in color_map.items():
        color_mask[mask == k] = color

    return color_mask

color_mask = mask_to_color(mask)

# 이미지와 컬러 마스크 합성
overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

# 결과 보여주기 (OpenCV 창)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
