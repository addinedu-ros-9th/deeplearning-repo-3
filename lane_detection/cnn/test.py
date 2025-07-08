from predict_and_save_mask import predict_and_save_mask
import os
import glob

# 경로 설정
train_root = os.path.abspath("SDLane/test")
images_root = os.path.join(train_root, "images")
model_path = "best_model.pth"
output_root = "inference_results"

# 결과 저장 폴더 생성
os.makedirs(output_root, exist_ok=True)

# 모든 이미지 경로 탐색 (UUID 하위 폴더 포함)
image_paths = glob.glob(os.path.join(images_root, "*", "*.jpg"))

for img_path in image_paths[:3]:
    # 출력 파일명 생성
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_root, filename.replace(".jpg", "_mask.png"))

    # 추론 및 저장
    predict_and_save_mask(model_path, img_path, save_path)
