import torch
import cv2
import numpy as np
from unet1 import UNet

def predict_and_save_mask(model_path, input_image_path, output_mask_path, input_size=(512, 256)):
    """
    단일 이미지에 대한 세그멘테이션 마스크를 예측하고 저장

    Args:
        model_path (str): 학습된 모델의 경로 (.pth)
        input_image_path (str): 입력 이미지 경로
        output_mask_path (str): 저장할 마스크 이미지 경로
        input_size (tuple): 모델 입력 사이즈 (width, height)
    """
    # ===== 모델 로딩 =====
    model = UNet(num_classes=3)
    # model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 이미지 전처리 =====
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)

    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # ===== 추론 =====
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1)
        pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)

    # ===== 시각화용 컬러 마스크 생성 =====
    h, w = input_size[1], input_size[0]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[pred_mask_np == 1] = [0, 255, 0]   # 클래스 1: 초록 (왼쪽 차선)
    color_mask[pred_mask_np == 2] = [0, 0, 255]   # 클래스 2: 파랑 (오른쪽 차선)

    # ===== 저장 =====
    cv2.imwrite(output_mask_path, color_mask)
    print(f"마스크 저장 완료: {output_mask_path}")
