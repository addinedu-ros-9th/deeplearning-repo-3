import torch
import cv2
import numpy as np
from unet2 import UNet

def predict_video(model_path, input_video_path, output_video_path, input_size=(512, 256), show_live=True):
    """
    영상에 대한 세그멘테이션 마스크를 프레임마다 예측하고, 마스크 영상 저장

    Args:
        model_path (str): 학습된 모델의 경로 (.pth)
        input_video_path (str): 입력 영상 경로
        output_video_path (str): 저장할 출력 영상 경로
        input_size (tuple): 모델 입력 사이즈 (width, height)
    """
    # ===== 모델 로딩 =====
    model = UNet(num_classes=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 영상 열기 =====
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    print("영상 추론 시작...")

    # 색상 정의: (BGR)
    color_map = {
        0: [0, 0, 0],         # 배경 - 검정
        1: [255, 255, 255],   # 흰색 실선 - 흰색
        2: [128, 128, 128],   # 흰색 점선 - 회색
        3: [0, 255, 255],     # 중앙선(노란 실선) - 노랑
        4: [0, 0, 255],       # 정지선 - 빨강
        5: [0, 255, 0],       # 횡단보도 - 초록
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()

        # 전처리
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, input_size)
        img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # 추론
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 마스크 컬러화
        color_mask = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_mask[pred_mask == class_id] = color

        # 마스크와 원본 합성 (alpha blending)
        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(original_frame, 0.7, color_mask_resized, 0.3, 0)

        # 저장
        out.write(overlay)

        # 실시간 디스플레이
        if show_live:
            cv2.imshow("Segmentation", overlay)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
                print("ESC 누름 → 영상 중단")
                break

    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"저장 완료: {output_video_path}")
