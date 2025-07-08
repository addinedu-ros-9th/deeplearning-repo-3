import torch
import cv2
import numpy as np
from unet2 import UNet

def predict_webcam(model_path, input_size=(512, 256), save_output=False, output_path="webcam_output.mp4", show_live=True):
    """
    웹캠 실시간 세그멘테이션 추론 및 시각화

    Args:
        model_path (str): 학습된 모델의 경로 (.pth)
        input_size (tuple): 모델 입력 사이즈 (width, height)
        save_output (bool): 결과 저장 여부
        output_path (str): 저장 경로
        show_live (bool): 실시간 디스플레이 여부
    """
    # ===== 모델 로딩 =====
    model = UNet(num_classes=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 웹캠 열기 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    # ===== 저장 설정 =====
    if save_output:
        fps = 20.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("웹캠 추론 시작 (ESC 키로 종료)...")

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
            print("웹캠 프레임을 읽을 수 없습니다.")
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

        # 원본 크기로 리사이즈 및 합성
        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(original_frame, 0.7, color_mask_resized, 0.3, 0)

        # 출력
        if save_output:
            out.write(overlay)
        if show_live:
            cv2.imshow("Webcam Segmentation", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC 누름 → 종료")
                break

    cap.release()
    if save_output:
        out.release()
    if show_live:
        cv2.destroyAllWindows()

    if save_output:
        print(f"저장 완료: {output_path}")
