import torch
import cv2
import numpy as np
from unet2 import UNet
from process_frame import process_frame


def predict_video_process_frame(model_path, input_video_path, output_video_path,
                                 input_size=(512, 256), show_live=True):
    color_map = {
        0: [0, 0, 0],         # 배경
        1: [255, 255, 255],   # 흰 실선
        2: [128, 128, 128],   # 흰 점선
        3: [0, 255, 255],     # 중앙선
        4: [0, 0, 255],       # 정지선
        5: [0, 255, 0],       # 횡단보도
    }

    model = UNet(num_classes=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"입력 영상 열기 실패: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    print("[INFO] 영상 추론 시작 (C=center, L=left, R=right, ESC=종료)")
    uuid_counter = 0
    frame_counter = 0
    mode = "center"
    mode_change_frame = None  # left/right 진입 시간 기록

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(
            frame=frame,
            model=model,
            device=device,
            uuid=uuid_counter,
            input_size=input_size,
            mode=mode,
            frame_count=frame_counter
        )
        uuid_counter += 1
        frame_counter += 1

        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # --- 조향각 시각화 ---
        angle = result["steering_angle"]
        center_x = result["steering_center_x"]
        if angle is not None:
            length = 100
            rad = np.radians(angle)
            start = (int(center_x * w / result['pred_mask_shape'][1]), h - 100)
            end = (int(start[0] + length * np.sin(rad)), int(start[1] - length * np.cos(rad)))
            cv2.arrowedLine(overlay, start, end, (0, 255, 0), 3)
            cv2.putText(overlay, f"Angle: {angle:.2f} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No Steering", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- 시각화 요소 추가 ---
        offset = result.get("offset")
        avg_x = result.get("avg_center_x_mask_res")
        fallback = result.get("fallback")

        if avg_x is not None:
            scale_x = w / result['pred_mask_shape'][1]
            x = int(avg_x * scale_x)
            cv2.line(overlay, (x, 0), (x, h), (0, 255, 255), 2)
            cv2.putText(overlay, "Lane Center", (x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if offset is not None:
            cv2.putText(overlay, f"Offset: {offset:+.1f}px", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 100), 2)

        # --- 스켈레톤 점 시각화 ---
        for x, y in result["skeleton_points"]:
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)

        # --- 기준 중앙선 표시 ---
        cx = int(result['steering_center_x'] * w / result['pred_mask_shape'][1])
        cv2.line(overlay, (cx, 0), (cx, h), (255, 255, 255), 1)

        # --- 마스크 시각화 ---
        color_mask = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_mask[result["pred_mask"] == class_id] = color
        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(overlay, 0.7, color_mask_resized, 0.3, 0)

        # --- 모드 표시 ---
        cv2.putText(overlay, f"Mode: {mode.upper()}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if fallback:
            cv2.putText(overlay, "[Fallback Line Used]", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        out.write(overlay)
        if show_live:
            cv2.imshow("Lane Steering", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                mode = "center"
                mode_change_frame = None
                print("[MODE] → CENTER")
            elif key == ord('l'):
                mode = "left"
                mode_change_frame = frame_counter
                print("[MODE] → LEFT")
            elif key == ord('r'):
                mode = "right"
                mode_change_frame = frame_counter
                print("[MODE] → RIGHT")

        # --- 자동 center 복귀 조건 ---
        if mode in ["left", "right"] and mode_change_frame is not None:
            elapsed = frame_counter - mode_change_frame
            if angle is not None and abs(angle) < 5.0 or elapsed > int(fps * 10):
                print("[AUTO] 안정화 or 시간 경과 → CENTER 전환")
                mode = "center"
                mode_change_frame = None

    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"[INFO] 저장 완료: {output_video_path}")