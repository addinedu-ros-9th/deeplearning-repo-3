import socket
import threading
import numpy as np
import cv2
import time
import torch
from process_frame import process_frame
from pack_lane_result import pack_lane_result
from unet2 import UNet
import json
import struct

TCP_SERVER_IP = "0.0.0.0"
TCP_SERVER_PORT = 12345

UDP_SERVER_IP = "0.0.0.0" # Listen on all available interfaces
UDP_SERVER_PORT = 54321

# --- Smoothing and State Variables (similar to predict_video.py) ---
prev_final_steering_angle = 0.0 # Stores the final smoothed angle from the PREVIOUS frame
prev_avg_center_x_mask_res = None # Stores avg_center_x in MASK resolution from PREVIOUS frame
max_x_change_mask_res = 3.0  # Max change in avg_center_x (mask pixels) per frame
delta_limit_deg = 0.2      # Max change in final steering angle (degrees) per frame - reduced for more smoothness
# Smoothing factor (0 < ema_alpha <= 1). 
# Smaller values = more smoothing.
ema_alpha = 0.02           # Reduced for more aggressive smoothing
angle_dead_zone_deg = 0.05 # New: If change is less than this, keep previous angle

# 모델 초기화
model = UNet(num_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/home/john/dev_ws/dl_project/best_model_final.pth", map_location=device))
model = model.to(device)
model.eval()

tcp_conn = None
tcp_lock = threading.Lock()

# 모드 상태 및 안정화 관리용 변수 (전역)
mode = "center"   # 초기 모드는 center로 시작
mode_lock = threading.Lock()

# 모드 자동 전환을 위한 파라미터
STEERING_ANGLE_THRESHOLD = 5.0  # 도 단위, 각도 임계값 (예: 5도 이하이면 중앙으로 전환 고려)
STABLE_FRAME_COUNT_THRESHOLD = 10  # 몇 프레임 연속 안정 시 전환
stable_frame_count = 0

SERVER_ANNOTATED_FRAME_WINDOW_NAME = "Server - Annotated Frame"
SERVER_PRED_MASK_WINDOW_NAME = "Server - Predicted Mask"


def handle_client(conn, addr):
    global tcp_conn
    print(f"[TCP] Connected from {addr}")
    with tcp_lock:
        tcp_conn = conn
    try:
        pass
    except:
        pass
    finally:
        with tcp_lock:
            if tcp_conn == conn: # Ensure clearing the correct connection
                tcp_conn = None
        conn.close()
        print(f"[TCP] Disconnected from {addr}")


def udp_video_receiver():
    global tcp_conn, mode, stable_frame_count, prev_final_steering_angle, prev_avg_center_x_mask_res
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind((UDP_SERVER_IP, UDP_SERVER_PORT))
    print(f"[UDP] Server listening on {UDP_SERVER_IP}:{UDP_SERVER_PORT}")

    fps_limit = 30
    frame_interval = 1.0 / fps_limit
    prev_time = 0

    try:
        while True:
            data, addr = udp_server.recvfrom(65535)
            try:
                if b'||' not in data:
                    print("[UDP] Invalid packet, missing delimiter")
                    continue

                header, img_data = data.split(b'||', 1)
                if len(header) != 4:
                    print(f"[UDP] Invalid UUID header length: {len(header)}")
                    continue

                uuid = int.from_bytes(header, 'big')

                np_data = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                current_time = time.time()
                if current_time - prev_time < frame_interval:
                    continue
                prev_time = current_time

                # 현재 모드를 복사해서 안전하게 사용
                with mode_lock:
                    current_mode = mode

                # 추론 + 결과
                result = process_frame(frame, model, device, uuid, mode=current_mode)
                
                # Extract raw values from process_frame
                raw_offset_from_process = result.get("offset") 
                raw_angle_from_process = result.get("steering_angle")
                avg_center_x_mask_res_from_process = result.get("avg_center_x_mask_res")
                pred_mask_shape_from_process = result.get("pred_mask_shape")
                pred_mask_from_process = result.get("pred_mask", None)

                # --- Apply Smoothing and Limiting (similar to predict_video.py) ---
                current_limited_avg_x_mask = avg_center_x_mask_res_from_process
                
                # 1. X-Coordinate Change Limit
                if prev_avg_center_x_mask_res is not None and current_limited_avg_x_mask is not None:
                    dx = current_limited_avg_x_mask - prev_avg_center_x_mask_res
                    if abs(dx) > max_x_change_mask_res:
                        current_limited_avg_x_mask = prev_avg_center_x_mask_res + np.sign(dx) * max_x_change_mask_res
                
                # Update prev_avg_center_x_mask_res for the next frame
                # Use the value *before* potential None, if current_limited_avg_x_mask became None later
                prev_avg_center_x_mask_res = current_limited_avg_x_mask if current_limited_avg_x_mask is not None else avg_center_x_mask_res_from_process

                # 2. Recalculate Offset and Angle using limited_avg_x
                final_offset_to_send = raw_offset_from_process # Default to raw if no valid limited_avg_x
                angle_for_ema = raw_angle_from_process       # Default to raw if no valid limited_avg_x

                if current_limited_avg_x_mask is not None and pred_mask_shape_from_process is not None:
                    h_mask, w_mask = pred_mask_shape_from_process
                    frame_center_x_mask = w_mask // 2
                    final_offset_to_send = float(current_limited_avg_x_mask - frame_center_x_mask)
                    
                    # Use h_mask // 2 as lookahead, similar to process_frame and predict_video
                    angle_rad = np.arctan2(final_offset_to_send, h_mask // 2) 
                    angle_for_ema = float(np.degrees(angle_rad))
                
                # 3. Calculate Potential Next Angle with EMA
                if angle_for_ema is not None:
                    potential_ema_angle = (1 - ema_alpha) * prev_final_steering_angle + ema_alpha * angle_for_ema
                else: # If angle_for_ema is None, aim to maintain previous angle
                    potential_ema_angle = prev_final_steering_angle
                    if final_offset_to_send is None: final_offset_to_send = 0.0 # Ensure offset is float

                # 4. Apply Dead Zone
                # If the change proposed by EMA is very small, consider it as no change.
                if abs(potential_ema_angle - prev_final_steering_angle) < angle_dead_zone_deg:
                    # Stay with the previous angle if change is within dead zone
                    angle_after_dead_zone = prev_final_steering_angle
                else:
                    # Otherwise, accept the EMA-smoothed angle as the target for this frame
                    angle_after_dead_zone = potential_ema_angle

                # 5. Delta Limiting
                # Limit the change from the previous final angle to the angle_after_dead_zone
                delta = angle_after_dead_zone - prev_final_steering_angle
                if abs(delta) > delta_limit_deg:
                    delta = np.sign(delta) * delta_limit_deg
                
                final_steering_angle_to_send = prev_final_steering_angle + delta
                prev_final_steering_angle = final_steering_angle_to_send # Update for next frame
                # Update result dictionary for packing and display
                result["offset"] = final_offset_to_send
                result["steering_angle"] = final_steering_angle_to_send
                # --- End of Smoothing ---
                    
                # Use final_steering_angle_to_send for mode switching logic
                abs_angle_for_mode_switch = abs(final_steering_angle_to_send)

                # Annotate frame for server-side display
                display_frame = frame.copy() 

                # # Draw skeleton points (drivable path from model)
                # if result.get("skeleton_points"):
                #     points_to_draw = result["skeleton_points"]
                #     if len(points_to_draw) > 1:
                #         pts_np = np.array(points_to_draw, dtype=np.int32)
                #         cv2.polylines(display_frame, [pts_np], isClosed=False, color=(0, 255, 0), thickness=3) # Green
                #     elif len(points_to_draw) == 1:
                #         cv2.circle(display_frame, tuple(points_to_draw[0]), 3, (0,255,0), -1)

                # Draw frame center and detected centerline (predict_video.py style)
                orig_h, orig_w = display_frame.shape[:2]
                if pred_mask_shape_from_process is not None:
                    mask_h, mask_w = pred_mask_shape_from_process
                    scale_to_orig_w = orig_w / mask_w
                    scale_to_orig_h = orig_h / mask_h

                    # Frame center line (Cyan)
                    cv2.line(display_frame, (orig_w // 2, orig_h), (orig_w // 2, int(orig_h * 0.5)), (255, 255, 0), 1) # Cyan

                    # Detected centerline (Yellow) - using current_limited_avg_x_mask
                    if current_limited_avg_x_mask is not None:
                        cx_orig = int(current_limited_avg_x_mask * scale_to_orig_w)
                        cy_bottom_orig = orig_h # int(mask_h * scale_to_orig_h)
                        cy_top_orig = int(mask_h * 0.5 * scale_to_orig_h) # Draw up to halfway the mask height
                        cv2.line(display_frame, (cx_orig, cy_bottom_orig), (cx_orig, cy_top_orig), (0, 255, 255), 2) # Yellow

                # Draw steering angle, offset text
                text_offset = f"Offset: {final_offset_to_send:.2f}" if final_offset_to_send is not None else "Offset: N/A"
                text_angle = f"Steering: {final_steering_angle_to_send:.2f} deg"
                text_mode = f"Mode: {current_mode}"
                cv2.putText(display_frame, text_offset, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, text_angle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, text_mode, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Overlay the predicted mask if available
                if pred_mask_from_process is not None:
                    # Create a colorized version of the mask for display
                    # pred_mask_from_process is (H, W) with values 0, 1, 2
                    mask_h_pred, mask_w_pred = pred_mask_from_process.shape
                    colorized_mask = np.zeros((mask_h_pred, mask_w_pred, 3), dtype=np.uint8)
                    colorized_mask[pred_mask_from_process == 1] = [255, 0, 0]  # Class 1: Blue
                    colorized_mask[pred_mask_from_process == 2] = [0, 0, 255]  # Class 2: Red
                    
                    # Resize colorized_mask to match display_frame dimensions
                    resized_colorized_mask = cv2.resize(colorized_mask, 
                                                        (display_frame.shape[1], display_frame.shape[0]), 
                                                        interpolation=cv2.INTER_NEAREST)

                    # Blend display_frame with the resized_colorized_mask
                    blended_frame = cv2.addWeighted(display_frame, 0.7, resized_colorized_mask, 0.3, 0.0)
                    cv2.imshow(SERVER_ANNOTATED_FRAME_WINDOW_NAME, blended_frame)
                else: # pred_mask_from_process is None
                    cv2.imshow(SERVER_ANNOTATED_FRAME_WINDOW_NAME, display_frame)

                cv2.waitKey(1)


                # 모드 자동 전환 로직 (using abs_angle_for_mode_switch)
                with mode_lock:
                    if current_mode in ["left", "right"]:
                        if abs_angle_for_mode_switch < STEERING_ANGLE_THRESHOLD:
                            stable_frame_count += 1
                            if stable_frame_count >= STABLE_FRAME_COUNT_THRESHOLD:
                                mode = "center"
                                stable_frame_count = 0
                                print(f"[MODE SWITCH] Stable steering detected, switching to CENTER mode")
                        else:
                            stable_frame_count = 0
                    elif current_mode == "center":
                        stable_frame_count = 0 
                
                # Ensure offset is also a float for packing if it was None
                if result.get("offset") is None:
                    result["offset"] = 0.0
                if result.get("steering_angle") is None: # Should be set by smoothing logic
                    result["steering_angle"] = 0.0

                # result_bytes = pack_lane_result(result)
                result_bytes = json.dumps(result).encode('utf-8')
                length = len(result_bytes)

                header = struct.pack('>I', length)
                packet = header + result_bytes

                if uuid % 30 == 0:
                    print(f"[TCP] Frame UUID: {uuid}, Mode: {current_mode}, Sent Steering Angle: {result['steering_angle']:.2f}")

                with tcp_lock:
                    if tcp_conn:
                        try:
                            tcp_conn.sendall(packet)
                        except Exception as e:
                            print(f"[TCP SEND ERROR] UUID: {uuid}, Error: {e}")
                            tcp_conn = None

            except Exception as e:
                print(f"[UDP FRAME ERROR] {e}")

    except Exception as e:
        print(f"[UDP ERROR] {e}")
    finally:
        udp_server.close()
        cv2.destroyAllWindows()


def main():
    threading.Thread(target=udp_video_receiver, daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((TCP_SERVER_IP, TCP_SERVER_PORT))
        server.listen()
        print(f"[TCP] Server listening on {TCP_SERVER_IP}:{TCP_SERVER_PORT}")

        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
