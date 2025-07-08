import socket
import threading
import numpy as np
import cv2
import time
import torch
from process_frame_refactored import process_frame
from unet2 import UNet
import json
import struct
import base64


TCP_SERVER_IP = "0.0.0.0"
TCP_SERVER_PORT = 12345

UDP_SERVER_IP = "0.0.0.0" # Listen on all available interfaces
UDP_SERVER_PORT = 54321


# 모델 초기화
model = UNet(num_classes=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/home/john/dev_ws/dl_project/cnn/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

tcp_conn = None
tcp_lock = threading.Lock()

SERVER_ANNOTATED_FRAME_WINDOW_NAME = "Server - Annotated Frame"
SERVER_PRED_MASK_WINDOW_NAME = "Server - Predicted Mask"

# 색상 정의: (BGR)
color_map = {
    0: [0, 0, 0],         # 배경 - 검정
    1: [255, 255, 255],   # 흰색 실선 - 흰색
    2: [128, 128, 128],   # 흰색 점선 - 회색
    3: [0, 255, 255],     # 중앙선(노란 실선) - 노랑
    4: [0, 0, 255],       # 정지선 - 빨강
    5: [0, 255, 0],       # 횡단보도 - 초록
}


def handle_client(conn, addr):
    global tcp_conn
    print(f"[TCP] Connected from {addr}")
    with tcp_lock:
        tcp_conn = conn
    try:
        while True:
            time.sleep(1)  # 연결을 유지
    except:
        pass
    finally:
        with tcp_lock:
            if tcp_conn == conn: # Ensure clearing the correct connection
                tcp_conn = None
        conn.close()
        print(f"[TCP] Disconnected from {addr}")

def encode_mask_png_base64(mask: np.ndarray) -> str:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    success, encoded_img = cv2.imencode('.png', mask)
    if not success:
        raise ValueError("mask PNG 인코딩 실패")
    return base64.b64encode(encoded_img).decode('utf-8')


def udp_video_receiver():
    global tcp_conn
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind((UDP_SERVER_IP, UDP_SERVER_PORT))
    print(f"[UDP] Server listening on {UDP_SERVER_IP}:{UDP_SERVER_PORT}")

    fps_limit = 20
    frame_interval = 1.0 / fps_limit
    prev_time = 0

    try:
        while True:
            data, addr = udp_server.recvfrom(65535)
            try:
                if b'||' not in data:
                    print("[UDP] Invalid packet, missing delimiter")
                    continue

                uuid, img_data = data.split(b'||', 1)
                if len(uuid) != 4:
                    print(f"[UDP] Invalid UUID header length: {len(uuid)}")
                    continue

                np_data = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                current_time = time.time()
                if current_time - prev_time < frame_interval:
                    continue
                prev_time = current_time


                # 추론 + 결과
                result = process_frame(frame, model, device)
                
                # Extract raw values from process_frame
                pred_mask_from_process = result.get("pred_mask", None)
                    

                # Annotate frame for server-side display
                display_frame = frame.copy() 

                # Overlay the predicted mask if available
                if pred_mask_from_process is not None:
                    # Create a colorized version of the mask for display
                    # pred_mask_from_process is (H, W) with values 0, 1, 2, 3, 4, 5
                    mask_h_pred, mask_w_pred = pred_mask_from_process.shape

                    # Mask colorized
                    colorized_mask = np.zeros((mask_h_pred, mask_w_pred, 3), dtype=np.uint8)
                    for class_id, color in color_map.items():
                        colorized_mask[result.get("pred_mask") == class_id] = color

                    # Resize colorized_mask to match display_frame dimensions
                    resized_colorized_mask = cv2.resize(colorized_mask, 
                                                        (display_frame.shape[1], display_frame.shape[0]), 
                                                        interpolation=cv2.INTER_NEAREST)
                    
                    result_resized = {
                        "pred_mask": resized_colorized_mask
                    
                    }

                    # Blend display_frame with the resized_colorized_mask
                    blended_frame = cv2.addWeighted(display_frame, 0.7, resized_colorized_mask, 0.3, 0.0)
                    cv2.imshow(SERVER_ANNOTATED_FRAME_WINDOW_NAME, blended_frame)

                else: # pred_mask_from_process is None
                    cv2.imshow(SERVER_ANNOTATED_FRAME_WINDOW_NAME, display_frame)

                cv2.waitKey(10)

                # 전처리
                if "pred_mask" in result:
                    result["pred_mask"] = encode_mask_png_base64(result["pred_mask"])

                # result_bytes = pack_lane_result(result)
                result_bytes = json.dumps(result).encode('utf-8')
                length = len(result_bytes)

                header = struct.pack('>I', length)
                packet = header + uuid + result_bytes

                # if "pred_mask" in result_resized:
                #     result_resized["pred_mask"] = encode_mask_png_base64(result_resized["pred_mask"])

                # result_bytes = json.dumps(result_resized).encode('utf-8')
                # length = len(result_bytes)

                # header = struct.pack('>I', length)
                # packet = header + uuid + result_bytes

                # if uuid % 30 == 0:
                #     print(f"[TCP] Frame UUID: {uuid}")

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
