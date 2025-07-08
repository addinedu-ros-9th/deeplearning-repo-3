import socket
import threading
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO

import json
import struct

TCP_SERVER_IP = "0.0.0.0"
TCP_SERVER_PORT = 12346

UDP_SERVER_IP = "0.0.0.0" # Listen on all available interfaces
UDP_SERVER_PORT = 54322


tcp_conn = None
tcp_lock = threading.Lock()

model_path = "/home/john/dev_ws/dl_project/server/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)


def handle_client(conn, addr):
    global tcp_conn
    print(f"[TCP] Connected from {addr}")
    with tcp_lock:
        tcp_conn = conn
    try:
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"[TCP ERROR] {e}")
    finally:
        with tcp_lock:
            if tcp_conn == conn: # Ensure clearing the correct connection
                tcp_conn = None
        conn.close()
        print(f"[TCP] Disconnected from {addr}")


def udp_video_receiver():
    global tcp_conn
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

                uuid, img_data = data.split(b'||', 1)
                if len(uuid) != 4:
                    print(f"[UDP] Invalid UUID uuid length: {len(uuid)}")
                    continue

                np_data = np.frombuffer(img_data, dtype=np.uint8)
                frame_raw = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                frame = frame_raw.copy()

                if frame is None:
                    continue

                current_time = time.time()
                if current_time - prev_time < frame_interval:
                    continue
                prev_time = current_time

                detections = model(frame)[0]
                result = []

                class_names = model.names

                for det in detections.boxes:
                    cls_id = int(det.cls)
                    conf = float(det.conf)
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    result.append({
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                    # 라벨 텍스트 생성
                    cls_name = class_names[cls_id]
                    label = f"{cls_name} {conf:.2f}"

                    # 텍스트 배경
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)

                    # 텍스트 쓰기
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
                    
                    cv2.imshow("Detections", frame)
                    cv2.waitKey(10)

                result_bytes = json.dumps(result).encode('utf-8')
                length = len(result_bytes)

                header = struct.pack('>I', length)
                packet = header + uuid + result_bytes

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
