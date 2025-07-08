import socket
import json
import cv2
import threading
import time
import numpy as np

# ===== 서버 IP 및 포트 설정 =====
LANE_SERVER_IP = "192.168.0.20"  # 차선 인식 서버
OBJ_SERVER_IP  = "192.168.0.21"  # 객체 탐지 서버

TCP_SERVER_PORT = 12345
UDP_SERVER_PORT = 54321

# ===== TCP 클라이언트 함수 =====
def tcp_client(server_ip, shutdown_event, name=""):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.connect((server_ip, TCP_SERVER_PORT))
            print(f"[{name} TCP CLIENT] Connected to {server_ip}:{TCP_SERVER_PORT}.")
            client.settimeout(1.0)

            message_counter = 0
            while not shutdown_event.is_set():
                message_counter += 1
                message = {
                    "type": "tcp_request",
                    "id": message_counter,
                    "data": f"Hello from {name} TCP client"
                }

                try:
                    client.sendall(json.dumps(message).encode('utf-8'))
                    print(f"[{name} TCP CLIENT] Sent: {message}")

                    data = client.recv(1024)
                    if not data:
                        print(f"[{name} TCP CLIENT] Server closed connection.")
                        break
                    response = json.loads(data.decode('utf-8'))
                    print(f"[{name} TCP CLIENT] Received: {response}")

                except socket.timeout:
                    continue
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"[{name} TCP CLIENT ERROR] Connection error: {e}")
                    break
                except json.JSONDecodeError as e:
                    print(f"[{name} TCP CLIENT ERROR] JSON decode failed: {e}")
                except Exception as e:
                    print(f"[{name} TCP CLIENT ERROR] Unexpected error: {e}")
                    break

                for _ in range(10):
                    if shutdown_event.is_set():
                        break
                    time.sleep(0.1)

        except Exception as e:
            print(f"[{name} TCP CLIENT ERROR] Connection failed: {e}")
        finally:
            print(f"[{name} TCP CLIENT] Connection closed.")

# ===== UDP 클라이언트 함수 =====
def udp_client(shutdown_event):
    input_video_path = "/home/john/dev_ws/dl_project/data/sample_video4.mp4"
    output_video_path = "/home/john/dev_ws/dl_project/inference_results/sample_video_mask4_final.mp4"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        print("[UDP CLIENT] Started.")
        client_socket.settimeout(0.5)
        display_window_name = "UDP Client - Video Stream"

        while cap.isOpened() and not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[UDP CLIENT] End of video or frame grab failed.")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            if not ret:
                print("[UDP CLIENT] Frame encoding failed.")
                continue

            frame_bytes = buffer.tobytes()

            try:
                # 두 서버에 모두 전송
                client_socket.sendto(frame_bytes, (LANE_SERVER_IP, UDP_SERVER_PORT))
                client_socket.sendto(frame_bytes, (OBJ_SERVER_IP, UDP_SERVER_PORT))

                # 응답은 차선 인식 서버로부터만 수신 (선택적)
                try:
                    data_received, _ = client_socket.recvfrom(65536)
                    processed_buffer = np.frombuffer(data_received, dtype=np.uint8)
                    processed_frame = cv2.imdecode(processed_buffer, cv2.IMREAD_COLOR)
                    if processed_frame is not None:
                        cv2.imshow(display_window_name, processed_frame)
                    else:
                        cv2.imshow(display_window_name, frame_resized)
                        print("[UDP CLIENT] Failed to decode processed frame.")

                except socket.timeout:
                    cv2.imshow(display_window_name, frame_resized)
                    print("[UDP CLIENT] Timeout waiting for response.")

            except Exception as e:
                print(f"[UDP CLIENT ERROR] Failed to send frame: {e}")
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                print("[UDP CLIENT] ESC pressed, stopping.")
                break

        print("[UDP CLIENT] Cleaning up.")
        cap.release()
        cv2.destroyAllWindows()

# ===== MAIN =====
if __name__ == "__main__":
    shutdown_event = threading.Event()

    tcp_lane_thread = threading.Thread(target=tcp_client, args=(LANE_SERVER_IP, shutdown_event, "LANE"), daemon=True)
    tcp_obj_thread = threading.Thread(target=tcp_client, args=(OBJ_SERVER_IP, shutdown_event, "OBJ"), daemon=True)
    udp_thread = threading.Thread(target=udp_client, args=(shutdown_event,), daemon=True)

    print("[MAIN] Starting TCP and UDP clients...")
    tcp_lane_thread.start()
    tcp_obj_thread.start()
    udp_thread.start()

    try:
        while True:
            if not tcp_lane_thread.is_alive() and not tcp_obj_thread.is_alive() and not udp_thread.is_alive():
                print("[MAIN] All threads finished.")
                break
            if shutdown_event.is_set():
                print("[MAIN] Shutdown event detected.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Ctrl+C detected, stopping all threads...")
        shutdown_event.set()
