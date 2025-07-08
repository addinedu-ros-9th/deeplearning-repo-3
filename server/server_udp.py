import cv2
import numpy as np
import socket

UDP_SERVER_IP = "0.0.0.0"
UDP_SERVER_PORT = 54321

def udp_video_receiver():
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind((UDP_SERVER_IP, UDP_SERVER_PORT))
    print(f"[PC2] UDP Server listening on {UDP_SERVER_IP}:{UDP_SERVER_PORT}")
    
    try:
        while True:
            try:
                data, addr = udp_server.recvfrom(65535)
                np_data = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.imshow("Received Video", frame)
                    if cv2.waitKey(1) == 27:  # ESC
                        break
            except KeyboardInterrupt:
                print("[PC2] KeyboardInterrupt detected. Exiting...")
                break
            except Exception as e:
                print(f"[UDP ERROR] {e}")
                continue

    finally:
        print("[PC2] Closing UDP server and destroying windows.")
        udp_server.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    udp_video_receiver()
