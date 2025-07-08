import socket
import struct
import threading
import json
import numpy as np
import cv2

TCP_SERVER_IP = '0.0.0.0'
TCP_SERVER_PORT = 12345

def handle_client(conn, addr):
    print(f"[PC2] Connected from {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            result_msg = {"message": "sample"}
            conn.sendall(json.dumps(result_msg).encode("utf-8"))
            print(f"[PC2] Sent verification result: {result_msg}")

    except Exception as e:
        print(f"[PC2 ERROR] {e}")
    finally:
        conn.close()
        print(f"[PC2] Disconnected from {addr}")


def main():
    # TCP 서버 시작
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((TCP_SERVER_IP, TCP_SERVER_PORT))
        server.listen()
        print(f"[PC2] TCP Server listening on {TCP_SERVER_IP}:{TCP_SERVER_PORT}")

        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()

