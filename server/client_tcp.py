import socket
import json

SERVER_IP = '192.168.2.30'
SERVER_PORT = 12345

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((SERVER_IP, SERVER_PORT))
        print("[CLIENT] Connected to server.")

        try:
            # 서버에 메시지 전송
            client.sendall(b"Hello Server")
            # 서버로부터 응답 수신
            data = client.recv(1024)
            message = json.loads(data.decode('utf-8'))
            print(f"[CLIENT] Received: {message}")
        except Exception as e:
            print(f"[CLIENT ERROR] {e}")
        finally:
            print("[CLIENT] Connection closed.")

if __name__ == "__main__":
    main()
