import cv2
import socket

SERVER_IP = '192.168.2.30'  # 서버의 IP
SERVER_PORT = 54321  # UDP 포트

def send_video_via_udp():
    cap = cv2.VideoCapture(0)  # 기본 웹캠

    udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 인코딩: JPEG 포맷으로 압축
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        # UDP로 전송
        udp_client.sendto(buffer.tobytes(), (SERVER_IP, SERVER_PORT))

        cv2.imshow("Sending...", frame)
        if cv2.waitKey(1) == 27:  # ESC 키
            break

    cap.release()
    udp_client.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    send_video_via_udp()
