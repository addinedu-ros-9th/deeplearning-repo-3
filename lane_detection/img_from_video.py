import cv2
import os

# 경로 지정
VIDEO_PATH = ""  # 여기에 영상 경로 입력
OUTPUT_PATH = "my_env/image"  # 저장할 디렉토리

# 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Cannot open video file: {VIDEO_PATH}")
    exit()

frame_idx = 1

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or failed to read frame.")
        break

    # 000001.jpg, 000002.jpg ...
    filename = f"{frame_idx:06d}.jpg"
    full_output_path = os.path.join(OUTPUT_PATH, filename)

    cv2.imwrite(full_output_path, frame)
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"Total {frame_idx - 1} frames saved in '{OUTPUT_PATH}'.")
