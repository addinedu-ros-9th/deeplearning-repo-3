import os
import json

from tqdm import tqdm

# 설정
image_dir = "ai_hub_dataset/train/images"
label_dir = "ai_hub_dataset/train/labels"

# 삭제된 항목 카운터
invalid_count = 0
total_checked = 0

# 삭제 로그 저장 (선택)
invalid_log = []

# 모든 JSON 순회
for fname in tqdm(sorted(os.listdir(label_dir))):
    if not fname.endswith(".json"):
        continue

    label_path = os.path.join(label_dir, fname)
    image_name = fname.replace(".json", ".jpg")
    image_path = os.path.join(image_dir, image_name)

    total_checked += 1

    # JSON 로드
    try:
        with open(label_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[오류] JSON 파싱 실패: {label_path}")
        continue

    annotations = data.get("annotations", [])
    invalid = False

    for ann in annotations:
        coords = ann.get("data", [])
        if len(coords) < 2:
            invalid = True
            break

    if invalid:
        # 삭제 처리
        try:
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            invalid_log.append(fname)
            invalid_count += 1
        except Exception as e:
            print(f"[오류] 삭제 실패: {fname} - {e}")

# 최종 결과 출력
print(f"총 확인된 라벨: {total_checked}")
print(f"삭제된 부정확한 라벨/이미지 쌍: {invalid_count}")
print(f"남은 유효 라벨 수: {total_checked - invalid_count}")

# (선택) 삭제 로그 저장
with open("removed_invalid_labels.txt", "w") as f:
    for name in invalid_log:
        f.write(name + "\n")
