import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from lane_dataset_sdlane import LaneDataset
from unet import UNet
from mixed_loss import mixed_loss
from tqdm import tqdm
from torch.amp import autocast
import os

# 경로 설정
TRAIN_LIST = "SDLane/train/train_list.txt"
TRAIN_IMAGES = "SDLane/train/resized_images"
TRAIN_MASKS = "SDLane/train/resized_masks"
SAVE_PATH = "best_model_finetuned.pth"  # 추가학습 후 저장할 경로
LOAD_PATH = "best_model.pth"            # 기존 학습된 모델 경로

# 데이터셋 로딩
dataset = LaneDataset(TRAIN_LIST, TRAIN_IMAGES, TRAIN_MASKS)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# 모델 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=3).to(device)

# 저장된 모델 가중치 불러오기
if os.path.exists(LOAD_PATH):
    print(f"{LOAD_PATH} 로부터 가중치 불러오는 중...")
    model.load_state_dict(torch.load(LOAD_PATH, map_location=device))
else:
    print(f"{LOAD_PATH} 파일을 찾을 수 없습니다. 새 모델로 시작합니다.")

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 학습률은 기존보다 작게
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# AMP (자동 mixed precision) 설정 (선택)
scaler = torch.amp.GradScaler(device="cuda")

# 추가학습 루프
num_epochs = 5
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc=f"Fine-tuning Epoch {epoch+1}/{num_epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            preds = model(imgs)
            loss = mixed_loss(preds, masks, alpha=0.7)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    scheduler.step(avg_loss)
    print(f"[Fine-tuning Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    # 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"모델이 개선되어 {SAVE_PATH}로 저장되었습니다.")

print("추가 학습 완료!")
