import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
# from lane_dataset_sdlane import LaneDataset
from lane_dataset_aihub import LaneDataset
from unet2 import UNet
from mixed_loss import mixed_loss
# from mixed_loss2 import FocalTverskyLoss
from tqdm import tqdm
import os
from torch.amp import autocast
import matplotlib.pyplot as plt
from iou import compute_iou


# 경로 설정
# TRAIN_LIST = "SDLane/train/train_list.txt"
# TRAIN_IMAGES = "SDLane/train/resized_images"
# TRAIN_MASKS = "SDLane/train/resized_masks"
# SAVE_PATH = "best_model.pth"

TRAIN_LIST = "ai_hub_dataset/train/train_list.txt"
TRAIN_IMAGES = "ai_hub_dataset/train/resized_images"
TRAIN_MASKS = "ai_hub_dataset/train/resized_masks"
SAVE_PATH = "best_model.pth"

os.makedirs("loss_result", exist_ok=True)



# 전체 dataset 생성
dataset = LaneDataset(TRAIN_LIST, TRAIN_IMAGES, TRAIN_MASKS)

# 비율로 나누기 (예: train 80%, val 20%)
val_ratio = 0.2
val_len = int(len(dataset) * val_ratio)
train_len = len(dataset) - val_len

# 고정된 시드로 재현성 유지
train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

# 각 DataLoader 생성
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

# 모델 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=6).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# AMP (자동 mixed precision) 설정 (선택)
# float 16 연산 보조 클래스
# 표현 범위가 작아 작은 값이 0이 되거나 nan이 될 수 있는데 그것을 방지
scaler = torch.amp.GradScaler(device="cuda")

print(f"Train size: {len(train_set)}, Validation size: {len(val_set)}")


# 학습 루프
num_epochs = 50
best_val_loss = float("inf")

epochs_no_improve = 0
early_stop_patience = 10

train_losses = []
val_losses = []
miou_list = []
per_class_iou_list = [[] for _ in range(6)]  # 클래스 수에 맞게 설정

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    print(f"[Epoch : {epoch+1}] Training...")
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda"): # 모델과 loss 계산을 자동으로 float16으로 수행
            outputs = model(images)
            loss = mixed_loss(outputs, masks) # focal + dice loss

        scaler.scale(loss).backward() # 작은 값이 float16에서 underflow 나는 걸 방지
        scaler.step(optimizer)
        scaler.update()

        # Cross Entropy
        # loss = F.cross_entropy(outputs, masks) 

        # Focal Tversky Loss
        # criterion = FocalTverskyLoss()           # Loss 함수 객체 생성
        # loss = criterion(outputs, masks)         # forward()로 호출

        # loss.backward()
        # optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    iou_scores = []

    with torch.no_grad():
        print(f"[Epoch : {epoch+1}] Validating...")
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            # loss = criterion(outputs, masks)
            loss = mixed_loss(outputs, masks)
            # loss = F.cross_entropy(outputs, masks)
            val_loss += loss.item()

            # === IoU 계산 ===
            batch_ious = compute_iou(outputs, masks, num_classes=6)  # 클래스 수 맞게 설정
            iou_scores.append(batch_ious)

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    ious_tensor = torch.stack([torch.tensor(i) for i in iou_scores]) # iou_scores는 리스트
    avg_iou = ious_tensor.nanmean(dim=0)  # class별 평균 IoU
    mean_iou = avg_iou.nanmean().item()  # 전체 클래스 평균
    print(f"Mean IoU: {mean_iou}")
    
    class_names = ["background", "white_solid", "white_dotted", "yellow_solid", "stop_line", "crosswalk"]

    for i, iou in enumerate(avg_iou):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{name:12}: {iou:.4f}")

    # 리스트에 저장
    miou_list.append(mean_iou)
    for i, iou in enumerate(avg_iou):
        per_class_iou_list[i].append(iou.item())

    # === 모델 저장 ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model improved and saved at epoch {epoch+1}!")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    # === Early stopping 조건 체크 ===
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break



    # === 로그 출력 ===
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

print("학습 완료!")

# === 시각화 ===

# === Loss ===
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.savefig("./loss_result/loss_plot.png")
plt.close()

# === mIoU ===
plt.figure()
plt.plot(miou_list, label="Mean IoU", color="black")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("Mean IoU over Epochs")
plt.grid(True)
plt.legend()
plt.savefig("./loss_result/miou_plot.png")
plt.close()

# === Class-wise IoU ===
plt.figure(figsize=(10, 6))
for i, class_iou in enumerate(per_class_iou_list):
    plt.plot(class_iou, label=class_names[i])
plt.xlabel("Epoch")
plt.ylabel("Class IoU")
plt.title("Class-wise IoU over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("./loss_result/class_iou_plot.png")
plt.close()