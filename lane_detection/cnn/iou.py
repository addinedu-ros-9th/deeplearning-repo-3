import torch

def compute_iou(preds, masks, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)  # B x H x W

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = torch.logical_and(pred_cls, mask_cls).sum().item()
        union = torch.logical_or(pred_cls, mask_cls).sum().item()

        if union == 0:
            ious.append(float('nan'))  # 해당 class가 없을 경우 NaN 처리
        else:
            ious.append(intersection / union)

    return ious  # class별 IoU list
