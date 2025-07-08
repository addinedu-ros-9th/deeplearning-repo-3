import cv2
import torch
import numpy as np
from typing import Tuple, Dict
import base64


def preprocess_image(frame: np.ndarray, input_size: Tuple[int, int] = (512, 256)) -> torch.Tensor:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def infer_mask(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def process_frame(frame: np.ndarray, model: torch.nn.Module, device: torch.device,
                  input_size: Tuple[int, int] = (512, 256)) -> Dict:
    
    img_tensor = preprocess_image(frame, input_size)
    pred_mask = infer_mask(model, img_tensor, device)

    return {
        "pred_mask": pred_mask
    }
