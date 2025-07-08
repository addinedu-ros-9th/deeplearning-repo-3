import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple, List, Dict, Optional
import scipy.ndimage


def preprocess_image(frame: np.ndarray, input_size: Tuple[int, int] = (512, 256)) -> torch.Tensor:
    """
    BGR -> RGB, resize, normalize, tensor 변환
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def infer_mask(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    모델 추론 후 class mask 반환 (H, W) uint8
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def extract_center_lane_mask(pred_mask: np.ndarray, center_ratio: float = 0.5) -> np.ndarray:
    """
    중앙 차선 마스크 추출 (1 or 2 클래스만 포함)
    """
    h, w = pred_mask.shape
    start_x = int(w * (0.5 - center_ratio / 2))
    end_x = int(w * (0.5 + center_ratio / 2))
    center_zone = np.zeros_like(pred_mask, dtype=bool)
    center_zone[:, start_x:end_x] = True
    center_mask = np.logical_and(np.isin(pred_mask, [1, 2]), center_zone)
    return center_mask.astype(np.uint8)


def extract_left_lane_mask(pred_mask: np.ndarray, lane_width_ratio: float = 0.5) -> np.ndarray:
    """
    좌측 두 개 차선을 기반으로 마스크 생성 (class 1 + class 2)
    """
    h, w = pred_mask.shape
    end_x = int(w * lane_width_ratio)
    left_zone = np.zeros_like(pred_mask, dtype=bool)
    left_zone[:, :end_x] = True

    # class 1 (좌측 차선) 또는 class 2 (우측 차선) 중 좌측에 위치한 영역 추출
    left_mask = np.logical_and(np.logical_or(pred_mask == 1, pred_mask == 2), left_zone)

    return left_mask.astype(np.uint8)



def extract_right_lane_mask(pred_mask: np.ndarray, lane_width_ratio: float = 0.25) -> np.ndarray:
    """
    우측 차선 마스크 추출 (class 2)
    """
    h, w = pred_mask.shape
    start_x = int(w * (1 - lane_width_ratio))
    right_zone = np.zeros_like(pred_mask, dtype=bool)
    right_zone[:, start_x:] = True
    right_mask = np.logical_and(np.logical_or(pred_mask == 1, pred_mask == 2), right_zone)

    return right_mask.astype(np.uint8)


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    마스크에 대해 skeletonize 수행
    """
    return skeletonize(mask > 0).astype(np.uint8)


def extract_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """
    skeleton (H, W)에서 (y, x) 좌표 배열 추출
    """
    points = np.column_stack(np.where(skeleton > 0))
    return points


def remap_skeleton_coords(points: np.ndarray, orig_shape: Tuple[int, int, int], mask_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    skeleton (y, x) 좌표를 원본 이미지 크기로 리맵 (x, y) 튜플 리스트 반환
    """
    if points.size == 0:
        return []
    scale_x = orig_shape[1] / mask_shape[1]
    scale_y = orig_shape[0] / mask_shape[0]
    remapped = [(int(x * scale_x), int(y * scale_y)) for y, x in points]
    return remapped


def compute_centerline(pred_mask: np.ndarray, class1: int = 1, class2: int = 2) -> np.ndarray:
    """
    두 클래스(class1, class2)의 skeleton 사이 중간 중심선을 계산하여 반환 (y, x 좌표 np.ndarray)
    """
    # 좌/우 마스크 분리
    mask1 = (pred_mask == class1).astype(np.uint8)
    mask2 = (pred_mask == class2).astype(np.uint8)

    # Skeletonize
    skel1 = skeletonize(mask1 > 0).astype(np.uint8)
    skel2 = skeletonize(mask2 > 0).astype(np.uint8)

    # 좌표 추출 (y, x)
    pts1 = np.column_stack(np.where(skel1 > 0))
    pts2 = np.column_stack(np.where(skel2 > 0))

    if pts1.size == 0 or pts2.size == 0:
        return np.array([])

    # y값 기준 정렬
    pts1 = pts1[np.argsort(pts1[:, 0])]
    pts2 = pts2[np.argsort(pts2[:, 0])]

    center_pts = []
    y_vals = np.intersect1d(pts1[:, 0], pts2[:, 0])  # 공통 y 좌표만 사용

    for y in y_vals:
        x1_vals = pts1[pts1[:, 0] == y][:, 1]
        x2_vals = pts2[pts2[:, 0] == y][:, 1]
        if len(x1_vals) > 0 and len(x2_vals) > 0:
            x1_mean = np.mean(x1_vals)
            x2_mean = np.mean(x2_vals)
            x_center = int((x1_mean + x2_mean) / 2)
            center_pts.append((y, x_center))

    return np.array(center_pts)


def smooth_centerline(centerline: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    중심선 좌표를 Gaussian smoothing으로 부드럽게 만듦
    Args:
        centerline: (N, 2) ndarray, (y, x) 좌표
        sigma: smoothing 강도
    Returns:
        smoothed_centerline: (N, 2) ndarray
    """
    if centerline.size == 0:
        return centerline

    ys = centerline[:, 0]
    xs = centerline[:, 1]

    # y는 그대로, x만 smoothing
    xs_smooth = scipy.ndimage.gaussian_filter1d(xs.astype(float), sigma=sigma)

    return np.stack((ys, xs_smooth), axis=1).astype(np.int32)


def compute_steering(center_points: np.ndarray, image_shape: Tuple[int, int],
                    lower_third_ratio: float = 0.66) -> Tuple[Optional[float], Optional[float]]:
    """
    Steering offset and angle calculation based on the mean x of points in the lower third.
    Mirrors the logic from predict_video.py.
    """
    h, w = image_shape
    frame_center_x = w // 2
    avg_center_x = None
    angle_deg = None

    # 초기화
    prev_angle = 0.0
    prev_center_x = None
    max_x_change = 3
    alpha = 0.05               # EMA 스무딩 계수
    delta_limit = 0.5          # 프레임 간 최대 조향각 변화 (degrees)

    if center_points.size == 0:
        return None, None

    # 하단 1/3만 고려 (하단에서 주행 판단이 유효)
    lower_third = center_points[center_points[:, 0] > h * lower_third_ratio]

    if lower_third.size == 0:
        return None, None

    if len(center_points) > 10:
        # 하단 1/3만 고려 (하단에서 주행 판단이 유효)
        lower_third = center_points[center_points[:, 0] > h * 0.66]
        if len(lower_third) > 0:
            avg_center_x = int(np.mean(lower_third[:, 1]))

            # --- 중앙선 x 변화 제한 ---
            if prev_center_x is not None:
                dx = avg_center_x - prev_center_x
                if abs(dx) > max_x_change:
                    avg_center_x = prev_center_x + np.sign(dx) * max_x_change
            prev_center_x = avg_center_x

            offset_x = avg_center_x - frame_center_x
            angle_rad = np.arctan2(offset_x, h // 2)
            angle_deg = np.degrees(angle_rad)

            # --- EMA + 변화량 제한 ---
            raw_angle = (1 - alpha) * prev_angle + alpha * angle_deg
            delta = raw_angle - prev_angle
            if abs(delta) > delta_limit:
                delta = np.sign(delta) * delta_limit
            smoothed_angle = prev_angle + delta
            prev_angle = smoothed_angle

            prev_angle = smoothed_angle

            print(f"[조향각] 원각: {angle_deg:.2f}, EMA+제한: {smoothed_angle:.2f}")
        else:
            print("중심선이 하단에서 감지되지 않음")
    else:
        print("중심선 추출 실패")

    return offset_x, smoothed_angle


def process_frame(frame: np.ndarray, model: torch.nn.Module, device: torch.device, uuid: int, input_size: Tuple[int, int] = (512, 256), mode: str = "center") -> Dict:
    """
    전체 프레임 처리 흐름
    mode: "center", "left", "right"
    """
    mode_funcs = {
        "left": extract_left_lane_mask,
        "right": extract_right_lane_mask,
        # "center": extract_center_lane_mask  # fallback only
    }

    img_tensor = preprocess_image(frame, input_size)
    pred_mask = infer_mask(model, img_tensor, device)

    if mode == "center":
        # 좌우 차선 skeleton 평균 위치
        # Adopt predict_video.py's method: merge class 1 and 2, then skeletonize
        center_lane_mask = np.logical_or(pred_mask == 1, pred_mask == 2)
        skeleton = compute_skeleton(center_lane_mask) # Reusing your compute_skeleton
        center_points = extract_skeleton_points(skeleton) # Reusing your extract_skeleton_points
        
        # Apply smoothing to the points before steering calculation
        if center_points.size == 0:
            center_points = smooth_centerline(center_points) # Uses new default sigma

        # Fallback if the primary method or smoothing resulted in no points
        if center_points.size == 0:
            # Original fallback: 중심 영역 내 클래스 1/2 마스크 (less preferred)
            fallback_mask = extract_center_lane_mask(pred_mask)
            fallback_skeleton = compute_skeleton(fallback_mask)
            center_points = extract_skeleton_points(fallback_skeleton)
    else:
        mask = mode_funcs[mode](pred_mask)
        skeleton = compute_skeleton(mask)
        # Extract points and sort by y for consistent input to compute_steering
        center_points = extract_skeleton_points(skeleton)
        if center_points.size > 0: # Ensure sorting before smoothing
            center_points = center_points[np.argsort(center_points[:, 0])]
            center_points = smooth_centerline(center_points) # Uses new default sigma


    offset, steering_angle = compute_steering(center_points, pred_mask.shape)
    skeleton_xy = remap_skeleton_coords(center_points, frame.shape, pred_mask.shape)

    return {
        "uuid": uuid,
        "offset": offset,
        "steering_angle": steering_angle,
        "skeleton_points": [list(p) for p in skeleton_xy],
        "pred_mask": pred_mask
    }
