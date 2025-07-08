import numpy as np

# Lane Distance
def estimate_lane_distance(mask: np.ndarray, scale_factor: float) -> float | None:
    """
    정지선 클래스가 있는 mask에서 거리 추정 (픽셀 → 실제 거리(m))
    Args:
        mask (np.ndarray): segmentation 결과 마스크 (H, W), 각 픽셀은 class ID
        scale_factor (float): 픽셀 → 미터 변환 계수 (ex: 0.05 m/pixel)
    Returns:
        float | None: 정지선까지의 거리 (m), 없으면 None 반환
    """
    STOP_LINE_CLASS_ID = 4  # 예: 정지선 class가 2로 라벨링 되어 있을 경우
    # 정지선에 해당하는 모든 픽셀 좌표 (y, x)
    stopline_ys = np.where(mask == STOP_LINE_CLASS_ID)[0]
    if stopline_ys.size == 0:
        return None  # 정지선 없음
    # 화면 하단에서 가장 가까운 정지선 y좌표
    y_max = np.max(stopline_ys)
    frame_height = mask.shape[0]
    pixel_distance = frame_height - y_max
    real_distance = pixel_distance * scale_factor
    return real_distance


# Object Distance
# 2. 클래스별 색상 및 실제 크기 정의
CLASS_COLORS = {
    "car": (0, 255, 0),
    "child_protection": (255, 255, 0),
    "construction": (255, 0, 0),
    "person": (0, 0, 255),
    "speed_limit_30": (0, 165, 255),
    "speed_limit_50": (128, 0, 128),
    "stop_sign": (0, 255, 255),
    "veh_go": (255, 0, 255),
    "veh_goLeft": (102, 0, 204),
    "veh_stop": (0, 128, 255),
    "veh_warning": (255, 128, 0),
}
REAL_DIMENSIONS = {
    "car": {"w": 1.8, "h": 1.5},
    "child_protection": {"w": 0.6, "h": 0.6},
    "construction": {"w": 0.3, "h": 0.7},
    "person": {"w": 0.5, "h": 1.7},
    "speed_limit_30": {"w": 0.6, "h": 0.6},
    "speed_limit_50": {"w": 0.6, "h": 0.6},
    "stop_sign": {"w": 0.7, "h": 0.7},
    "veh_go": {"w": 0.5, "h": 0.5},
    "veh_goLeft": {"w": 0.5, "h": 0.5},
    "veh_stop": {"w": 0.5, "h": 0.5},
    "veh_warning": {"w": 0.5, "h": 0.5},
}
FOCAL_LENGTH = 1250  # 조정 가능

def estimate_obj_distance(obj_class, box):
    x1, y1, x2, y2 = box
    box_w = abs(x2 - x1)
    box_h = abs(y2 - y1)
    if obj_class not in REAL_DIMENSIONS:
        return None
    real = REAL_DIMENSIONS[obj_class]
    use_width = obj_class in {
        "car", "child_protection", "speed_limit_30",
        "speed_limit_50", "stop_sign", "veh_go",
        "veh_goLeft", "veh_stop", "veh_warning"
    }
    if use_width and box_w > 0:
        return (real["w"] * FOCAL_LENGTH) / box_w
    elif box_h > 0:
        return (real["h"] * FOCAL_LENGTH) / box_h
    return None
