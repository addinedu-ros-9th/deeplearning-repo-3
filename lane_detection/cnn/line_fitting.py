import cv2
import numpy as np

class LaneLineFitter:
    def __init__(self, angle_threshold_deg=5):
        self.prev_line = None
        self.angle_threshold_rad = np.radians(angle_threshold_deg)

    def fit_line(self, points):
        if len(points) < 2:
            return self.prev_line

        points = np.array(points, dtype=np.int32)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        # 벡터 아래쪽 향하게 보정
        if vy < 0:
            vx, vy, x0, y0 = -vx, -vy, -x0, -y0

        return (vx, vy, x0, y0)

    def compute_angle_diff(self, line1, line2):
        vx1, vy1, _, _ = line1
        vx2, vy2, _, _ = line2
        dot = vx1 * vx2 + vy1 * vy2
        mag1 = np.sqrt(vx1**2 + vy1**2)
        mag2 = np.sqrt(vx2**2 + vy2**2)
        cos_theta = dot / (mag1 * mag2 + 1e-6)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta)

    def update(self, skeleton_points):
        new_line = self.fit_line(skeleton_points)
        if new_line is None:
            return self.prev_line

        if self.prev_line is not None:
            angle_diff = self.compute_angle_diff(self.prev_line, new_line)
            if angle_diff < self.angle_threshold_rad:
                return self.prev_line  # 변화 작으면 이전 선 유지

        self.prev_line = new_line
        return new_line

    def get_x_at_bottom(self, line, height):
        if line is None:
            return None
        vx, vy, x0, y0 = line
        if vy == 0:
            return int(x0)
        t = (height - y0) / vy
        x = x0 + t * vx
        return int(x)
