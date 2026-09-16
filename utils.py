"""
utils.py
Chứa các hàm/class dùng chung: PID controller, tính lane center từ mask,
điều chỉnh tốc độ thích ứng theo góc lái.
"""

import numpy as np


class PID:
    """Bộ điều khiển PID đơn giản dùng để tính góc lái từ sai số (error)."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error: float, dt: float = 1 / 30) -> float:
        self.integral += error * dt
        # Chống windup: giới hạn tích lũy integral
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


def compute_lane_error(mask: np.ndarray, road_class: int = 1, row_ratio: float = 0.75,
                        keep_right_ratio: float = 0.5):
    """
    Tính sai số (error) giữa vị trí MỤC TIÊU trên đường và tâm ảnh, dựa trên mask segmentation.

    mask: ảnh 2D (H, W) với giá trị mỗi pixel là class (0 = nền, 1 = đường, ...)
    road_class: giá trị class ứng với "đường đi được" (gộp cả 2 chiều/2 làn nếu có)
    row_ratio: xét dòng ảnh ở vị trí này (0.0 = trên cùng, 1.0 = dưới cùng)

    keep_right_ratio: vị trí mục tiêu trong bề rộng đường, tính từ mép TRÁI sang mép PHẢI:
        0.5  -> đi giữa toàn bộ bề rộng đường (dùng cho đường 1 chiều / không cần giữ làn)
        0.75 -> đi lệch về nửa bên phải (gần đúng cho đường 2 chiều, giữ bên phải vạch giữa,
                giả định 2 làn rộng bằng nhau: tâm làn phải nằm ở khoảng 75% bề rộng từ mép trái)
        Cần tinh chỉnh giá trị này bằng cách quan sát trực tiếp khi xe chạy map 2 chiều,
        vì mask hiện tại gộp chung cả 2 làn (chưa phân biệt vạch đứt ở giữa) - đây chỉ là
        xấp xỉ hình học, không phải phát hiện vạch tim đường thật sự.

    Trả về: (error, row_used)
        error > 0: mục tiêu lệch về bên phải tâm ảnh -> cần lái phải
        error < 0: mục tiêu lệch về bên trái tâm ảnh -> cần lái trái
        Nếu không tìm thấy đường ở dòng đó, trả về None để main.py xử lý fallback.
    """
    h, w = mask.shape[:2]
    row = int(h * row_ratio)
    row = min(max(row, 0), h - 1)

    line = mask[row, :]
    road_pixels = np.where(line == road_class)[0]

    if len(road_pixels) == 0:
        return None, row

    left_edge = road_pixels.min()
    right_edge = road_pixels.max()

    # Vị trí mục tiêu trong bề rộng đường, thay vì luôn lấy đúng tâm (0.5)
    target_x = left_edge + (right_edge - left_edge) * keep_right_ratio

    img_center = w / 2.0
    error = target_x - img_center
    return float(error), row


def adaptive_speed(angle: float, max_speed: float = 90.0, min_speed_ratio: float = 0.35) -> float:
    """
    Giảm tốc khi cua gấp, tăng tốc khi đường thẳng.
    angle: góc lái hiện tại (độ), giả sử trong khoảng [-25, 25]
    """
    angle_abs = abs(angle)
    if angle_abs > 15:
        ratio = min_speed_ratio
    elif angle_abs > 8:
        ratio = 0.6
    else:
        ratio = 0.9
    return max_speed * ratio


def clip_control(speed: float, angle: float, max_speed: float = 90.0, max_angle: float = 25.0):
    """Đảm bảo lệnh điều khiển không vượt giới hạn cho phép của cuộc thi."""
    speed = float(np.clip(speed, 0.0, max_speed))
    angle = float(np.clip(angle, -max_angle, max_angle))
    return speed, angle