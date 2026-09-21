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


def compute_lane_error(mask: np.ndarray, road_class: int = 1, row_ratio: float = 0.75):
    """
    Tính sai số (error) giữa TÂM đường và tâm ảnh, dựa trên mask segmentation.
    Xe sẽ cố đi GIỮA bề rộng đường phát hiện được (không bám lề phải/trái).

    mask: ảnh 2D (H, W) với giá trị mỗi pixel là class (0 = nền, 1 = đường, ...)
    road_class: giá trị class ứng với "đường đi được"
    row_ratio: xét dòng ảnh ở vị trí này (0.0 = trên cùng, 1.0 = dưới cùng).
               Nên lấy gần đáy ảnh (gần xe) để phản ứng nhanh, nhưng không
               quá sát đáy để tránh nhiễu do nắp capo/thân xe che khuất.

    Trả về: (error, row_used, debug_info)
        error > 0: đường lệch về bên phải tâm ảnh -> cần lái phải
        error < 0: đường lệch về bên trái tâm ảnh -> cần lái trái
        debug_info: {"left", "right", "target"} (toạ độ pixel trong không gian
            mask) - dùng để vẽ trực quan lên ảnh debug/live view.
        Nếu không tìm thấy đường ở dòng đó, trả về (None, row, None) để
        main.py xử lý fallback.
    """
    h, w = mask.shape[:2]
    row = int(h * row_ratio)
    row = min(max(row, 0), h - 1)

    line = mask[row, :]
    road_pixels = np.where(line == road_class)[0]

    if len(road_pixels) == 0:
        return None, row, None

    left_edge = road_pixels.min()
    right_edge = road_pixels.max()
    target_x = (left_edge + right_edge) / 2.0  # luôn đi giữa bề rộng đường

    img_center = w / 2.0
    error = target_x - img_center
    debug_info = {"left": int(left_edge), "right": int(right_edge), "target": float(target_x)}
    return float(error), row, debug_info


class ErrorSmoother:
    """
    Làm mượt tín hiệu error bằng exponential moving average (EMA), giúp giảm
    ảnh hưởng của các frame bị nhiễu tức thời (model segmentation "nhìn nhầm"
    1-2 frame khiến biên đường bị tính sai đột ngột) - nguyên nhân phổ biến
    gây hiện tượng xe đang đi thẳng bỗng rẽ gắt bất thường.
    """

    def __init__(self, alpha: float = 0.4):
        # alpha càng nhỏ càng mượt (ít nhạy với thay đổi tức thời) nhưng phản
        # ứng chậm hơn với lệch làn thật; alpha càng lớn càng nhạy nhưng dễ
        # bị nhiễu ảnh hưởng. 0.3-0.5 là khoảng hợp lý để bắt đầu.
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def rate_limit_angle(new_angle: float, prev_angle: float, max_delta: float) -> float:
    """
    Giới hạn mức thay đổi góc lái tối đa giữa 2 frame liên tiếp, để dù error
    tính ra có nhảy vọt bất thường (do model nhiễu 1 frame), góc lái thực tế
    gửi xuống xe vẫn thay đổi từ từ, không bị "giật" đột ngột gây chạm vạch.
    """
    delta = new_angle - prev_angle
    delta = np.clip(delta, -max_delta, max_delta)
    return prev_angle + delta


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