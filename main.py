"""
main.py
Code chính điều khiển xe tự hành cho UIT CAR RACING - Bảng chuyên nghiệp.
Xe đi GIỮA bề rộng đường phát hiện được (không bám lề phải/trái).

Luồng xử lý mỗi frame:
    1. Lấy ảnh camera thô từ Unity (GetRaw())
    2. Đưa ảnh qua model segmentation (đã train, export ONNX) để lấy mask
       phân biệt "đường đi được" và "không phải đường"
    3. Tính sai số (error) giữa tâm đường và tâm ảnh
    4. Làm mượt error, đưa qua bộ điều khiển PID để tính góc lái
    5. Giới hạn tốc độ thay đổi góc lái mỗi frame (tránh giật đột ngột)
    6. Điều chỉnh tốc độ theo góc lái (cua gấp thì chậm lại)
    7. Gửi lệnh điều khiển xuống Unity (AVControl)

Yêu cầu trước khi chạy:
    - Đã huấn luyện model bằng train.py và có file 'lane_seg.onnx'
    - Đã cài: onnxruntime (hoặc onnxruntime-gpu nếu có CUDA)
    - Unity mô phỏng đang chạy sẵn
"""

import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
from utils import PID, compute_lane_error, adaptive_speed, clip_control, ErrorSmoother, rate_limit_angle
from live_view import LiveViewer

# ===================== CẤU HÌNH =====================
MODEL_PATH = "lane_seg.onnx"
IMG_SIZE = 128           # phải khớp với img-size lúc train
ROAD_CLASS = 1           # nhãn class ứng với "đường đi được"
MAX_SPEED = 28.0         # giới hạn tốc độ THỰC TẾ dùng lúc test (luật cho phép tối đa 90)
MAX_ANGLE = 25.0         # giới hạn góc lái theo luật thi đấu

# Hệ số PID - CẦN TỰ TINH CHỈNH (tune) lại trên map mẫu thực tế.
PID_KP = 0.30 # toc do phan ung
PID_KI = 0.0002 # triet tieu sai so
PID_KD = 0.10

# Nếu mất dấu đường (không tìm thấy pixel đường ở dòng xét) trong bao nhiêu
# frame liên tiếp thì coi là mất làn thật sự, chuyển sang chế độ an toàn (giảm tốc mạnh)
MAX_LOST_FRAMES = 8

# Làm mượt tín hiệu error để giảm ảnh hưởng nhiễu tức thời từ model (model mới
# train từ ít ảnh dễ "nhìn nhầm" 1-2 frame, khiến biên đường tính sai đột ngột).
ERROR_SMOOTHING_ALPHA = 0.4

# Giới hạn góc lái được phép thay đổi tối đa giữa 2 frame liên tiếp (độ/frame).
# Lớp bảo vệ cuối: dù error có nhảy vọt bất thường vì lý do gì, góc lái thực
# tế gửi xuống xe vẫn không thể "giật" đột ngột.
MAX_ANGLE_DELTA_PER_FRAME = 6.0

# Xem trực tiếp (gần real-time) ảnh camera + mask qua trình duyệt, không cần
# GUI trong container. Mở http://localhost:<LIVE_VIEW_PORT> trên máy host,
# hoặc dùng tab "Ports" trong VS Code nếu đang chạy Dev Container.
LIVE_VIEW_ENABLED = True
LIVE_VIEW_PORT = 8080
LIVE_VIEW_EVERY = 3  # cập nhật ảnh live view mỗi N frame (nhỏ hơn SAVE_DEBUG_EVERY để mượt hơn)

# Vì container không có GUI, dùng cách này để vẫn lưu lại lịch sử: lưu 1 ảnh
# debug (ghép ảnh gốc + mask + speed/angle) ra file mỗi N frame.
# Đặt None để tắt hẳn (không lưu gì, dùng khi thi đấu chính thức để tối đa tốc độ).
SAVE_DEBUG_EVERY = 30
DEBUG_DIR = "debug_frames"

# Vì bản headless không có phím 'q' để dừng thủ công, tự dừng sau thời gian này
# (giây) để tránh chạy vô hạn khi bạn quên tắt. Đặt None để chạy vô hạn (không tự dừng).
MAX_RUNTIME_SEC = 300

# Tắt hiển thị debug khi chạy thi đấu chính thức -> giảm overhead, an toàn hơn cho
# giới hạn FPS 60 của máy chấm. Chỉ bật True nếu máy đang chạy CÓ giao diện đồ họa
# (không phải container headless) để xem cửa sổ camera/mask trực tiếp.
DEBUG_DISPLAY = False
# =====================================================


def load_model(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[Main] Đã load model '{model_path}' với providers: {session.get_providers()}")
    return session


def preprocess(img: np.ndarray, img_size: int) -> np.ndarray:
    """Resize + chuẩn hoá ảnh RGB về định dạng input của model (NCHW, float32, 0-1)."""
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = img_rgb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return inp


def infer_mask(session: ort.InferenceSession, img: np.ndarray, img_size: int) -> np.ndarray:
    """Chạy model segmentation, trả về mask class-index kích thước (img_size, img_size)."""
    inp = preprocess(img, img_size)
    outputs = session.run(None, {"input": inp})
    logits = outputs[0]  # (1, n_classes, H, W)
    mask = np.argmax(logits[0], axis=0).astype(np.uint8)  # (H, W)
    return mask


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Chuyển mask class-index sang ảnh màu để debug/hiển thị bằng imshow."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == ROAD_CLASS] = (0, 0, 255)   # đường: đỏ
    color[mask != ROAD_CLASS] = (40, 40, 40)  # nền: xám đậm
    return color


def main():
    session = load_model(MODEL_PATH)
    pid_steer = PID(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    error_smoother = ErrorSmoother(alpha=ERROR_SMOOTHING_ALPHA)

    last_valid_error = 0.0
    lost_frame_count = 0
    last_time = time.time()
    run_start_time = time.time()
    frame_counter = 0
    last_angle = 0.0

    # Tần suất in thông tin terminal
    PRINT_EVERY = 5

    fps_window = []
    FPS_LOG_EVERY = 60

    if SAVE_DEBUG_EVERY:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    live_viewer = None
    if LIVE_VIEW_ENABLED:
        live_viewer = LiveViewer(port=LIVE_VIEW_PORT)
        live_viewer.start()

    print("[UCR 2026] Bắt đầu điều khiển xe.")
    if DEBUG_DISPLAY:
        print("[UCR 2026] Nhấn 'q' trên cửa sổ camera để dừng.")
    else:
        print("[UCR 2026] Chế độ headless - nhấn Ctrl+C để dừng"
              + (f", hoặc tự dừng sau {MAX_RUNTIME_SEC}s." if MAX_RUNTIME_SEC else " (chạy vô hạn)."))
        if SAVE_DEBUG_EVERY:
            print(f"[UCR 2026] Ảnh debug sẽ lưu vào '{DEBUG_DIR}/' mỗi {SAVE_DEBUG_EVERY} frame.")

    try:
        while True:
            if MAX_RUNTIME_SEC and (time.time() - run_start_time) >= MAX_RUNTIME_SEC:
                print(f"[UCR 2026] Đã chạy đủ {MAX_RUNTIME_SEC}s, tự dừng.")
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            dt = min(max(dt, 1e-3), 0.2)

            state = GetStatus()
            raw_image = GetRaw()

            mask = infer_mask(session, raw_image, IMG_SIZE)
            error, row_used, lane_debug = compute_lane_error(
                mask, road_class=ROAD_CLASS, row_ratio=0.75,
            )

            if error is None:
                lost_frame_count += 1
                error = last_valid_error
            else:
                lost_frame_count = 0
                last_valid_error = error

            smoothed_error = error_smoother.update(error)
            raw_angle = pid_steer.compute(smoothed_error, dt=dt)
            angle = rate_limit_angle(raw_angle, last_angle, MAX_ANGLE_DELTA_PER_FRAME)
            last_angle = angle

            if lost_frame_count >= MAX_LOST_FRAMES:
                speed = MAX_SPEED * 0.25
            else:
                speed = adaptive_speed(angle, max_speed=MAX_SPEED)

            speed, angle = clip_control(speed, angle, MAX_SPEED, MAX_ANGLE)

            AVControl(speed, angle)
            frame_counter += 1

            # ---- Hiển thị thông số điều khiển ----
            if frame_counter % PRINT_EVERY == 0:
                print(
                    f"Frame {frame_counter:5d} | "
                    f"Speed: {speed:5.1f} | "
                    f"Error: {error:6.1f} | "
                    f"Angle: {angle:6.1f} | "
                    f"Lost: {lost_frame_count}"
                )

            # ---- Theo dõi FPS thực tế ----
            frame_time = time.time() - now
            fps_window.append(frame_time)
            if len(fps_window) >= FPS_LOG_EVERY:
                avg_time = sum(fps_window) / len(fps_window)
                avg_fps = 1.0 / avg_time if avg_time > 0 else float("inf")
                status = "OK" if avg_fps >= 55 else "CẢNH BÁO - có thể không đáp ứng kịp 60 FPS"
                print(f"[FPS] Trung bình {avg_fps:.1f} FPS trong {FPS_LOG_EVERY} frame gần nhất ({status})")
                fps_window = []

            # ---- Tính overlay debug cho live view / lưu file ----
            need_overlay = (
                (LIVE_VIEW_ENABLED and frame_counter % LIVE_VIEW_EVERY == 0) or
                (SAVE_DEBUG_EVERY and frame_counter % SAVE_DEBUG_EVERY == 0)
            )
            if need_overlay:
                debug_img = raw_image.copy()
                h_raw, w_raw = debug_img.shape[:2]

                if lane_debug is not None:
                    scale_x = w_raw / IMG_SIZE
                    scale_y = h_raw / IMG_SIZE
                    y_line = int(row_used * scale_y)
                    left_x = int(lane_debug["left"] * scale_x)
                    right_x = int(lane_debug["right"] * scale_x)
                    target_x_raw = int(lane_debug["target"] * scale_x)

                    cv2.line(debug_img, (left_x, y_line - 5), (left_x, y_line + 5), (255, 200, 0), 2)
                    cv2.line(debug_img, (right_x, y_line - 5), (right_x, y_line + 5), (255, 200, 0), 2)
                    cv2.line(debug_img, (w_raw // 2, 0), (w_raw // 2, h_raw), (0, 255, 255), 1)
                    cv2.circle(debug_img, (target_x_raw, y_line), 6, (0, 0, 255), -1)

                cv2.putText(debug_img, f"speed={speed:.1f} angle={angle:.1f} err={error:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                mask_color = mask_to_color(mask)
                mask_color_resized = cv2.resize(
                    mask_color, (raw_image.shape[1], raw_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                combined = np.hstack([debug_img, mask_color_resized])

                if live_viewer is not None and frame_counter % LIVE_VIEW_EVERY == 0:
                    live_viewer.update(combined)

                if SAVE_DEBUG_EVERY and frame_counter % SAVE_DEBUG_EVERY == 0:
                    cv2.imwrite(f"{DEBUG_DIR}/frame_{frame_counter:06d}.png", combined)

            if DEBUG_DISPLAY:
                cv2.imshow("UCR 2026 - Front Camera", debug_img)
                cv2.imshow("Segmentation Mask", mask_color)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[UCR 2026] Nhận Ctrl+C, dừng lại.")
    except Exception as e:
        print(f"\n[!] Lỗi (Unity có đang chạy không?): {e}")

    finally:
        print('\n[UCR 2026] Đóng kết nối...')
        CloseSocket()
        if live_viewer is not None:
            live_viewer.stop()
        if DEBUG_DISPLAY:
            cv2.destroyAllWindows()
        print("[UCR 2026] Đã thoát.")


if __name__ == "__main__":
    main()