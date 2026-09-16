"""
main.py
Code chính điều khiển xe tự hành cho UIT CAR RACING - Bảng chuyên nghiệp.

Luồng xử lý mỗi frame:
    1. Lấy ảnh camera thô từ Unity (GetRaw())
    2. Đưa ảnh qua model segmentation (đã train, export ONNX) để lấy mask
       phân biệt "đường đi được" và "không phải đường"
    3. Tính sai số (error) giữa tâm đường và tâm ảnh
    4. Đưa error qua bộ điều khiển PID để tính góc lái
    5. Điều chỉnh tốc độ theo góc lái (cua gấp thì chậm lại)
    6. Gửi lệnh điều khiển xuống Unity (AVControl)

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
from utils import PID, compute_lane_error, adaptive_speed, clip_control

# ===================== CẤU HÌNH =====================
MODEL_PATH = "lane_seg.onnx"
IMG_SIZE = 128          # phải khớp với img-size lúc train (giảm từ 224 để tăng tốc)
ROAD_CLASS = 1          # nhãn class ứng với "đường đi được"
MAX_SPEED = 35.0        # giới hạn tốc độ theo luật thi đấu
MAX_ANGLE = 25.0        # giới hạn góc lái theo luật thi đấu

# Hệ số PID - CẦN TỰ TINH CHỈNH (tune) lại trên map mẫu thực tế,
# đây chỉ là giá trị khởi điểm hợp lý.
PID_KP = 0.35
PID_KI = 0.0002
PID_KD = 0.08

# Vị trí mục tiêu trong bề rộng đường (0.5 = đi giữa, >0.5 = lệch phải).
# Thể lệ vòng sơ loại đã ghi "Đường 2 chiều" -> CẦN giữ bên phải, không đi giữa.
# 0.5 chỉ dùng tạm khi chưa xác nhận map có phải 2 chiều thật sự hay không.
# Tăng dần (0.6 -> 0.7 -> 0.75...) và quan sát trực tiếp lúc xe chạy để tinh chỉnh,
# vì đây là xấp xỉ hình học (mask chưa phân biệt vạch đứt ở giữa 2 làn).
KEEP_RIGHT_RATIO = 0.5  # TODO: đổi thành ~0.7-0.78 sau khi xác nhận + quan sát thực tế

# Tắt hiển thị debug khi chạy thi đấu chính thức -> giảm overhead, an toàn hơn cho
# giới hạn FPS 60 của máy chấm. Chỉ bật True nếu máy đang chạy CÓ giao diện đồ họa
# (không phải container headless) để xem cửa sổ camera/mask trực tiếp.
DEBUG_DISPLAY = False

# Vì container không có GUI, dùng cách này để vẫn "nhìn thấy" xe đang lái ra sao:
# lưu 1 ảnh debug (ghép ảnh gốc + mask + speed/angle) ra file mỗi N frame.
# Đặt None để tắt hẳn (không lưu gì, dùng khi thi đấu chính thức để tối đa tốc độ).
SAVE_DEBUG_EVERY = 30
DEBUG_DIR = "debug_frames"

# Vì bản headless không có phím 'q' để dừng thủ công, tự dừng sau thời gian này
# (giây) để tránh chạy vô hạn khi bạn quên tắt. Đặt None để chạy vô hạn (không tự dừng).
MAX_RUNTIME_SEC = 300

# Nếu mất dấu đường (không tìm thấy pixel đường ở dòng xét) trong bao nhiêu
# frame liên tiếp thì coi là mất làn thật sự, chuyển sang chế độ an toàn (giảm tốc mạnh)
MAX_LOST_FRAMES = 8


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

    last_valid_error = 0.0
    lost_frame_count = 0
    last_time = time.time()
    run_start_time = time.time()
    frame_counter = 0

    if SAVE_DEBUG_EVERY:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # Đo FPS trung bình để tự kiểm tra có đáp ứng giới hạn 60 FPS của máy chấm không
    fps_window = []
    FPS_LOG_EVERY = 60  # in trung bình FPS mỗi 60 frame

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
            # Chặn dt bất thường (ví dụ lúc mới khởi động, hoặc bị treo tạm thời)
            # để tránh đạo hàm PID nhảy vọt sai lệch.
            dt = min(max(dt, 1e-3), 0.2)

            state = GetStatus()
            raw_image = GetRaw()

            mask = infer_mask(session, raw_image, IMG_SIZE)
            error, row_used = compute_lane_error(
                mask, road_class=ROAD_CLASS, row_ratio=0.75,
                keep_right_ratio=KEEP_RIGHT_RATIO,
            )

            if error is None:
                # Không thấy đường ở dòng xét -> dùng lại error cũ, giảm tốc
                # để tránh đâm ra ngoài đường trong lúc "mù" tạm thời.
                lost_frame_count += 1
                error = last_valid_error
                print(f"[!] Mất dấu đường ({lost_frame_count} frame liên tiếp) - dùng error cũ: {error:.1f}")
            else:
                lost_frame_count = 0
                last_valid_error = error

            angle = pid_steer.compute(error, dt=dt)  # dùng dt ĐO THỰC TẾ, không hardcode

            if lost_frame_count >= MAX_LOST_FRAMES:
                # Mất làn quá lâu -> ưu tiên an toàn: đi chậm, giữ nguyên
                # hướng lái cuối cùng thay vì đoán liều.
                speed = MAX_SPEED * 0.25
            else:
                speed = adaptive_speed(angle, max_speed=MAX_SPEED)

            speed, angle = clip_control(speed, angle, MAX_SPEED, MAX_ANGLE)

            AVControl(speed, angle)
            frame_counter += 1

            # ---- Theo dõi FPS thực tế để đối chiếu với giới hạn 60 FPS của máy chấm ----
            frame_time = time.time() - now
            fps_window.append(frame_time)
            if len(fps_window) >= FPS_LOG_EVERY:
                avg_time = sum(fps_window) / len(fps_window)
                avg_fps = 1.0 / avg_time if avg_time > 0 else float("inf")
                status = "OK" if avg_fps >= 55 else "CẢNH BÁO - có thể không đáp ứng kịp 60 FPS"
                print(f"[FPS] Trung bình {avg_fps:.1f} FPS trong {FPS_LOG_EVERY} frame gần nhất ({status})")
                fps_window = []
            # -----------------------------------------------------------------------------

            # ---- Lưu ảnh debug định kỳ ra file (dùng khi container không có GUI) ----
            if SAVE_DEBUG_EVERY and frame_counter % SAVE_DEBUG_EVERY == 0:
                debug_img = raw_image.copy()
                cv2.putText(debug_img, f"speed={speed:.1f} angle={angle:.1f} err={error:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                mask_color = mask_to_color(mask)
                mask_color_resized = cv2.resize(
                    mask_color, (raw_image.shape[1], raw_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                combined = np.hstack([debug_img, mask_color_resized])
                cv2.imwrite(f"{DEBUG_DIR}/frame_{frame_counter:06d}.png", combined)
            # --------------------------------------------------------------------------

            # ---- Debug hiển thị trực tiếp: CHỈ dùng khi máy có GUI thật, không dùng
            # trong container headless (sẽ lỗi qt.qpa.xcb). Luôn để False khi thi
            # đấu chính thức để tối đa tốc độ. ----
            if DEBUG_DISPLAY:
                cv2.imshow("UCR 2026 - Front Camera", debug_img)
                cv2.imshow("Segmentation Mask", mask_color)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            # --------------------------------------------------------------------------

    except KeyboardInterrupt:
        print("\n[UCR 2026] Nhận Ctrl+C, dừng lại.")
    except Exception as e:
        print(f"\n[!] Lỗi (Unity có đang chạy không?): {e}")

    finally:
        print('\n[UCR 2026] Đóng kết nối...')
        CloseSocket()
        if DEBUG_DISPLAY:
            cv2.destroyAllWindows()
        print("[UCR 2026] Đã thoát.")


if __name__ == "__main__":
    main()