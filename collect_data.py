"""
collect_data.py
Script chạy thử map mẫu và tự động lưu ảnh từ camera (GetRaw()) để làm
dữ liệu huấn luyện model segmentation.

PHIÊN BẢN HEADLESS: không dùng cv2.imshow()/cv2.waitKey() vì môi trường
container thường không có màn hình đồ họa (không có Xserver), gọi các
hàm này sẽ bị lỗi "could not connect to display" / "Aborted (core dumped)".

Cách dùng:
    - Chạy Unity mô phỏng trước.
    - Chạy: python collect_data.py
    - Script tự dừng khi đủ MAX_FRAMES ảnh hoặc đủ MAX_DURATION_SEC giây,
      tùy điều kiện nào đến trước. Có thể dừng sớm bằng Ctrl+C trong terminal.
    - Cố gắng để xe đi qua đủ các tình huống: khúc cua, đoạn có bóng cây,
      đoạn sáng/tối khác nhau, ngã ba/ngã tư (xem thêm ghi chú ở cuối file
      nếu map mẫu có cua mà xe đi thẳng cố định sẽ bị văng ra ngoài).

Ảnh sẽ được lưu vào thư mục dataset/raw/, mỗi vài frame lưu 1 ảnh để
tránh dữ liệu bị trùng lặp quá nhiều.
"""

import argparse
import os
import time

import cv2

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket

DEFAULT_SAVE_DIR = "dataset/raw"
SAVE_EVERY_N_FRAMES = 3      # chỉ lưu 1 trong mỗi N frame
DEFAULT_MAX_FRAMES = 5000    # tự dừng sau khi lưu đủ số ảnh này
DEFAULT_MAX_DURATION_SEC = 600  # hoặc tự dừng sau 10 phút, tùy điều kiện nào tới trước
PRINT_EVERY_N_SAVED = 50     # in tiến độ mỗi N ảnh đã lưu, để theo dõi qua log

# Tốc độ/góc lái mặc định khi thu thập dữ liệu (chỉnh tay nếu cần lái thủ công)
DEFAULT_SPEED = 15.0
DEFAULT_ANGLE = 0.0


def main(save_dir: str, max_frames: int, max_duration_sec: int):
    os.makedirs(save_dir, exist_ok=True)
    frame_id = 0
    saved_count = 0
    start_time = time.time()

    print(f"[Collect] Bắt đầu thu thập dữ liệu vào '{save_dir}' (chế độ headless).")
    print(f"[Collect] Sẽ tự dừng sau {max_frames} ảnh lưu được hoặc {max_duration_sec}s, "
          f"tùy điều kiện nào đến trước. Nhấn Ctrl+C để dừng sớm.")

    try:
        while True:
            state = GetStatus()
            img = GetRaw()

            if frame_id % SAVE_EVERY_N_FRAMES == 0:
                filename = f"{save_dir}/frame_{int(time.time() * 1000)}.png"
                cv2.imwrite(filename, img)
                saved_count += 1

                if saved_count % PRINT_EVERY_N_SAVED == 0:
                    elapsed = time.time() - start_time
                    print(f"[Collect] Đã lưu {saved_count} ảnh ({elapsed:.0f}s trôi qua)...")

            frame_id += 1

            # TODO: nếu muốn lái tay để chủ động đi qua nhiều tình huống
            # khác nhau (thay vì đi thẳng tốc độ cố định), cần một cách
            # nhập lệnh không phụ thuộc cửa sổ đồ họa, ví dụ đọc phím qua
            # terminal (module `keyboard`/`pynput`) hoặc gán sẵn 1 kịch bản
            # (speed, angle) thay đổi theo thời gian để mô phỏng lái lượn.
            AVControl(DEFAULT_SPEED, DEFAULT_ANGLE)

            if saved_count >= max_frames:
                print(f"[Collect] Đã đạt {max_frames} ảnh, dừng lại.")
                break
            if time.time() - start_time >= max_duration_sec:
                print(f"[Collect] Đã chạy đủ {max_duration_sec}s, dừng lại.")
                break

    except KeyboardInterrupt:
        print("\n[Collect] Nhận Ctrl+C, dừng thu thập.")
    except Exception as e:
        print(f"\n[!] Lỗi (Unity có đang chạy không?): {e}")

    finally:
        print(f"\n[Collect] Đã lưu {saved_count} ảnh vào '{save_dir}'.")
        print("[Collect] Đóng kết nối...")
        CloseSocket()
        print("[Collect] Hoàn tất.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                         help="Thư mục lưu ảnh, ví dụ dataset/raw_map2 khi thu thập map mới")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-duration-sec", type=int, default=DEFAULT_MAX_DURATION_SEC)
    args = parser.parse_args()
    main(args.save_dir, args.max_frames, args.max_duration_sec)