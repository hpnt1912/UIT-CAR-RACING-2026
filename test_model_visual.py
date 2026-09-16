"""
test_model_visual.py
Chạy model segmentation trên vài frame THẬT lấy trực tiếp từ Unity, lưu kết
quả (ảnh gốc + mask dự đoán ghép cạnh nhau) ra file PNG để xem bằng mắt,
vì container không có GUI để dùng cv2.imshow trực tiếp.

Cách dùng:
    - Mở Unity, cho xe chạy (chạy tay hoặc để AVControl tốc độ cố định)
    - Chạy: python test_model_visual.py --num-frames 10
    - Mở các file trong thư mục 'test_results/' bằng VS Code Explorer để xem
      model dự đoán mask có đúng vùng đường không.
"""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
from main import MODEL_PATH, IMG_SIZE, ROAD_CLASS, infer_mask, mask_to_color

OUTPUT_DIR = "test_results"


def main(num_frames: int, speed: float):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = ort.InferenceSession(
        MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"[Test] Đã load model, providers: {session.get_providers()}")
    print(f"[Test] Sẽ chụp và lưu {num_frames} frame vào '{OUTPUT_DIR}/'")

    try:
        for i in range(num_frames):
            raw_image = GetRaw()
            mask = infer_mask(session, raw_image, IMG_SIZE)
            mask_color = mask_to_color(mask)

            # Resize mask_color về đúng kích thước ảnh gốc để ghép cạnh nhau dễ so sánh
            mask_color_resized = cv2.resize(
                mask_color, (raw_image.shape[1], raw_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            combined = np.hstack([raw_image, mask_color_resized])

            out_path = os.path.join(OUTPUT_DIR, f"test_{i:03d}.png")
            cv2.imwrite(out_path, combined)
            print(f"  Đã lưu {out_path}")

            # Giữ xe chạy nhẹ trong lúc chụp để có nhiều tình huống khác nhau
            AVControl(speed, 0.0)

    except Exception as e:
        print(f"\n[!] Lỗi (Unity có đang chạy không?): {e}")

    finally:
        CloseSocket()
        print(f"\n[Test] Hoàn tất. Mở các file trong '{OUTPUT_DIR}/' để xem "
              f"(bên trái = ảnh gốc, bên phải = mask model dự đoán).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--speed", type=float, default=10.0)
    args = parser.parse_args()
    main(args.num_frames, args.speed)