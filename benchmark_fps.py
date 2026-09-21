"""
benchmark_fps.py
Đo tốc độ suy luận (inference) thực tế của model segmentation, để kiểm tra
TRƯỚC khi thi xem có đáp ứng giới hạn 60 FPS của máy chấm hay không.

Lý do cần script riêng: theo lưu ý từ BTC, có đội năm ngoái code chạy ổn ở
FPS cao hơn nhưng KHÔNG chạy nổi khi bị giới hạn đúng 60 FPS như máy chấm
thật. Máy dev có thể mạnh hơn máy chấm thật -> cần đo trực tiếp thời gian
xử lý mỗi frame, không chỉ dựa cảm giác "chạy mượt" khi code + Unity trên
máy cá nhân.

Cách dùng:
    python benchmark_fps.py --runs 200
"""

import argparse
import time

import numpy as np
import onnxruntime as ort

from main import MODEL_PATH, IMG_SIZE, preprocess


def benchmark(model_path: str, img_size: int, n_runs: int = 200):
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"Providers đang dùng: {session.get_providers()}")

    # Ảnh giả lập kích thước giống ảnh thật từ GetRaw() để đo preprocess + inference
    # thực tế. Thay (480, 640, 3) bằng đúng kích thước camera Unity nếu khác.
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warm-up vài lần đầu (thường chậm hơn do khởi tạo CUDA context, cache...)
    for _ in range(10):
        inp = preprocess(dummy_frame, img_size)
        session.run(None, {"input": inp})

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        inp = preprocess(dummy_frame, img_size)
        session.run(None, {"input": inp})
        times.append(time.time() - t0)

    times = np.array(times)
    avg_ms = times.mean() * 1000
    p95_ms = np.percentile(times, 95) * 1000
    max_ms = times.max() * 1000
    avg_fps = 1000 / avg_ms

    budget_ms = 1000 / 60  # ngân sách thời gian cho phép mỗi frame ở 60 FPS

    print(f"\n--- Kết quả benchmark ({n_runs} lần chạy) ---")
    print(f"Thời gian trung bình mỗi frame : {avg_ms:.2f} ms  (~{avg_fps:.1f} FPS)")
    print(f"Thời gian P95                   : {p95_ms:.2f} ms")
    print(f"Thời gian tối đa (chậm nhất)     : {max_ms:.2f} ms")
    print(f"Ngân sách thời gian ở 60 FPS     : {budget_ms:.2f} ms/frame")

    if avg_ms < budget_ms * 0.5:
        print("\n✅ RẤT AN TOÀN — model chạy nhanh hơn nhiều so với giới hạn 60 FPS, "
              "còn nhiều dư địa cho phần xử lý điều khiển khác.")
    elif avg_ms < budget_ms * 0.8:
        print("\n✅ ỔN — model đáp ứng được 60 FPS, nhưng nên theo dõi thêm khi "
              "chạy thật cùng với phần code điều khiển + socket.")
    elif avg_ms < budget_ms:
        print("\n⚠️ SÁT NGƯỠNG — model chỉ vừa đủ đáp ứng 60 FPS trên máy này. "
              "Máy chấm có thể yếu hơn hoặc có overhead khác (socket, I/O) khiến "
              "không kịp trong thực tế. NÊN tối ưu thêm (giảm img-size, dùng "
              "TensorRT/FP16, hoặc kiến trúc nhẹ hơn).")
    else:
        print("\n❌ KHÔNG ĐÁP ỨNG ĐƯỢC 60 FPS trên máy này — thời gian xử lý mỗi "
              "frame đã vượt quá ngân sách cho phép. Bắt buộc phải tối ưu trước "
              "khi thi, nếu không xe sẽ phản ứng trễ/giật khi chạy thật ở máy chấm.")

    print("\nGợi ý nếu cần tối ưu thêm:")
    print("  - Giảm --img-size lúc train (ví dụ 224 -> 160 hoặc 128)")
    print("  - Dùng onnxruntime-gpu thay vì onnxruntime (nếu máy chấm có GPU)")
    print("  - Convert sang TensorRT hoặc chạy FP16 nếu framework hỗ trợ")
    print("  - Tắt DEBUG_DISPLAY trong main.py khi chạy thi thật (bỏ cv2.imshow)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--runs", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.model, args.img_size, args.runs)