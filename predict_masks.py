"""
predict_masks.py
Dùng model ĐÃ TRAIN (từ map cũ) để tự động dự đoán mask cho ảnh của MAP MỚI.
Đây là bước "bootstrapping" giúp đỡ phải vẽ tay từ đầu khi đổi map - model
cũ đã biết khái quát "đường trông như thế nào", chỉ cần bạn duyệt lại và
sửa tay chỗ sai (đặc biệt các đoạn bóng cây/ánh sáng khác mà model cũ
chưa từng thấy), thay vì tự vẽ 100% từ đầu.

Cách dùng:
    python predict_masks.py --raw-dir dataset/raw_map2 \\
                             --output-dir dataset/masks_map2_auto \\
                             --model lane_seg.onnx

Sau khi chạy xong:
    1. Mở lane_labeler.html, load thư mục ảnh gốc (--raw-dir) VÀ mở song song
       ảnh mask tương ứng trong --output-dir để đối chiếu (lane_labeler.html
       hiện tại vẽ từ đầu, không load sẵn mask có sẵn để sửa - xem ghi chú
       cuối file nếu muốn nâng cấp việc này).
    2. Với ảnh mask đã đúng -> copy thẳng sang dataset/masks/ (không cần vẽ lại).
    3. Với ảnh mask sai (đặc biệt vùng bóng cây, ánh sáng khác lạ) -> vẽ tay
       lại bằng lane_labeler.html như bình thường, lưu đè vào dataset/masks/.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img: np.ndarray, img_size: int) -> np.ndarray:
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = img_rgb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None, ...]
    return inp


def predict_mask(session: ort.InferenceSession, img: np.ndarray, img_size: int) -> np.ndarray:
    inp = preprocess(img, img_size)
    outputs = session.run(None, {"input": inp})
    logits = outputs[0]
    mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    return mask


def main(raw_dir: str, output_dir: str, model_path: str, img_size: int, road_class: int):
    os.makedirs(output_dir, exist_ok=True)

    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"[Predict] Đã load model '{model_path}', providers: {session.get_providers()}")

    image_paths = sorted(
        glob.glob(os.path.join(raw_dir, "*.png")) +
        glob.glob(os.path.join(raw_dir, "*.jpg"))
    )
    if not image_paths:
        print(f"[!] Không tìm thấy ảnh trong '{raw_dir}'. Kiểm tra lại đường dẫn.")
        return

    print(f"[Predict] Đang xử lý {len(image_paths)} ảnh từ '{raw_dir}'...")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        mask = predict_mask(session, img, img_size)

        # Resize mask về đúng kích thước ảnh gốc để khớp khi dùng với
        # lane_labeler.html hoặc train.py (train.py tự resize lại nên không
        # bắt buộc, nhưng giữ đúng kích thước gốc giúp xem/so sánh trực quan
        # dễ hơn khi kiểm tra bằng mắt).
        mask_resized = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        binary_mask = (mask_resized == road_class).astype(np.uint8) * 255

        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, binary_mask)

        if (i + 1) % 50 == 0:
            print(f"  Đã xử lý {i + 1}/{len(image_paths)} ảnh")

    print(f"\n[Predict] Xong. Mask dự đoán lưu ở '{output_dir}'.")
    print("[Predict] Hãy mở từng ảnh (hoặc dùng script review) để kiểm tra, "
          "sửa tay những chỗ sai bằng lane_labeler.html, rồi copy sang "
          "dataset/masks/ trước khi train.")
    print("[Predict] ƯU TIÊN kiểm tra kỹ các đoạn có bóng cây/ánh sáng khác lạ "
          "- đây là nơi model cũ dễ đoán sai nhất vì chưa từng thấy lúc train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True,
                         help="Thư mục ảnh gốc của map mới, ví dụ dataset/raw_map2")
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Thư mục lưu mask dự đoán, ví dụ dataset/masks_map2_auto")
    parser.add_argument("--model", type=str, default="lane_seg.onnx")
    parser.add_argument("--img-size", type=int, default=128,
                         help="PHẢI khớp với img-size lúc train model đang dùng")
    parser.add_argument("--road-class", type=int, default=1)
    args = parser.parse_args()

    main(args.raw_dir, args.output_dir, args.model, args.img_size, args.road_class)