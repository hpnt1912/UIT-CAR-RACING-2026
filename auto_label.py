"""
auto_label.py
Tạo nhãn (mask) SƠ BỘ tự động bằng threshold cổ điển (adaptive threshold),
để giảm công sức gán nhãn tay. Sau khi chạy script này, bạn NÊN mở từng
ảnh trong thư mục dataset/masks_auto/ và sửa lại bằng tay (dùng LabelMe/
Roboflow/CVAT) những chỗ threshold làm sai (bóng cây, cua gắt, vật cản...),
rồi lưu bản đã sửa vào dataset/masks/ để dùng huấn luyện thật.

Quy ước nhãn:
    0 = nền / không phải đường
    1 = đường đi được

Cách dùng:
    python auto_label.py
"""

import os
import glob

import cv2
import numpy as np

RAW_DIR = "dataset/raw"
AUTO_MASK_DIR = "dataset/masks_auto"


def robust_road_mask(img: np.ndarray) -> np.ndarray:
    """
    Tạo mask nhị phân (0/1) cho vùng đường, dùng adaptive threshold trên
    kênh V (Value) sau khi cân bằng histogram, giúp giảm ảnh hưởng của
    bóng cây và ánh sáng thay đổi so với threshold màu cố định.

    LƯU Ý: đây chỉ là nhãn khởi điểm để tăng tốc gán nhãn, không phải
    ground-truth hoàn hảo. Cần kiểm tra và sửa tay trước khi dùng để train.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_eq = cv2.equalizeHist(v)

    mask = cv2.adaptiveThreshold(
        v_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=-10,
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Chỉ giữ vùng đường ở nửa dưới ảnh (gần xe), bỏ nhiễu ở nửa trên
    # (bầu trời, cây cối xa) để tránh threshold bắt nhầm.
    h_img = mask.shape[0]
    mask[: int(h_img * 0.35), :] = 0

    binary_mask = (mask > 0).astype(np.uint8)  # 0 hoặc 1
    return binary_mask


def main():
    os.makedirs(AUTO_MASK_DIR, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))

    if not image_paths:
        print(f"[!] Không tìm thấy ảnh trong '{RAW_DIR}'. "
              f"Hãy chạy collect_data.py trước.")
        return

    print(f"[Auto Label] Đang xử lý {len(image_paths)} ảnh...")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        mask = robust_road_mask(img)

        # Lưu mask dưới dạng ảnh grayscale (giá trị 0 hoặc 1) để dễ dùng
        # khi load bằng Dataset trong train.py. Nhân 255 chỉ để dễ xem
        # bằng mắt khi mở file ảnh trực tiếp (không ảnh hưởng lúc train
        # vì train.py sẽ tự chuẩn hoá lại về 0/1).
        filename = os.path.basename(path)
        save_path = os.path.join(AUTO_MASK_DIR, filename)
        cv2.imwrite(save_path, mask * 255)

        if (i + 1) % 50 == 0:
            print(f"  Đã xử lý {i + 1}/{len(image_paths)} ảnh")

    print(f"[Auto Label] Xong. Nhãn sơ bộ lưu ở '{AUTO_MASK_DIR}'.")
    print("[Auto Label] Hãy mở và sửa tay các ảnh sai trước khi copy sang "
          "'dataset/masks/' để huấn luyện.")


if __name__ == "__main__":
    main()