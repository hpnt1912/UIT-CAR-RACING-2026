"""
train.py
Huấn luyện SmallUNet trên dataset đã gán nhãn.

Cấu trúc thư mục dữ liệu mong đợi:
    dataset/
        raw/     -> ảnh gốc (.png), ví dụ frame_123.png
        masks/   -> mask tương ứng CÙNG TÊN FILE, giá trị pixel 0 hoặc 255
                    (0 = nền, 255 = đường). Đây là mask ĐÃ ĐƯỢC KIỂM TRA/
                    SỬA TAY sau bước auto_label.py, không dùng thẳng
                    masks_auto/.

Cách dùng:
    python train.py --epochs 40 --batch-size 8 --img-size 224
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model import SmallUNet


class LaneSegDataset(Dataset):
    def __init__(self, raw_dir: str, mask_dir: str, img_size: int = 224, augment: bool = True):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.filenames = sorted(
            f for f in os.listdir(raw_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        )
        if not self.filenames:
            raise RuntimeError(
                f"Không tìm thấy cặp ảnh/mask cùng tên trong "
                f"'{raw_dir}' và '{mask_dir}'. Kiểm tra lại dữ liệu."
            )

        # Augment màu sắc/ánh sáng để model chịu được bóng cây, ánh sáng
        # thay đổi (yêu cầu quan trọng theo thể lệ cuộc thi).
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = cv2.imread(os.path.join(self.raw_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Chuẩn hoá mask về nhãn class 0/1
        mask = (mask > 127).astype(np.int64)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.augment:
            img_tensor = self.color_jitter(img_tensor)
            # Lật ngang ngẫu nhiên (đường vẫn hợp lệ khi lật)
            if np.random.rand() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                mask = np.fliplr(mask).copy()

        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss cho bài toán 2 class, giúp segmentation bám biên tốt hơn CE thuần."""
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # xác suất class "đường"
    target_f = (target == 1).float()

    intersection = (probs * target_f).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Sử dụng thiết bị: {device}")

    dataset = LaneSegDataset(
        raw_dir=os.path.join(args.data_dir, "raw"),
        mask_dir=os.path.join(args.data_dir, "masks"),
        img_size=args.img_size,
        augment=True,
    )

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SmallUNet(n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = ce_loss_fn(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = ce_loss_fn(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"  -> Đã lưu model tốt nhất vào '{args.output}' (val_loss={val_loss:.4f})")

    print("[Train] Hoàn tất huấn luyện.")

    # Xuất sang ONNX để dùng inference nhanh trong main.py
    export_onnx(model, args.img_size, args.onnx_output, device)


def export_onnx(model, img_size, onnx_path, device):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=12,
        dynamic_axes=None,
    )
    print(f"[Train] Đã export model sang ONNX: '{onnx_path}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="lane_seg_best.pt")
    parser.add_argument("--onnx-output", type=str, default="lane_seg.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())