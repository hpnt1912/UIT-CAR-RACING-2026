"""
model.py
Kiến trúc U-Net thu gọn (Small U-Net) cho bài toán segmentation 2 lớp
(nền / đường đi được). Thiết kế nhẹ để đảm bảo tốc độ inference real-time
(giới hạn FPS 60 theo thể lệ).
"""

import torch
import torch.nn as nn


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """
    U-Net rút gọn: 3 tầng encoder/decoder, số kênh nhỏ (16/32/64).
    Input:  (B, 3, H, W)  - ảnh RGB
    Output: (B, n_classes, H, W) - logits cho từng class mỗi pixel
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)  # 64 = 32 (up) + 32 (skip)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)  # 32 = 16 (up) + 16 (skip)

        self.out_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)              # (B,16,H,W)
        e2 = self.enc2(self.pool(e1))  # (B,32,H/2,W/2)
        e3 = self.enc3(self.pool(e2))  # (B,64,H/4,W/4)

        d2 = self.up2(e3)                          # (B,32,H/2,W/2)
        d2 = torch.cat([d2, e2], dim=1)             # (B,64,H/2,W/2)
        d2 = self.dec2(d2)                          # (B,32,H/2,W/2)

        d1 = self.up1(d2)                           # (B,16,H,W)
        d1 = torch.cat([d1, e1], dim=1)              # (B,32,H,W)
        d1 = self.dec1(d1)                           # (B,16,H,W)

        return self.out_conv(d1)  # (B, n_classes, H, W)


if __name__ == "__main__":
    # Kiểm tra nhanh kích thước input/output và tốc độ forward pass
    import time

    model = SmallUNet(n_classes=2)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        t0 = time.time()
        out = model(dummy)
        t1 = time.time()

    print("Output shape:", out.shape)
    print(f"Forward pass time (CPU): {(t1 - t0) * 1000:.2f} ms")
    print("Số tham số:", sum(p.numel() for p in model.parameters()))