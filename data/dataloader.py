"""数据加载工具：读取 test_01.jpg → BGR转RGB → 归一化[0,1] → Resize至256×704 → 输出Tensor。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# 图像相对路径
IMAGE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "captures" / "test_01.jpg"
TARGET_HEIGHT = 256
TARGET_WIDTH = 704


def load_and_preprocess(image_path: str | Path | None = None) -> torch.Tensor:
    """读取图像并完成预处理流水线。

    Args:
        image_path: 图像路径，默认使用 simulation/captures/test_01.jpg。

    Returns:
        shape 为 (3, 256, 704)、dtype 为 float32、值域 [0, 1] 的 Tensor。
    """
    path = Path(image_path) if image_path else IMAGE_PATH

    # 1. 读取图像 (OpenCV 默认 BGR)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像：{path}")

    # 2. BGR → RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3. 归一化到 [0, 1]
    rgb_f = rgb.astype(np.float32) / 255.0

    # 4. Resize 至 256×704 (cv2.resize 参数顺序为 width×height)
    resized = cv2.resize(rgb_f, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # 5. HWC → CHW 并转为 Tensor
    tensor = torch.from_numpy(resized).permute(2, 0, 1)

    return tensor


if __name__ == "__main__":
    result = load_and_preprocess()
    print(f"Tensor shape: {result.shape}")   # 期望: torch.Size([3, 256, 704])
    print(f"dtype: {result.dtype}")          # torch.float32
    print(f"min: {result.min():.4f}, max: {result.max():.4f}")  # [0, 1]
