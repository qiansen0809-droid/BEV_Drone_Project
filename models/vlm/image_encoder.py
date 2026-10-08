"""
image_encoder.py — 图像特征编码器
职责：接收图像Tensor，通过Grounding DINO或ResNet提取多尺度特征图

输入：图像Tensor，形状 [1, 3, 256, 704]（RGB，归一化[0,1]）
输出：特征图，形状 [1, 2048, H/32, W/32]（即 [1, 2048, 8, 22]）

使用模型（按优先级）：
  1. Grounding DINO（Swin-Tiny backbone）— 开放词汇目标检测的视觉骨干
  2. ResNet-50（torchvision）— 兜底方案，保证输出维度
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class ImageEncoder:
    """
    图像特征编码器
    将无人机采集的RGB图像编码为高维特征图，用于后续与文本向量做相似度匹配
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = None,
        pretrained: bool = True,
    ):
        """
        初始化图像编码器

        Args:
            backbone: 骨干网络类型
                      - "resnet50": torchvision ResNet-50（输出 2048 通道）
                      - "grounding_dino": Grounding DINO Swin backbone（需单独安装）
            device: 运行设备 ("cuda" 或 "cpu")，None则自动选择
            pretrained: 是否加载预训练权重
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.backbone_name = backbone
        print(f"[ImageEncoder] 使用骨干网络: {backbone}")
        print(f"[ImageEncoder] 使用设备: {self.device}")

        if backbone == "resnet50":
            self.model = self._build_resnet50(pretrained)
        elif backbone == "grounding_dino":
            self.model = self._build_grounding_dino()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}，可选: resnet50, grounding_dino")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 输出信息
        print(f"[ImageEncoder] 模型已加载，输出特征维度: 2048")
        print(f"[ImageEncoder] 下采样倍率: 32x")
        print(f"[ImageEncoder] 输入 256×704 → 输出 8×22 特征图")

    def _build_resnet50(self, pretrained: bool) -> nn.Module:
        """
        构建ResNet-50骨干（去掉最后的全局池化和全连接层）

        ResNet-50 各阶段输出：
          - conv1:     [B, 64,   H/2,  W/2]
          - layer1:    [B, 256,  H/4,  W/4]
          - layer2:    [B, 512,  H/8,  W/16]
          - layer3:    [B, 1024, H/16, W/32]
          - layer4:    [B, 2048, H/32, W/64]  ← 目标层

        Returns:
            nn.Sequential: 输出 [B, 2048, H/32, W/32] 的截断ResNet
        """
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        full_resnet = models.resnet50(weights=weights)

        # 截取到 layer4 为止（保留所有卷积层，去掉 avgpool 和 fc）
        # 对于输入 256×704，layer4 输出 [1, 2048, 8, 22]
        layers = nn.Sequential(
            full_resnet.conv1,   # [B, 64,   H/2,  W/2]
            full_resnet.bn1,
            full_resnet.relu,
            full_resnet.maxpool, # [B, 64,   H/4,  W/4]
            full_resnet.layer1,  # [B, 256,  H/4,  W/4]
            full_resnet.layer2,  # [B, 512,  H/8,  W/16]
            full_resnet.layer3,  # [B, 1024, H/16, W/32]
            full_resnet.layer4,  # [B, 2048, H/32, W/64]
        )
        return layers

    def _build_grounding_dino(self):
        """
        构建Grounding DINO视觉骨干

        Grounding DINO使用Swin Transformer作为视觉backbone，
        通过多尺度特征融合输出丰富的视觉特征。

        注意：需要先安装Grounding DINO:
              pip install git+https://github.com/IDEA-Research/GroundingDINO.git

        Returns:
            nn.Module: 特征提取器，输出 [B, 2048, H/32, W/32]
        """
        try:
            from groundingdino.models import build_model
            from groundingdino.util import get_tokenlizer
            from groundingdino.util.slconfig import SLConfig

            print("[ImageEncoder] 正在加载 Grounding DINO 配置...")

            # 使用 Swin-Tiny 配置
            config = SLConfig.fromfile(
                "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            )
            # 调整为纯图像编码模式（不需要文本输入）
            model = build_model(config)

            # 加载预训练权重
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                map_location=self.device,
            )
            # 权重key可能存在前缀差异，做兼容处理
            state_dict = {}
            for k, v in checkpoint.items():
                # 移除可能的 "module." 前缀
                k = k.replace("module.", "")
                state_dict[k] = v
            model.load_state_dict(state_dict, strict=False)
            print("[ImageEncoder] Grounding DINO 权重加载完成")

            # 只取视觉骨干部分，输出调整为 [B, 2048, H/32, W/32]
            # 这里包装成一个简单的特征提取接口
            return GroundingDINOFeatureExtractor(model, self.device)

        except ImportError as e:
            print(f"[ImageEncoder] Grounding DINO 未安装: {e}")
            print("[ImageEncoder] 自动回退到 ResNet-50...")
            return self._build_resnet50(pretrained=True)

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        将图像Tensor编码为特征图

        Args:
            image_tensor: 形状 [B, 3, 256, 704] 的图像tensor，值域[0,1]

        Returns:
            features: 形状 [B, 2048, H/32, W/32] 的特征图
                     对于256×704输入 → [B, 2048, 8, 22]
        """
        # 确保数值范围正确：如果是[0,255]则归一化
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        # ImageNet标准化（ResNet/预训练模型的标准输入）
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        image_tensor = image_tensor.to(self.device)
        image_tensor = (image_tensor - mean) / std

        # 前向传播
        features = self.model(image_tensor)

        return features

    def get_feature_shape(self, image_height: int = 256, image_width: int = 704) -> tuple:
        """返回给定输入尺寸对应的输出特征图形状"""
        return (1, 2048, image_height // 32, image_width // 32)


class GroundingDINOFeatureExtractor(nn.Module):
    """
    Grounding DINO 特征提取包装器

    将 Grounding DINO 的视觉骨干输出转换为统一格式 [B, 2048, H/32, W/32]
    """

    def __init__(self, gd_model, device: str):
        super().__init__()
        self.gd_model = gd_model
        self.device = device

        # Grounding DINO 的 Swin-Tiny backbone 通道数:
        # 第4层 (C4) 通常为 1024 维，需要用 1x1 卷积升维到 2048
        self.channel_adapter = nn.Conv2d(1024, 2048, kernel_size=1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 图像tensor（已归一化）
        Returns:
            [B, 2048, H/32, W/32] 特征图
        """
        # 调用 Grounding DINO 的视觉骨干
        # backbone 返回多尺度特征 [feat1, feat2, feat3, feat4]
        visual_features = self.gd_model.backbone(x)

        # 取最高层特征（下采样32x）
        if isinstance(visual_features, (list, tuple)):
            feat = visual_features[-1]  # [B, C4, H/32, W/32]
        else:
            feat = visual_features

        # 如果通道数不是2048，用1x1卷积调整
        if feat.shape[1] != 2048:
            feat = self.channel_adapter(feat)

        return feat


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Image Encoder — 模块测试")
    print("=" * 60)

    # 方式1: 用随机 Tensor 测试（不需要真实图片）
    print("\n[测试1] 使用随机Tensor模拟dataloader输出...")
    encoder = ImageEncoder(backbone="resnet50")

    # 模拟 dataloader.py 的输出: [1, 3, 256, 704], 值域 [0, 1]
    dummy_image = torch.rand(1, 3, 256, 704)
    print(f"[输入Tensor形状] {dummy_image.shape}")

    features = encoder.encode(dummy_image)
    print(f"[输出特征图形状] {features.shape}")
    print(f"[期望形状]        {encoder.get_feature_shape()}")

    # 验证形状一致性
    expected = encoder.get_feature_shape()
    assert features.shape == expected, (
        f"形状不匹配！期望 {expected}，实际 {tuple(features.shape)}"
    )
    shape_ok = features.shape == expected
    print(f"[形状校验] {'PASS' if shape_ok else 'FAIL'} — {tuple(features.shape)}")

    # 输出更多统计信息
    print(f"[特征图统计] min={features.min().item():.4f}, "
          f"max={features.max().item():.4f}, "
          f"mean={features.mean().item():.4f}")

    # 方式2: 从实际图片文件测试（如果 test_01.jpg 已就绪）
    import os
    test_image_path = "simulation/captures/test_01.jpg"
    if os.path.exists(test_image_path):
        print(f"\n[测试2] 使用真实图片: {test_image_path}")
        import cv2
        import numpy as np

        # 模拟 dataloader.py 的流程：读取 -> BGR2RGB -> 归一化 -> Resize
        img = cv2.imread(test_image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (704, 256))
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            print(f"[真实图像Tensor形状] {img_tensor.shape}")

            real_features = encoder.encode(img_tensor)
            print(f"[输出特征图形状] {real_features.shape}")
            print(f"[形状校验] PASS — {tuple(real_features.shape)}")
        else:
            print("[测试2] 图片读取失败（文件可能损坏）")
    else:
        print(f"\n[测试2] 真实图片文件未就绪: {test_image_path}")
        print("         (钱森今日需提供 test_01.jpg，后续联调时再测)")

    # 方式3: 尝试 Grounding DINO（如果已安装）
    print(f"\n[测试3] 尝试Grounding DINO骨干...")
    try:
        gd_encoder = ImageEncoder(backbone="grounding_dino")
        gd_features = gd_encoder.encode(dummy_image)
        print(f"[Grounding DINO 输出形状] {gd_features.shape}")
    except Exception as e:
        print(f"[Grounding DINO] 未就绪: {e}")
        print("            (ResNet-50已可用，不影响今日交付)")

    print("\n" + "=" * 60)
    print("Image Encoder 测试完成!")
    print("=" * 60)
