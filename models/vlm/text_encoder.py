"""
text_encoder.py — CLIP文本编码器
职责：接收中文指令文本，通过CLIP模型输出固定维度的文本特征向量

输入：自然语言指令字符串（如"飞到红色屋顶建筑附近"）
输出：文本特征向量（512维 或 768维），打印形状

使用模型：openai/clip-vit-base-patch32（输出512维）
备选中文模型：OFA-Sys/chinese-clip-vit-base-patch16（输出768维，中文效果更好）
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class TextEncoder:
    """
    CLIP文本编码器
    将自然语言指令编码为固定维度的特征向量，用于后续与视觉特征进行相似度计算
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        初始化文本编码器

        Args:
            model_name: HuggingFace模型名称
                        - "openai/clip-vit-base-patch32": 原始CLIP，输出512维
                        - "OFA-Sys/chinese-clip-vit-base-patch16": 中文CLIP，输出768维
            device: 运行设备 ("cuda" 或 "cpu")，None则自动选择
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[TextEncoder] 加载模型: {model_name}")
        print(f"[TextEncoder] 使用设备: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # 获取模型输出的向量维度
        self.embed_dim = self.model.config.projection_dim
        print(f"[TextEncoder] 模型已加载，文本向量维度: {self.embed_dim}")

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        将文本编码为特征向量

        Args:
            text: 输入文本字符串（中文/英文）

        Returns:
            text_features: 形状为 [1, embed_dim] 的特征向量
        """
        # 预处理文本
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        # 将输入移到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播获取文本特征
        text_features = self.model.get_text_features(**inputs)

        # 兼容不同版本transformers的返回格式
        # 新版返回 BaseModelOutputWithPooling，需提取 tensor
        if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
            text_features = text_features.pooler_output
        elif hasattr(text_features, "last_hidden_state"):
            # 如果没有pooler，取[CLS] token的输出
            text_features = text_features.last_hidden_state[:, 0, :]

        # L2归一化（CLIP标准做法）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def encode_to_numpy(self, text: str) -> np.ndarray:
        """将文本编码为numpy数组（用于后续与BEV特征做相似度计算）"""
        features = self.encode(text)
        return features.cpu().numpy()


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # 测试指令（来自任务文档）
    test_instruction = "飞到红色屋顶建筑附近"

    print("=" * 60)
    print("CLIP Text Encoder — 模块测试")
    print("=" * 60)

    # 初始化编码器
    # 如果中文效果不佳，可切换为 "OFA-Sys/chinese-clip-vit-base-patch16"
    encoder = TextEncoder(model_name="openai/clip-vit-base-patch32")

    # 编码文本
    print(f"\n[输入文本] {test_instruction}")
    text_vector = encoder.encode(test_instruction)

    # 输出结果
    print(f"\n[输出向量形状] {text_vector.shape}")
    print(f"[输出向量维度] {text_vector.shape[1]} 维")
    print(f"[向量前10个值]  {text_vector[0, :10].cpu().numpy()}")
    print(f"[向量范数]      {text_vector.norm(dim=-1).item():.6f} (L2归一化后应为1.0)")

    # 保存向量到文件（用于后续模块联调）
    output_path = "data/text_vector.npy"
    import os
    os.makedirs("data", exist_ok=True)
    np.save(output_path, encoder.encode_to_numpy(test_instruction))
    print(f"\n[保存] 文本向量已保存至: {output_path}")

    print("=" * 60)
    print("Text Encoder 测试通过！")
    print("=" * 60)
