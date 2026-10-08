"""
similarity.py — 文本-BEV特征余弦相似度计算
=============================================
职责：将文本向量与BEV图上的假特征（随机生成）做余弦相似度点积，
      打印相似度数值。这是联调前的独立验证模块。

输入：
  - 文本指令 "飞到红色屋顶建筑附近" → 文本向量 [1, 512]
  - 假BEV特征（随机生成） → [1, 512, 200, 200]

输出：
  - 相似度热力图统计值（max/mean/min）
  - 相似度数值（用于后续与 json_parser 联调）

依赖：
  - text_encoder.py: CLIP文本编码
  - base_config.yaml: BEV尺寸 200×200
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import yaml

from models.vlm.text_encoder import TextEncoder


def load_bev_config():
    """从配置文件读取BEV尺寸"""
    config_path = os.path.join(PROJECT_ROOT, "configs", "base_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["bev"]["height"], config["bev"]["width"]


def generate_fake_bev_features(bev_h, bev_w, feature_dim, device="cpu"):
    """
    生成假BEV特征（模拟LSS模型输出）

    真实联调时，这里会替换为 LSS 模型的实际输出。
    目前用随机高斯噪声代替，确保相似度在合理范围(0~1)。

    Args:
        bev_h: BEV 高度 (200)
        bev_w: BEV 宽度 (200)
        feature_dim: 特征维度 (512, 与CLIP文本向量对齐)
        device: 计算设备

    Returns:
        fake_bev: [1, feature_dim, bev_h, bev_w]
    """
    # 用高斯随机数模拟，F.normalize 确保每个空间位置的向量模长为1
    fake_bev = torch.randn(1, feature_dim, bev_h, bev_w, device=device)
    # L2归一化，使余弦相似度 = 向量点积
    fake_bev = fake_bev / fake_bev.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return fake_bev


def cosine_similarity_map(text_vec, bev_features):
    """
    计算文本向量与BEV每个空间位置的余弦相似度

    Args:
        text_vec:   [1, D] — L2归一化后的文本向量
        bev_features: [1, D, H, W] — L2归一化后的BEV特征

    Returns:
        sim_map: [1, H, W] — 每个BEV格子的相似度 (范围[-1, 1])
    """
    # text_vec: [1, D] -> [1, D, 1, 1]
    text_expanded = text_vec.unsqueeze(-1).unsqueeze(-1)
    # 余弦相似度 = 点积（因为两个向量已归一化）
    sim_map = (bev_features * text_expanded).sum(dim=1, keepdim=False)
    return sim_map  # [1, H, W]


def main():
    print("=" * 60)
    print("  Text-BEV Cosine Similarity")
    print("  Task: 7.19 - Xing Yun")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 读取配置
    bev_h, bev_w = load_bev_config()
    print(f"\n  BEV config: {bev_h} x {bev_w} (from base_config.yaml)")

    # 2. 加载文本编码器，编码"周二的指令"
    print("\n  Loading TextEncoder...")
    encoder = TextEncoder(model_name="openai/clip-vit-base-patch32", device=device)
    instruction = "飞到红色屋顶建筑附近"
    text_vec = encoder.encode(instruction)  # [1, 512], L2归一化
    feature_dim = text_vec.shape[1]
    print(f"  Instruction: {instruction}")
    print(f"  Text vector shape: {list(text_vec.shape)}")
    print(f"  Text vector norm:  {text_vec.norm(dim=-1).item():.6f} (should be 1.0)")

    # 3. 生成假BEV特征（模拟LSS输出）
    print(f"\n  Generating fake BEV features ({feature_dim}, {bev_h}, {bev_w})...")
    fake_bev = generate_fake_bev_features(bev_h, bev_w, feature_dim, device)
    print(f"  Fake BEV shape: {list(fake_bev.shape)}")
    print(f"  Per-location norm check: "
          f"min={fake_bev.norm(dim=1).min().item():.4f}, "
          f"max={fake_bev.norm(dim=1).max().item():.4f}")

    # 4. 计算余弦相似度
    print(f"\n  Computing cosine similarity map...")
    sim_map = cosine_similarity_map(text_vec, fake_bev)  # [1, 200, 200]
    sim_np = sim_map.cpu().numpy().squeeze()  # [200, 200]

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Similarity map shape: {list(sim_map.shape)}")
    print(f"  Max  similarity: {sim_np.max():.6f}")
    print(f"  Min  similarity: {sim_np.min():.6f}")
    print(f"  Mean similarity: {sim_np.mean():.6f}")
    print(f"  Std  similarity: {sim_np.std():.6f}")

    # 找最匹配的位置（模拟目标定位）
    max_idx = np.unravel_index(sim_np.argmax(), sim_np.shape)
    print(f"\n  Best match position (max similarity):")
    print(f"    BEV coord: [{max_idx[1]}, {max_idx[0]}]")
    print(f"    Similarity: {sim_np[max_idx]:.6f}")

    # 6. 打印最终交付数值
    print("\n" + "=" * 60)
    print("  Deliverable: Cosine Similarity Value")
    print("=" * 60)
    print(f"  similarity_max  = {sim_np.max():.6f}")
    print(f"  similarity_mean = {sim_np.mean():.6f}")
    print(f"  similarity_min  = {sim_np.min():.6f}")
    print(f"\n  (Note: values are random since BEV features are fake.")
    print(f"   Will be meaningful when LSS outputs real BEV features.)")
    print("=" * 60)

    # 7. 保存相似度数值到文件（供后续联调使用）
    output_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "similarity_values.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"instruction: {instruction}\n")
        f.write(f"similarity_max: {sim_np.max():.6f}\n")
        f.write(f"similarity_mean: {sim_np.mean():.6f}\n")
        f.write(f"similarity_min: {sim_np.min():.6f}\n")
        f.write(f"similarity_std: {sim_np.std():.6f}\n")
        f.write(f"best_match_bev_coord: [{max_idx[1]}, {max_idx[0]}]\n")
    print(f"\n  Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
