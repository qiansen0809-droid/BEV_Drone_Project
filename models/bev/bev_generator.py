"""将前视 RGB 图像 + 深度图"拍扁"成 BEV 俯视图并保存为 bev_map.png。

核心流程:
    1. 读取 test_01.jpg (RGB) + depth.png (uint16 mm 深度)
    2. 利用相机内参将每个像素反投影为 3D 点云
    3. 将点云投影到俯视 BEV 栅格，聚合 RGB 颜色
    4. 保存 bev_map.png

所有相机参数从 configs/base_config.yaml 读取。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base_config.yaml"
CAPTURES_DIR = PROJECT_ROOT / "simulation" / "captures"
RGB_PATH = CAPTURES_DIR / "test_01.jpg"
DEPTH_PATH = CAPTURES_DIR / "depth.png"
OUTPUT_PATH = CAPTURES_DIR / "bev_map.png"


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_intrinsics(intr: dict) -> np.ndarray:
    """从配置构建 3×3 内参矩阵 K."""
    return np.array([
        [intr["fx"], 0,          intr["cx"]],
        [0,          intr["fy"], intr["cy"]],
        [0,          0,          1         ],
    ], dtype=np.float32)


def build_extrinsics_rot(extr: dict) -> np.ndarray:
    """从配置的 pitch/roll/yaw (度) 构建旋转矩阵 R.

    Tait-Bryan: R = Rz(yaw) * Ry(roll) * Rx(pitch)
    目前只用到 pitch (绕 X 轴俯仰)，其余为 0.
    """
    pitch = np.deg2rad(extr["pitch_deg"])
    roll = np.deg2rad(extr["roll_deg"])
    yaw = np.deg2rad(extr["yaw_deg"])

    Rx = np.array([[1, 0,           0         ],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)

    Ry = np.array([[ np.cos(roll), 0, np.sin(roll)],
                    [ 0,           1, 0           ],
                    [-np.sin(roll), 0, np.cos(roll)]], dtype=np.float32)

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,            0,           1]], dtype=np.float32)

    return Rz @ Ry @ Rx


def build_extrinsics_trans(extr: dict) -> np.ndarray:
    """从配置构建平移向量 t (NED)."""
    t = extr["translation"]
    return np.array([t["x"], t["y"], t["z"]], dtype=np.float32)


def load_rgb(path: Path) -> np.ndarray:
    """读取 RGB 图像, BGR→RGB, 转为 float [0,1]."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_depth_meters(path: Path) -> np.ndarray:
    """读取 uint16 mm 深度图, 转为 float 米."""
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"无法读取: {path}")
    depth_m = depth_mm.astype(np.float32) / 1000.0  # mm → m
    depth_m[depth_m <= 0] = np.inf
    return depth_m


def unproject(depth_m: np.ndarray, K_inv: np.ndarray, H: int, W: int) -> np.ndarray:
    """像素坐标 + 深度 + 内参逆 → 相机坐标系 3D 点 (H, W, 3)."""
    v, u = np.mgrid[0:H, 0:W]
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float32)
    rays = uv1 @ K_inv.T  # (H, W, 3)
    return rays * depth_m[:, :, np.newaxis]


def cam_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """相机坐标系 → 世界 NED 坐标系."""
    return pts_cam @ R.T + t


def generate_bev(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K_inv: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    grid_size: int,
    x_range: tuple,
    y_range: tuple,
) -> np.ndarray:
    """RGB + 深度 + 内外参 → BEV 俯视图."""
    x_res = (x_range[1] - x_range[0]) / grid_size
    y_res = (y_range[1] - y_range[0]) / grid_size

    # 1. 反投影: 像素 → 相机坐标系 3D
    pts_cam = unproject(depth_m, K_inv, H, W)

    # 2. 相机 → 世界
    pts_world = cam_to_world(pts_cam, R, t)

    # 3. 世界 → BEV 栅格索引
    x_w, y_w = pts_world[:, :, 0], pts_world[:, :, 1]
    xi = ((y_w - y_range[0]) / y_res).astype(np.int32)
    yi = ((x_range[1] - x_w) / x_res).astype(np.int32)
    valid = (xi >= 0) & (xi < grid_size) & (yi >= 0) & (yi < grid_size)

    # 4. 聚合 RGB 到 BEV 栅格
    bev = np.zeros((grid_size, grid_size, 3), dtype=np.float64)
    cnt = np.zeros((grid_size, grid_size), dtype=np.int32)

    xi_v, yi_v = xi[valid], yi[valid]
    rgb_v = rgb[valid]

    np.add.at(bev, (yi_v, xi_v), rgb_v)
    np.add.at(cnt, (yi_v, xi_v), 1)

    valid_bev = cnt > 0
    bev[valid_bev] /= cnt[valid_bev, np.newaxis]

    # 5. 填充空洞
    mask_u8 = (cnt > 0).astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=2)
    fill_mask = (dilated > 0) & (cnt == 0)
    if fill_mask.any():
        filled = cv2.blur(bev.astype(np.float32), (5, 5))
        bev[fill_mask] = filled[fill_mask]

    return np.clip(bev, 0, 1)


def add_grid_overlay(bev: np.ndarray, grid_size: int, x_range: tuple, y_range: tuple) -> np.ndarray:
    """在 BEV 图上叠加距离网格."""
    x_res = (x_range[1] - x_range[0]) / grid_size
    y_res = (y_range[1] - y_range[0]) / grid_size

    result = bev.copy()
    color = np.array([0.3, 0.3, 0.3])

    for d in range(5, 50, 5):
        idx = int((x_range[1] - d) / x_res)
        if 0 <= idx < grid_size:
            result[idx, :] = np.where(result[idx, :] > 0, result[idx, :] * 0.6 + color * 0.4, color)

    for d in range(-20, 25, 5):
        idx = int((d - y_range[0]) / y_res)
        if 0 <= idx < grid_size:
            result[:, idx] = np.where(result[:, idx] > 0, result[:, idx] * 0.6 + color * 0.4, color)

    return result


def main() -> None:
    # 加载配置
    print(f"加载配置: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    cam_cfg = cfg["camera"]["front"]
    intr = cam_cfg["intrinsics"]
    extr = cam_cfg["extrinsics"]
    bev_cfg = cfg["bev"]

    H, W = intr["height"], intr["width"]
    K_inv = np.linalg.inv(build_intrinsics(intr))
    R = build_extrinsics_rot(extr)
    t = build_extrinsics_trans(extr)

    grid_size = bev_cfg["height"]  # 200
    x_range = tuple(bev_cfg["x_range"])
    y_range = tuple(bev_cfg["y_range"])
    depth_min = bev_cfg["depth_min"]
    depth_max = bev_cfg["depth_max"]

    print(f"  内参 fx={intr['fx']}, fy={intr['fy']}, cx={intr['cx']}, cy={intr['cy']}")
    print(f"  外参 pitch={extr['pitch_deg']}°, trans={t}")
    print(f"  BEV {grid_size}×{grid_size}, X{x_range}, Y{y_range}")

    # 加载数据
    print("加载 RGB...")
    rgb = load_rgb(RGB_PATH)

    print("加载深度图...")
    depth_m = load_depth_meters(DEPTH_PATH)
    depth_m = np.clip(depth_m, depth_min, depth_max)

    # 生成 BEV
    print("生成 BEV 俯视图...")
    bev = generate_bev(rgb, depth_m, K_inv, R, t, H, W, grid_size, x_range, y_range)

    # 叠加网格
    print("叠加网格...")
    bev_with_grid = add_grid_overlay(bev, grid_size, x_range, y_range)

    # 保存
    bev_uint8 = (bev_with_grid * 255).astype(np.uint8)
    bev_bgr = cv2.cvtColor(bev_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_PATH), bev_bgr)
    print(f"BEV 俯视图已保存: {OUTPUT_PATH}")
    print(f"  尺寸: {grid_size}×{grid_size}")
    print(f"  范围: X {x_range}m, Y {y_range}m")


if __name__ == "__main__":
    main()
