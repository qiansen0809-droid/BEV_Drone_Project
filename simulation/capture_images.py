"""采集 AirSim 无人机前视、下视 RGB 图像和下视深度图。"""

import argparse
import json
from pathlib import Path

import airsim
import cv2
import numpy as np

from airsim_utils import wait_for_multirotor


FRONT_CAMERA = "front_center"
DOWN_CAMERA = "bottom_center"
DEPTH_SCALE = 1000.0  # depth.png 以毫米为单位
MAX_DEPTH_METERS = 65.535  # uint16 毫米深度图的最大可表示距离


def decode_scene(response: airsim.ImageResponse) -> np.ndarray:
    """把 AirSim 返回的压缩 Scene 图像解码为 OpenCV BGR 图像。"""
    if not response.image_data_uint8:
        raise RuntimeError("AirSim 返回了空的 RGB 图像")

    encoded = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV 无法解码 AirSim RGB 图像")
    return image


def decode_depth(response: airsim.ImageResponse) -> np.ndarray:
    """把 DepthPlanar 浮点数据转换为 H×W、单位为米的数组。"""
    expected = response.width * response.height
    depth = np.asarray(response.image_data_float, dtype=np.float32)
    if response.width <= 0 or response.height <= 0 or depth.size != expected:
        raise RuntimeError(
            "AirSim 深度图尺寸异常："
            f"width={response.width}, height={response.height}, "
            f"values={depth.size}"
        )
    return depth.reshape(response.height, response.width)


def save_depth(
    depth_m: np.ndarray,
    output_dir: Path,
    stem: str,
) -> dict[str, float]:
    """保存 uint16 毫米深度图、原始数组和便于截图的彩色预览。"""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        raise RuntimeError("深度图中没有有效像素")

    clean_depth = np.where(valid, depth_m, 0.0).astype(np.float32)
    depth_mm = np.clip(
        clean_depth * DEPTH_SCALE,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)

    if not cv2.imwrite(str(output_dir / f"{stem}.png"), depth_mm):
        raise RuntimeError(f"保存 {stem}.png 失败")
    np.save(output_dir / f"{stem}.npy", clean_depth)

    valid_values = clean_depth[valid]
    near = float(np.percentile(valid_values, 2))
    far = float(np.percentile(valid_values, 98))
    if far <= near:
        far = near + 1.0
    preview_gray = np.clip((clean_depth - near) / (far - near), 0, 1)
    preview_gray = (255 * (1 - preview_gray)).astype(np.uint8)
    preview = cv2.applyColorMap(preview_gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    preview_name = f"{stem}_preview.png"
    if not cv2.imwrite(str(output_dir / preview_name), preview):
        raise RuntimeError(f"保存 {preview_name} 失败")

    return {
        "min_depth_m": float(valid_values.min()),
        "max_depth_m": float(valid_values.max()),
        "mean_depth_m": float(valid_values.mean()),
    }


def capture(output_dir: Path, vehicle_name: str, takeoff_height: float) -> None:
    """连接 AirSim、起飞、同步采集三路图像并保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = wait_for_multirotor()
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)

    state = client.getMultirotorState(vehicle_name)
    if state.landed_state == airsim.LandedState.Landed:
        print("无人机起飞...")
        client.takeoffAsync(vehicle_name=vehicle_name).join()

    print(f"移动到约 {takeoff_height:.1f} 米高度...")
    client.moveToZAsync(
        -abs(takeoff_height),
        velocity=1,
        vehicle_name=vehicle_name,
    ).join()
    client.hoverAsync(vehicle_name=vehicle_name).join()

    print("同步采集前视/下视 RGB 和前视/下视 DepthPlanar...")
    client.simPause(True)
    try:
        responses = client.simGetImages(
            [
                airsim.ImageRequest(
                    FRONT_CAMERA,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=True,
                ),
                airsim.ImageRequest(
                    DOWN_CAMERA,
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=True,
                ),
                airsim.ImageRequest(
                    FRONT_CAMERA,
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False,
                ),
                airsim.ImageRequest(
                    DOWN_CAMERA,
                    airsim.ImageType.DepthPlanar,
                    pixels_as_float=True,
                    compress=False,
                ),
            ],
            vehicle_name=vehicle_name,
        )
    finally:
        client.simPause(False)

    if len(responses) != 4:
        raise RuntimeError(f"预期返回 4 张图像，实际返回 {len(responses)} 张")

    front = decode_scene(responses[0])
    down = decode_scene(responses[1])
    depth_m = decode_depth(responses[2])
    depth_down_m = decode_depth(responses[3])

    if not cv2.imwrite(
        str(output_dir / "test_01.jpg"),
        front,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError("保存 test_01.jpg 失败")
    if not cv2.imwrite(
        str(output_dir / "test_01_down.jpg"),
        down,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError("保存 test_01_down.jpg 失败")

    depth_stats = save_depth(depth_m, output_dir, "depth")
    depth_down_stats = save_depth(depth_down_m, output_dir, "depth_down")
    position = client.getMultirotorState(
        vehicle_name
    ).kinematics_estimated.position
    metadata = {
        "vehicle_name": vehicle_name,
        "front_camera": FRONT_CAMERA,
        "down_camera": DOWN_CAMERA,
        "depth_camera": FRONT_CAMERA,
        "depth_down_camera": DOWN_CAMERA,
        "rgb_width": int(front.shape[1]),
        "rgb_height": int(front.shape[0]),
        "depth_width": int(depth_m.shape[1]),
        "depth_height": int(depth_m.shape[0]),
        "depth_type": "DepthPlanar",
        "depth_png_unit": "millimeter",
        "depth_png_scale": DEPTH_SCALE,
        "depth_png_max_m": MAX_DEPTH_METERS,
        "position_ned_m": {
            "x": position.x_val,
            "y": position.y_val,
            "z": position.z_val,
        },
        **depth_stats,
        "down_depth_min_m": depth_down_stats["min_depth_m"],
        "down_depth_max_m": depth_down_stats["max_depth_m"],
        "down_depth_mean_m": depth_down_stats["mean_depth_m"],
    }
    (output_dir / "capture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"前视 RGB：{front.shape[1]}×{front.shape[0]} -> test_01.jpg")
    print(
        f"下视 RGB：{down.shape[1]}×{down.shape[0]} -> test_01_down.jpg"
    )
    print(
        f"前视深度：{depth_m.shape[1]}×{depth_m.shape[0]}，"
        f"范围 {depth_stats['min_depth_m']:.2f}~"
        f"{depth_stats['max_depth_m']:.2f} m -> depth.png"
    )
    print(
        f"下视深度：{depth_down_m.shape[1]}×{depth_down_m.shape[0]}，"
        f"范围 {depth_down_stats['min_depth_m']:.2f}~"
        f"{depth_down_stats['max_depth_m']:.2f} m -> depth_down.png"
    )
    print(f"图像采集成功！输出目录：{output_dir.resolve()}")

    print("无人机降落...")
    client.landAsync(vehicle_name=vehicle_name).join()
    client.armDisarm(False, vehicle_name)
    client.enableApiControl(False, vehicle_name)


def main() -> None:
    default_output = Path(__file__).resolve().parent / "captures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="输出目录，默认 simulation/captures",
    )
    parser.add_argument("--vehicle-name", default="Drone1")
    parser.add_argument("--height", type=float, default=3.0)
    args = parser.parse_args()

    try:
        capture(args.output_dir, args.vehicle_name, args.height)
    except Exception as exc:
        print(f"图像采集失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
