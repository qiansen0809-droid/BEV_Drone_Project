"""7 月 17 日 AirSim PythonClient 连通与移动示例。"""

import argparse

import airsim

from airsim_utils import wait_for_multirotor


def run_demo(move: bool) -> None:
    """连接无人机；可选执行起飞和前移演示。"""
    client = wait_for_multirotor()
    client.simRunConsoleCommand("t.MaxFPS 15")
    client.enableApiControl(True)
    client.armDisarm(True)

    state = client.getMultirotorState()
    print(f"当前飞行状态：{state.landed_state}")

    if move:
        if state.landed_state == airsim.LandedState.Landed:
            print("无人机起飞到约 2 米高度...")
            client.takeoffAsync().join()
            client.moveToZAsync(-2, 1).join()

        print("无人机向前移动 2 米...")
        client.moveByVelocityBodyFrameAsync(1, 0, 0, 2).join()
        client.hoverAsync().join()

    position = client.getMultirotorState().kinematics_estimated.position
    print(
        "无人机位置："
        f"x={position.x_val:.2f}, "
        f"y={position.y_val:.2f}, "
        f"z={position.z_val:.2f}"
    )
    print("PythonClient 示例运行成功！")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move",
        action="store_true",
        help="连接后自动起飞并向前移动 2 米",
    )
    args = parser.parse_args()

    try:
        run_demo(args.move)
    except Exception as exc:
        print(f"示例运行失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
