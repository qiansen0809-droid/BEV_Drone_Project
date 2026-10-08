"""使用 Windows 键盘通过 AirSim API 控制多旋翼无人机。"""

import ctypes
import time

import airsim

from airsim_utils import wait_for_multirotor


KEY = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "Q": 0x51,
    "E": 0x45,
    "R": 0x52,
    "F": 0x46,
    "T": 0x54,
    "H": 0x48,
    "L": 0x4C,
    "X": 0x58,
}


def is_pressed(name: str) -> bool:
    """判断指定按键当前是否按下。"""
    return bool(ctypes.windll.user32.GetAsyncKeyState(KEY[name]) & 0x8000)


def print_help() -> None:
    print("\n键盘控制已启动，请保持本终端或 Blocks 窗口处于前台：")
    print("  T       起飞")
    print("  W / S   前进 / 后退")
    print("  A / D   左移 / 右移")
    print("  R / F   上升 / 下降")
    print("  Q / E   左转 / 右转")
    print("  H       悬停")
    print("  L       降落")
    print("  X       降落并退出")


def main() -> None:
    client = wait_for_multirotor()
    client.enableApiControl(True)
    client.armDisarm(True)
    print_help()

    speed = 2.0
    yaw_speed = 35.0
    last_keys: set[str] = set()
    last_status = 0.0

    try:
        while True:
            pressed = {name for name in KEY if is_pressed(name)}
            new_keys = pressed - last_keys

            if "X" in new_keys:
                print("正在降落并退出...")
                client.landAsync().join()
                break
            if "T" in new_keys:
                print("起飞...")
                client.takeoffAsync().join()
            if "L" in new_keys:
                print("降落...")
                client.landAsync().join()
            if "H" in new_keys:
                print("悬停")
                client.hoverAsync().join()

            vx = speed * (("W" in pressed) - ("S" in pressed))
            vy = speed * (("D" in pressed) - ("A" in pressed))
            vz = speed * (("F" in pressed) - ("R" in pressed))
            yaw_rate = yaw_speed * (("E" in pressed) - ("Q" in pressed))

            if vx or vy or vz or yaw_rate:
                client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    vz,
                    0.15,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(
                        is_rate=True,
                        yaw_or_rate=yaw_rate,
                    ),
                )

            now = time.monotonic()
            if now - last_status >= 1:
                position = (
                    client.getMultirotorState().kinematics_estimated.position
                )
                print(
                    f"\r位置 x={position.x_val:7.2f} "
                    f"y={position.y_val:7.2f} "
                    f"z={position.z_val:7.2f}",
                    end="",
                    flush=True,
                )
                last_status = now

            last_keys = pressed
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，无人机悬停后退出。")
        client.hoverAsync().join()
    finally:
        state = client.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            client.armDisarm(False)
            client.enableApiControl(False)
        print("\n键盘控制结束。")


if __name__ == "__main__":
    main()
