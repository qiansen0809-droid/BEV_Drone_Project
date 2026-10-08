"""AirSim 示例共享的连接工具。"""

import time

import airsim


def wait_for_multirotor(timeout: float = 90.0) -> airsim.MultirotorClient:
    """等待 Blocks 启动并返回已经连通的多旋翼客户端。"""
    deadline = time.monotonic() + timeout
    print("正在等待 AirSim Blocks 启动", end="", flush=True)

    while time.monotonic() < deadline:
        client = airsim.MultirotorClient(timeout_value=2)
        try:
            if client.ping():
                print("\nAirSim RPC 连接成功！")
                print(
                    f"Client Ver:{client.getClientVersion()}, "
                    f"Server Ver:{client.getServerVersion()}"
                )
                return client
        except Exception:  # AirSim 未启动时 RPC 会抛出连接异常
            pass

        print(".", end="", flush=True)
        time.sleep(1)

    raise TimeoutError(
        "在规定时间内没有连接到 AirSim。请确认 Blocks 窗口已经正常打开。"
    )
