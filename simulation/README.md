# AirSim Blocks（钱森 7 月 17 日任务）

## 一键验收

双击项目根目录下的 `钱森_7月17日AirSim验收.bat`。脚本会：

1. 以 640×480 窗口模式启动 Blocks，适配无独立显卡的电脑；
2. 等待 AirSim RPC 服务就绪；
3. 运行 PythonClient 示例，让无人机起飞并前移；
4. 进入键盘控制模式。

如果 Blocks 已经打开，只需双击 `simulation/start_keyboard_control.bat`
即可直接进入键盘控制。

## 键盘控制

| 按键 | 动作 |
|---|---|
| `T` | 起飞 |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `R` / `F` | 上升 / 下降 |
| `Q` / `E` | 左转 / 右转 |
| `H` | 悬停 |
| `L` | 降落 |
| `X` | 降落并退出 |

官方 AirSim 的多旋翼模式不原生支持键盘飞行，本项目通过
`keyboard_control.py` 调用 AirSim API 实现键盘移动。

## 分步运行

```powershell
.\simulation\launch_blocks_low_spec.bat
& C:\ProgramData\anaconda3\envs\bev_drone\python.exe .\simulation\hello_drone.py --move
& C:\ProgramData\anaconda3\envs\bev_drone\python.exe .\simulation\keyboard_control.py
```

验收截图建议保留 Blocks 画面，以及控制台中的
`AirSim RPC 连接成功`、`PythonClient 示例运行成功` 和实时坐标。

## 7 月 18 日图像与深度采集

双击项目根目录的 `钱森_7月18日图像采集验收.bat`，脚本会自动启动
Blocks、起飞到约 3 米、同步采集图像并降落。输出位于 `captures/`：

| 文件 | 内容 |
|---|---|
| `test_01.jpg` | 前视 RGB 图像 |
| `test_01_down.jpg` | 下视 RGB 图像 |
| `depth.png` | 与 `test_01.jpg` 对齐的前视 DepthPlanar，uint16，单位为毫米 |
| `depth.npy` | 前视原始 float32 深度，单位为米 |
| `depth_preview.png` | 前视深度的彩色预览 |
| `depth_down.png` | 与下视 RGB 对齐的下视深度 |
| `depth_down.npy` | 下视原始 float32 深度，单位为米 |
| `depth_down_preview.png` | 下视深度的彩色预览 |
| `capture_metadata.json` | 相机、尺寸、深度范围与位姿信息 |

相机分辨率统一配置为 704×256。读取真实深度时使用：

```python
depth_m = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED).astype("float32") / 1000
```
