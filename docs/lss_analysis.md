# LSS (Lift, Splat, Shoot) 参考分析

> 原文: [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)
> 仓库: [nv-tlabs/lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)

---

## 1. 整体架构

```
多相机图像 (B×N×3×H×W) + 相机标定参数
         │
    ┌────▼────┐  Lift: EfficientNet-B0 提取图像特征 + DepthNet 预测逐像素深度分布
    │ CamEncode│  → 3D-aware Features (B×N×D×H'×W'×C')
    └────┬────┘
         │
    ┌────▼────┐  Splat: 相机坐标系 → 自车坐标系 → 体素池化（Voxel Pooling）
    │VoxelPool │  → BEV Feature Grid (B×C×X×Y)
    └────┬────┘
         │
    ┌────▼────┐  Shoot: ResNet-18 编码器 → 上采样层 → 最终输出
    │ BevEncode│  → BEV Segmentation (B×1×X×Y)
    └─────────┘
```

### 三阶段含义

| 阶段 | 含义 | 核心操作 |
|------|------|---------|
| **Lift** | 将 2D 图像"抬升"到 3D | EfficientNet 提取特征 + 预测 D 个深度 bin 的概率分布，特征 × 深度概率 → 3D 感知特征 |
| **Splat** | 将多视角 3D 特征"拍扁"到 BEV | 相机坐标系→自车坐标系转换 → 体素池化聚合 |
| **Shoot** | 从 BEV 特征"射"出最终结果 | ResNet-18 + 上采样 → 语义分割输出 |

---

## 2. 输入

### 2.1 参数列表

| 字段 | 形状 | 说明 |
|------|------|------|
| `imgs` | `(B, N, 3, H, W)` | N 路摄像头图像 |
| `rots` | `(B, N, 3, 3)` | 相机外参旋转矩阵 |
| `trans` | `(B, N, 3)` | 相机外参平移向量 |
| `intrins` | `(B, N, 3, 3)` | 相机内参矩阵 K |
| `post_rots` | `(B, N, 3, 3)` | 数据增强后的旋转修正 |
| `post_trans` | `(B, N, 3)` | 数据增强后的平移修正 |

### 2.2 图像预处理

- 原始尺寸: `(H=900, W=1600)` → 缩放/裁剪 → `final_dim`（默认 `(128, 352)`）
- 归一化: `(x / 255.0 - 0.5)`
- 训练增强: 随机缩放 `(0.193~0.225)` + 旋转 `(-5.4°~5.4°)` + 翻转 + 底部裁剪

---

## 3. Grid Configuration

```python
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],   # X: ±50m, 分辨率 0.5m → 200 格
    'ybound': [-50.0, 50.0, 0.5],   # Y: ±50m, 分辨率 0.5m → 200 格
    'zbound': [-10.0, 10.0, 20.0],  # Z: ±10m, 步长 20m → 1 个 bin（忽略高度）
}
```

BEV 输出: **200×200 栅格**，对应实车 ±50m 范围。

---

## 4. 输出

| 字段 | 形状 | 说明 |
|------|------|------|
| BEV 分割 | `(B, 1, 200, 200)` | 车辆占用二值分割图 |

- 原始论文: Vehicle IOU **32.07%**（仓库版本 **33.03%**）
- 训练: 10k epochs（实际约 20k steps 收敛），8×2080Ti 约 3 天

---

## 5. 数据增强配置

```python
data_aug_conf = {
    'resize_lim': (0.193, 0.225),
    'rot_lim': (-5.4, 5.4),
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,  # 训练时随机选 5 路
}
```

---

## 6. 关键源码文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `main.py` | ~30 | CLI 入口，调度训练/评估/可视化 |
| `train.py` | ~80 | 标准 PyTorch 训练循环 |
| `data.py` | ~270 | nuScenes 数据加载 (`SegmentationData`, `VizData`) |
| `models.py` | ~260 | 模型定义 (`LiftSplatShoot`, `CamEncode`, `BevEncode`) |
| `tools.py` | ~180 | LiDAR 处理、坐标变换、数据增强 |
| `explore.py` | ~120 | 模型验证与可视化 |

---

## 7. 与本项目的适配点

| LSS 原版 | 本项目 (BEV Drone) | 适配 |
|----------|-------------------|------|
| nuScenes 车规 6 相机 | AirSim 无人机 4 相机 | 修改 `camera.num: 4` |
| BEV 200×200 / ±50m | `bev.height: 200, width: 200` | 一致 |
| 图像 128×352 | `image.height: 256, width: 704` | 需调整 backbone 输入 |
| EfficientNet-B0 骨干 | 可替换为更轻量网络 | 适配无人机嵌入式场景 |
| 车辆占用分割 | 无人机环境感知 | 扩展为多类语义分割 |
| LiDAR 真值深度 | AirSim 仿真提供真值 | 可直接使用 |

---

## 8. 依赖

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

- 无 CUDA 算子编译需求
- 无 mmdet3d / mmcv 依赖
- 代码简洁，适合作为 BEV 入门基线
