# 原始项目规划（存档）

> 以下为输入压缩包原有 README 的内容，保留团队项目规划与历史环境说明。
> 其中的目标架构、模型清单与部署设想不代表全部已经实现或验证；当前发布状态请以[根目录 README](../README.md)为准。相对链接仍沿用原文位置。

---

# 基于BEV的无人机环境感知与自主导航

## 项目简介

本项目旨在构建一个基于鸟瞰视角（BEV, Bird's Eye View）的无人机环境感知与自主导航系统。通过融合多摄像头图像输入，结合视觉语言模型（VLM）和大语言模型（LLM），实现对无人机周围环境的实时感知、语义理解与自主决策。

## 团队成员

| 姓名 | 角色 |
|------|------|
| 李丫馨 | |
| 钱森   | |
| 邢芸   | |
| 朱永强 | |

## 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.9 |
| 深度学习框架 | PyTorch 2.0+ |
| GPU加速 | CUDA 11.8 |
| 感知模型 | BEV感知（BEVDet / BEVFormer 等） |
| 视觉语言模型 | VLM（CLIP, Grounding DINO） |
| 大语言模型 | LLM（基于 Transformers） |
| 仿真平台 | AirSim |
| 计算机视觉 | OpenCV |

## 目录结构

```
BEV_Drone_Project/
├── data/                        # 数据集存放目录（已加入 .gitignore）
├── configs/                     # 配置文件
│   ├── __init__.py
│   └── base_config.yaml         # 基础配置：BEV/图像/相机参数
├── models/                      # 模型模块
│   ├── __init__.py
│   ├── bev/                     # BEV感知模型
│   │   └── __init__.py
│   ├── vlm/                     # 视觉语言模型 (CLIP, Grounding DINO)
│   │   └── __init__.py
│   └── llm/                     # 大语言模型
│       └── __init__.py
├── simulation/                  # AirSim 仿真相关代码
│   └── __init__.py
├── .gitignore                   # Git 忽略规则
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖清单
└── setup_env.ps1                # 一键环境配置脚本 (Windows)
```

## 快速开始

### 1. 环境要求

| 组件 | 最低版本 |
|------|----------|
| 操作系统 | Windows 10+ / Ubuntu 20.04+ |
| Python | 3.9 |
| CUDA | 11.8 |
| GPU 驱动 | ≥ 520.61.05 |
| Git | 2.0+ |

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd BEV_Drone_Project
```

### 3. 环境配置

#### 方式一：使用 conda（推荐，跨平台）

```bash
# 创建虚拟环境
conda create -n bev_drone python=3.9 -y

# 激活环境
conda activate bev_drone

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用 venv + 自动化脚本（Windows）

在 PowerShell 中执行：

```powershell
.\setup_env.ps1
```

该脚本会自动完成：创建虚拟环境 → 升级 pip → 安装所有依赖 → 验证安装。

> **注意**：如果 PowerShell 提示"无法加载文件，因为在此系统上禁止运行脚本"，请先执行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 方式三：手动 venv（通用）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 4. 验证安装

运行以下命令确认环境配置成功：

```bash
python -c "import torch; import cv2; import transformers; print('环境验证成功'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果输出中包含 `环境验证成功` 和 `CUDA available: True`，说明环境就绪。

### 5. 下载预训练模型权重

```bash
# 将模型权重文件放置到对应目录
# models/bev/    → BEV 模型权重 (.pth)
# models/vlm/    → VLM 模型权重
# models/llm/    → LLM 模型权重
```

> 模型权重文件已被 `.gitignore` 忽略，不会提交到 Git 仓库。

## 配置文件说明

[configs/base_config.yaml](configs/base_config.yaml) 包含项目的基础参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bev.height` | 200 | BEV网格高度 |
| `bev.width` | 200 | BEV网格宽度 |
| `image.height` | 256 | 输入图像高度 |
| `image.width` | 704 | 输入图像宽度 |
| `camera.num` | 4 | 相机数量 |

可通过修改此文件或创建新的 YAML 配置文件来调整参数。

## 主要依赖说明

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.0.1+cu118 | 深度学习框架（CUDA 11.8） |
| `torchvision` | 0.15.2+cu118 | 计算机视觉工具库 |
| `opencv-python` | 4.8.1.78 | 图像处理 |
| `transformers` | 4.35.2 | HuggingFace 模型库（LLM 支持） |
| `numpy` | 1.24.4 | 数值计算 |
| `pyyaml` | 6.0.1 | YAML 配置文件解析 |
| `airsim` | latest | 无人机仿真平台 |
| `CLIP` | GitHub | 视觉-语言对齐模型 |
| `Grounding DINO` | GitHub | 开放词汇目标检测 |

## 开发约定

- 代码风格：遵循 PEP 8
- 注释语言：中文
- 配置文件：统一使用 YAML 格式，存放于 [configs/](configs/) 目录
- 数据集：存放于 [data/](data/) 目录，不纳入版本控制
- 模型权重：存放于对应 `models/` 子目录，不纳入版本控制

## 常见问题

**Q: 安装时提示 CUDA 不可用？**
A: 请确认已安装 CUDA 11.8 工具包，且 GPU 驱动版本 ≥ 520.61。运行 `nvidia-smi` 检查。

**Q: AirSim 安装失败？**
A: Windows 下 AirSim 需要预编译的二进制文件。请确保系统已安装 Visual C++ Redistributable。

**Q: Grounding DINO 安装报错？**
A: 请确保已安装 `gcc` / `g++` 编译器（Linux）或 Visual Studio Build Tools（Windows）。
