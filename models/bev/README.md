# BEV 感知模块

## 参考基线

见 [docs/lss_analysis.md](../../docs/lss_analysis.md)

Lift-Splat-Shoot (ECCV 2020) 是 BEV 感知的开山之作，架构简洁、依赖少，是本模块的首选基线。

## 文件说明

| 文件 | 功能 |
|------|------|
| `lss.py` | 基于 LSS 架构的 BEV 感知骨架（Lift/Splat/Shoot 三阶段） |
| `__init__.py` | 模块入口 |

## 后续扩展

- [ ] 适配 AirSim 4 路相机输入 (当前 LSS 用 6 路 nuScenes)
- [ ] 将 EfficientNet-B0 替换为轻量化 backbone
- [ ] 扩展输出为多类语义分割（道路/障碍物/人员等）
- [ ] 集成深度真值辅助训练（AirSim 可提供）
