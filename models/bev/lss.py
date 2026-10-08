"""
基于 LSS (Lift-Splat-Shoot) 架构的 BEV 感知模型骨架。

参考: nv-tlabs/lift-splat-shoot (ECCV 2020)
"""

import torch
import torch.nn as nn


class CamEncode(nn.Module):
    """
    Lift 阶段: 图像编码器 + 深度预测

    输入: (B, N, 3, H, W) 多路相机图像
    输出: (B, N, D, H', W', C) 3D-aware 特征

    D: 深度 bin 数量
    H', W': 下采样后的特征图尺寸
    C: 特征通道数
    """

    def __init__(self, D: int = 41, C: int = 64):
        super().__init__()
        self.D = D  # 深度 bin 数 (原版: 41, 范围 4m~45m)
        self.C = C

        # ---- 特征提取主干 (原版: EfficientNet-B0) ----
        # TODO: 替换为适合无人机场景的轻量 backbone
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ---- 深度预测网络 ----
        self.depthnet = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.D, kernel_size=1),
            nn.Softmax(dim=1),  # 归一化为概率分布
        )

    def get_depth_dist(self, x: torch.Tensor) -> torch.Tensor:
        """预测逐像素深度分布，输出 (B*N, D, H', W')"""
        return self.depthnet(x)

    def get_depth_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        将特征与深度分布结合，生成 3D-aware 特征。
        输入: (B*N, C, H', W') 图像特征
        输出: (B*N, D*C, H', W')
        """
        # 提取特征
        trunk_feat = self.trunk(x)  # (B*N, C, H', W')
        # 预测深度
        depth = self.get_depth_dist(trunk_feat)  # (B*N, D, H', W')

        # 外积: 特征 × 深度 → 3D 感知特征
        # (B*N, C, 1, H', W') × (B*N, 1, D, H', W') → (B*N, C, D, H', W')
        return trunk_feat.unsqueeze(2) * depth.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, 3, H, W) 多相机图像拼接
        返回: (B*N, C*D, H', W') 3D-aware 特征
        """
        B, N, Ci, H, W = x.shape
        x = x.view(B * N, Ci, H, W)
        x = self.get_depth_feat(x)  # (B*N, C, D, H', W')
        x = x.view(B * N, self.C * self.D, x.shape[-2], x.shape[-1])
        return x


class BevEncode(nn.Module):
    """
    Shoot 阶段: BEV 特征编码器

    输入: (B, C, X, Y) 体素池化后的 BEV 特征
    输出: (B, outC, X, Y) BEV 语义分割

    原版: ResNet-18 类结构
    """

    def __init__(self, inC: int = 64, outC: int = 1):
        super().__init__()
        self.outC = outC

        # 基础卷积层
        self.trunk = nn.Sequential(
            nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(64, outC, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        x = self.up1(x)
        x = self.head(x)
        return x


class LiftSplatShoot(nn.Module):
    """
    LSS 主模型

    输入:
        imgs:     (B, N, 3, H, W)  多相机图像
        rots:     (B, N, 3, 3)     外参旋转
        trans:    (B, N, 3)        外参平移
        intrins:  (B, N, 3, 3)     内参矩阵
        post_rots:  (B, N, 3, 3)   增强旋转修正
        post_trans: (B, N, 3)      增强平移修正

    输出:
        (B, outC, X, Y) BEV 语义分割
    """

    def __init__(self, grid_conf: dict, outC: int = 1):
        super().__init__()
        self.grid_conf = grid_conf

        # 解析 BEV 栅格参数
        # X 轴
        xbound = grid_conf['xbound']  # [min, max, step]
        self.dx = xbound[2]
        self.nx = int((xbound[1] - xbound[0]) / self.dx)
        # Y 轴
        ybound = grid_conf['ybound']
        self.dy = ybound[2]
        self.ny = int((ybound[1] - ybound[0]) / self.dy)
        # Z 轴（深度）
        zbound = grid_conf['zbound']
        self.dz = zbound[2]
        self.nz = int((zbound[1] - zbound[0]) / self.dz)

        # 子模块
        self.camencode = CamEncode(D=self.nz)
        self.bevencode = BevEncode(inC=self.nz * 64, outC=outC)  # inC = D * C

        # ---- 预计算相机视锥 ----
        self._create_frustum()

    def _create_frustum(self):
        """
        创建相机坐标系下的视锥点云模板。
        形状: (D, H_f, W_f, 3)
        """
        # 深度 bin
        zbound = self.grid_conf['zbound']
        d_coords = torch.arange(*zbound, dtype=torch.float)  # (D,)
        d_coords = d_coords.view(-1, 1, 1).expand(-1, 8, 22)  # (D, H_f, W_f)

        # 像素坐标 (简化: 假设下采样后 H_f=8, W_f=22)
        x_coords = torch.arange(22, dtype=torch.float)  # (W_f,)
        y_coords = torch.arange(8, dtype=torch.float)    # (H_f,)
        x_coords, y_coords = torch.meshgrid(x_coords, y_coords, indexing='xy')

        self.frustum = torch.stack([x_coords, y_coords, d_coords], dim=-1)  # (D, H_f, W_f, 3)
        self.frustum = nn.Parameter(self.frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        将相机坐标系下的视锥点变换到自车坐标系。

        rots:       (B, N, 3, 3)
        trans:      (B, N, 3)
        intrins:    (B, N, 3, 3)
        post_rots:  (B, N, 3, 3)
        post_trans: (B, N, 3)

        返回: (B, N, D, H_f, W_f, 3) 自车坐标系下的 3D 点
        """
        B, N, _, _ = rots.shape
        D, H_f, W_f, _ = self.frustum.shape

        # Step 1: 像素坐标 → 归一化相机坐标 (用内参逆变换)
        points = self.frustum.to(rots.device)  # (D, H_f, W_f, 3)
        # 简化: 假设内参 K 已知，做逆投影
        # 实际需: points = K^{-1} @ frustum

        # Step 2: 恢复到数据增强前的坐标
        # 实际需: points = post_rots^{-1} @ (points - post_trans)

        # Step 3: 相机坐标 → 自车坐标 (用外参)
        # points = rots @ points + trans

        # Mock 返回值, 实际实现需展开
        geometry = points.unsqueeze(0).unsqueeze(0).expand(B, N, D, H_f, W_f, 3)
        return geometry

    def voxel_pooling(self, geom_feats: torch.Tensor) -> torch.Tensor:
        """
        体素池化: 将 3D 特征聚合到 BEV 栅格中。

        geom_feats: (B*N, D*H_f*W_f, C+3)
            每个点: [x, y, z, feat_0, ..., feat_{C-1}]

        返回: (B, C, X, Y)
        """
        # 简化实现: 使用 scatter 聚合
        # 实际 LSS 使用了排序+累积和优化，这里给出概念骨架
        B = geom_feats.shape[0] // 6  # 假设 6 相机

        # 将 3D 坐标映射到 BEV 栅格索引
        # 聚合特征
        # 返回 (B, C, X, Y)
        C_sum = 64
        return torch.zeros(B, C_sum, self.nx, self.ny, device=geom_feats.device)

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        Lift + Splat 阶段:
        1. CamEncode: 图像 → 3D-aware 特征
        2. get_geometry: 计算 3D 点位置
        3. voxel_pooling: 聚合到 BEV 栅格
        """
        # 1. Lift: 图像编码
        x = self.camencode(x)  # (B*N, C*D, H_f, W_f)

        # 2. 计算 3D 点几何位置
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)

        # 3. Splat: 体素池化
        x = self.voxel_pooling(x)  # TODO: 结合 geom 做实际池化

        return x

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """
        完整前向传播。

        imgs:       (B, N, 3, H, W)
        rots:       (B, N, 3, 3)
        trans:      (B, N, 3)
        intrins:    (B, N, 3, 3)
        post_rots:  (B, N, 3, 3)
        post_trans: (B, N, 3)

        返回: (B, outC, X, Y) BEV 分割图
        """
        # Lift + Splat
        voxels = self.get_voxels(imgs, rots, trans, intrins, post_rots, post_trans)

        # Shoot: BEV 编码
        output = self.bevencode(voxels)

        return output
