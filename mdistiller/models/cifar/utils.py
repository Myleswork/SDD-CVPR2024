import torch.nn as nn
import torch
import torch.nn.functional as F


class SPP(nn.Module):
    def __init__(self, M=None):
        super(SPP, self).__init__()
        self.pooling_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

        self.M = M
        print(self.M)

    def forward(self, x):
        x_4x4 = self.pooling_4x4(x)
        x_2x2 = self.pooling_2x2(x_4x4)
        x_1x1 = self.pooling_1x1(x_4x4)

        x_4x4_flatten = torch.flatten(x_4x4, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        if self.M == '[1,2,4]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten, x_4x4_flatten), dim=2)
        elif self.M == '[1,2]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten), dim=2)
        elif self.M=='[1]':
            x_feature = x_1x1_flatten
        else:
            raise NotImplementedError('ERROR M')

        x_strength = x_feature.permute((2, 0, 1))
        x_strength = torch.mean(x_strength, dim=2)


        return x_feature, x_strength

class DynamicDecoupling(nn.Module):
    def __init__(self, in_channels, num_regions=21):
        """
        in_channels: 输入特征图的通道数 (例如 ResNet32x4 的最后一层)
        num_regions: 你希望解耦出多少个区域？
                     原版 SDD [1,2,4] 对应 1+4+16=21 个区域。
                     为了公平对比，我们可以设为 21。
        """
        super(DynamicDecoupling, self).__init__()
        self.num_regions = num_regions
        
        # === 动态滤波器生成器 (Generator) ===
        # 参考 4.pdf 的思想，这是一个轻量级的网络，从输入特征生成权重
        # 结构：Conv(1x1) -> BN -> ReLU -> Conv(1x1) -> Softmax/Sigmoid
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_regions, kernel_size=1, bias=False),
        )
        
        # 初始化，让初始状态接近平均分布，避免训练初期梯度爆炸
        for m in self.mask_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # ================== 新增检测代码 ==================
        # 使用 getattr 检查是否已经打印过，防止刷屏
        if not getattr(self, 'has_printed', False):
            print("\n" + "="*50)
            print("[SUCCESS] DynamicDecoupling module is successfully CALLED!")
            print(f"          Input Shape: {x.shape}")
            print("="*50 + "\n")
            self.has_printed = True
        # ==================================================
        # x shape: [B, C, H, W] (例如 [64, 256, 8, 8])
        B, C, H, W = x.shape
        
        # 1. 生成动态空间掩码 (Dynamic Spatial Masks)
        # masks shape: [B, K, H, W], 其中 K = num_regions
        raw_masks = self.mask_generator(x)
        
        # 使用 Spatial Softmax 确保每个掩码在空间上的权重之和为 1
        # 这样它的物理意义就变成了“加权平均池化” (Weighted Average Pooling)
        masks = F.softmax(raw_masks.view(B, self.num_regions, -1), dim=2) # [B, K, H*W]
        masks = masks.view(B, self.num_regions, H, W)
        
        # 2. 特征聚合 (Feature Aggregation)
        # 我们需要计算每个区域的特征向量： f_k = Sum(Mask_k * X)
        
        # Reshape for matrix multiplication
        x_flat = x.view(B, C, H * W)          # [B, C, N]
        masks_flat = masks.view(B, self.num_regions, H * W) # [B, K, N]
        
        # 矩阵乘法: [B, C, N] @ [B, N, K] -> [B, C, K]
        # transpose(1, 2) 把 masks 变成 [B, N, K]
        x_feature = torch.bmm(x_flat, masks_flat.transpose(1, 2)) # [B, C, K]
        
        # 3. 计算 x_strength (用于 SDD 的辅助项，保持原逻辑)
        # 这里我们取所有区域的平均值作为全局强度的近似
        x_strength = x_feature.mean(dim=2) # [B, C]

        # SDD 期望的输出格式: x_feature [B, C, K], x_strength [B, C]
        return x_feature, x_strength, masks
