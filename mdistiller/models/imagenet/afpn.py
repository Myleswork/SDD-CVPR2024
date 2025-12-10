import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# =========================================================================
# AFPN (Asymptotic Feature Pyramid Network) - ImageNet 4-Scale Version
# 适配: ImageNet (ResNet18/34/50/101, MobileNetV2 等 4 阶段模型)
# =========================================================================

class AFPNBasicConv(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size, stride=1, pad=None):
        super(AFPNBasicConv, self).__init__()
        if pad is None:
            pad = (kernel_size - 1) // 2 if kernel_size else 0
        self.conv = nn.Conv2d(filter_in, filter_out, kernel_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(filter_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AFPNBasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        super(AFPNBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class AFPNBlockBody(nn.Module):
    def __init__(self, channels):
        super(AFPNBlockBody, self).__init__()
        # 确保输入是 4 个尺度
        assert len(channels) == 4, f"ImageNet AFPN expects 4 scales, got {len(channels)}"
        
        # Level 1: 1x1 Conv 调整通道
        self.blocks_scale0_1 = AFPNBasicConv(channels[0], channels[0], 1)
        self.blocks_scale1_1 = AFPNBasicConv(channels[1], channels[1], 1)
        self.blocks_scale2_1 = AFPNBasicConv(channels[2], channels[2], 1)
        self.blocks_scale3_1 = AFPNBasicConv(channels[3], channels[3], 1)

        # 融合操作 (ASFF 思想简化版: 统一对齐到 Scale 2)
        # Scale 0 (Start) -> Scale 2: Downsample 4x
        self.down0_to_2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[2], 4, 4, 0, bias=False),
            nn.BatchNorm2d(channels[2]), nn.ReLU()
        )
        # Scale 1 -> Scale 2: Downsample 2x
        self.down1_to_2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 2, 2, 0, bias=False),
            nn.BatchNorm2d(channels[2]), nn.ReLU()
        )
        # Scale 3 -> Scale 2: Upsample 2x
        self.up3_to_2 = nn.Sequential(
            AFPNBasicConv(channels[3], channels[2], 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 融合后的处理 (Level 2)
        self.fusion_block = nn.Sequential(
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2])
        )

    def forward(self, x):
        x0, x1, x2, x3 = x # 接收 4 个输入
        
        # 1. 特征变换
        x0 = self.blocks_scale0_1(x0)
        x1 = self.blocks_scale1_1(x1)
        x2 = self.blocks_scale2_1(x2)
        x3 = self.blocks_scale3_1(x3)

        # 2. 空间对齐 (Target: Scale 2)
        x0_aligned = self.down0_to_2(x0)
        x1_aligned = self.down1_to_2(x1)
        x2_aligned = x2
        x3_aligned = self.up3_to_2(x3)

        # 3. 融合 (Add)
        fused = x0_aligned + x1_aligned + x2_aligned + x3_aligned
        
        # 4. 提炼
        out = self.fusion_block(fused)
        
        return out

class AFPN(nn.Module):
    """
    AFPN 主模块 (支持 4 尺度输入)
    in_channels: [c1, c2, c3, c4]
    out_channels: 最终输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super(AFPN, self).__init__()
        self.in_channels = in_channels
        
        # 内部维度: 统一压缩到 c3 的 1/4 或 1/8，或者固定值
        # ImageNet 常见配置: 统一压缩到 256
        inner_dim = 256
        self.inner_dims = [inner_dim] * 4 
        
        # 输入投影
        self.conv0 = AFPNBasicConv(in_channels[0], inner_dim, 1)
        self.conv1 = AFPNBasicConv(in_channels[1], inner_dim, 1)
        self.conv2 = AFPNBasicConv(in_channels[2], inner_dim, 1)
        self.conv3 = AFPNBasicConv(in_channels[3], inner_dim, 1)

        # AFPN Body
        self.body = AFPNBlockBody(self.inner_dims)
        
        # 输出投影 (如果需要调整回特定维度)
        self.out_conv = AFPNBasicConv(inner_dim, out_channels, 1)

    def forward(self, feats):
        # ImageNet ResNet/MobileNet 传进来通常是 list/tuple
        f0, f1, f2, f3 = feats
        
        # 投影到内部维度
        x0 = self.conv0(f0)
        x1 = self.conv1(f1)
        x2 = self.conv2(f2)
        x3 = self.conv3(f3)
        
        # 融合
        out_fused = self.body([x0, x1, x2, x3])
        
        # 输出调整
        out = self.out_conv(out_fused)
        
        return out