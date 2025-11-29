import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# =========================================================================
# AFPN (Asymptotic Feature Pyramid Network) 公共模块
# 适配: CIFAR-100 (小分辨率场景)
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

class AFPNUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(AFPNUpsample, self).__init__()
        self.upsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.upsample(x)

class AFPNDownsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x2, self).__init__()
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 2, 2, 0)
        )
    def forward(self, x): return self.downsample(x)

class AFPNDownsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x4, self).__init__()
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 4, 4, 0)
        )
    def forward(self, x): return self.downsample(x)

class AFPNDownsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x8, self).__init__()
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 8, 8, 0)
        )
    def forward(self, x): return self.downsample(x)

class AFPNASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(AFPNASFF_2, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8
        self.weight_level_1 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = AFPNBasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        return self.conv(fused_out_reduced)

class AFPNASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(AFPNASFF_3, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8
        self.weight_level_1 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = AFPNBasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
        return self.conv(fused_out_reduced)

class AFPNBlockBody(nn.Module):
    def __init__(self, channels):
        super(AFPNBlockBody, self).__init__()
        # L2 level (2 inputs)
        self.blocks_scalezero1 = AFPNBasicConv(channels[0], channels[0], 1)
        self.blocks_scaleone1 = AFPNBasicConv(channels[1], channels[1], 1)
        self.downsample_scalezero1_2 = AFPNDownsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = AFPNUpsample(channels[1], channels[0], scale_factor=2)
        self.asff_scalezero1 = AFPNASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = AFPNASFF_2(inter_dim=channels[1])

        # L3 level (3 inputs)
        self.blocks_scalezero2 = nn.Sequential(AFPNBasicBlock(channels[0], channels[0]), AFPNBasicBlock(channels[0], channels[0]))
        self.blocks_scaleone2 = nn.Sequential(AFPNBasicBlock(channels[1], channels[1]), AFPNBasicBlock(channels[1], channels[1]))
        
        self.downsample_scalezero2_2 = AFPNDownsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = AFPNDownsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = AFPNDownsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = AFPNUpsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = AFPNUpsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = AFPNUpsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = AFPNASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = AFPNASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = AFPNASFF_3(inter_dim=channels[2])
        
        # Final blocks
        self.blocks_scalezero3 = nn.Sequential(AFPNBasicBlock(channels[0], channels[0]), AFPNBasicBlock(channels[0], channels[0]))
        self.blocks_scaleone3 = nn.Sequential(AFPNBasicBlock(channels[1], channels[1]), AFPNBasicBlock(channels[1], channels[1]))
        self.blocks_scaletwo3 = nn.Sequential(AFPNBasicBlock(channels[2], channels[2]), AFPNBasicBlock(channels[2], channels[2]))

    def forward(self, x):
        x0, x1, x2 = x

        # Level 2 Fusion
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        scalezero_l2 = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone_l2 = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        # Level 3 Fusion
        x0_l2 = self.blocks_scalezero2(scalezero_l2)
        x1_l2 = self.blocks_scaleone2(scaleone_l2)
        
        scalezero_l3 = self.asff_scalezero2(x0_l2, self.upsample_scaleone2_2(x1_l2), self.upsample_scaletwo2_4(x2))
        scaleone_l3 = self.asff_scaleone2(self.downsample_scalezero2_2(x0_l2), x1_l2, self.upsample_scaletwo2_2(x2))
        scaletwo_l3 = self.asff_scaletwo2(self.downsample_scalezero2_4(x0_l2), self.downsample_scaleone2_2(x1_l2), x2)

        out0 = self.blocks_scalezero3(scalezero_l3)
        out1 = self.blocks_scaleone3(scaleone_l3)
        out2 = self.blocks_scaletwo3(scaletwo_l3)

        return out0, out1, out2

class AFPN(nn.Module):
    """
    AFPN 主模块
    输入: [f1, f2, f3] (Channels List)
    输出: 增强后的特征 (Channel = out_channels)
    """
    def __init__(self, in_channels, out_channels):
        super(AFPN, self).__init__()
        self.in_channels = in_channels
        # 压缩通道数以减少计算量 (例如全部压缩到 1/4 或 1/8)
        # 这里我们设定内部维度为各个通道数的 1/4，最小为32
        self.inner_dims = [max(32, c // 4) for c in in_channels]
        
        # 1. 输入投影层
        self.conv0 = AFPNBasicConv(in_channels[0], self.inner_dims[0], 1)
        self.conv1 = AFPNBasicConv(in_channels[1], self.inner_dims[1], 1)
        self.conv2 = AFPNBasicConv(in_channels[2], self.inner_dims[2], 1)

        # 2. AFPN 主体 (3尺度融合)
        self.body = AFPNBlockBody(self.inner_dims)

        # 3. 输出投影层 (融合回 out_channels)
        # 我们将 body 的三个输出全部对齐到 f3 的尺寸，然后 concat 或只取 f3
        # 为了简单且高效，我们只取最深层的输出 out2 (对应 f3 尺度)，并映射回 out_channels
        self.out_conv = AFPNBasicConv(self.inner_dims[2], out_channels, 1)

    def forward(self, x):
        f1, f2, f3 = x
        
        # 投影
        f1_inner = self.conv0(f1)
        f2_inner = self.conv1(f2)
        f3_inner = self.conv2(f3)

        # 融合
        out0, out1, out2 = self.body([f1_inner, f2_inner, f3_inner])

        # 输出 (取与 f3 尺度相同的 out2)
        out = self.out_conv(out2)
        return out