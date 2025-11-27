from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
from mdistiller.models.cifar.utils import SPP, DynamicDecoupling
import torch
from collections import OrderedDict


__all__ = ["resnet"]

def AFPNBasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    # 如果未提供pad参数，则依据卷积核尺寸自动计算填充量
    if not pad:
        # 当卷积核尺寸为奇数时，确保两侧填充值均等
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        # 若用户给定pad，则直接采用该数值
        pad = pad

    # 利用nn.Sequential构建一个有序层序列
    return nn.Sequential(OrderedDict([
        # 添加卷积操作
        ("conv", nn.Conv2d(in_channels=filter_in, out_channels=filter_out,
                           kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        # 添加批标准化操作
        ("bn", nn.BatchNorm2d(num_features=filter_out)),
        # 应用ReLU激活函数（原地操作）
        ("relu", nn.ReLU(inplace=True)),
    ]))

class AFPNBasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        # 调用父类 nn.Module 的构造方法
        super(AFPNBasicBlock, self).__init__()
        # 初始化第一层卷积
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=3, padding=1)
        # 设置第一层批归一化
        self.bn1 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)
        # 初始化ReLU激活模块
        self.relu = nn.ReLU(inplace=True)
        # 构造第二层卷积
        self.conv2 = nn.Conv2d(in_channels=filter_out, out_channels=filter_out, kernel_size=3, padding=1)
        # 配置第二层批归一化
        self.bn2 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)

    def forward(self, x):
        # 保存输入特征以便后续执行残差相加
        residual = x
        # 第一阶段：卷积 -> 批归一化 -> ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二阶段：卷积 -> 批归一化
        out = self.conv2(out)
        out = self.bn2(out)
        # 将原始输入添加回输出，形成残差连接
        out += residual
        # 最后进行ReLU激活，输出最终结果
        out = self.relu(out)
        return out

class AFPNUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(AFPNUpsample, self).__init__()
        # 构建上采样模块：先用1x1卷积调整通道，再采用双线性插值进行放大
        self.upsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 1),  # 采用1x1卷积调整通道
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')  # 双线性插值上采样
        )

    def forward(self, x):
        # 利用上采样模块处理输入数据
        x = self.upsample(x)
        return x

class AFPNDownsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x2, self).__init__()
        # 构建下采样模块，通过步长为2的卷积实现2倍降采样
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 2, 2, 0)  # 使用2x2卷积，步长为2
        )

    def forward(self, x):
        # 利用下采样模块降低特征图分辨率
        x = self.downsample(x)
        return x

class AFPNDownsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x4, self).__init__()
        # 构建下采样模块，通过步长为4的卷积实现4倍降采样
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 4, 4, 0)  # 采用4x4卷积实现步长为4的下采样
        )

    def forward(self, x):
        # 使用下采样模块对输入进行处理
        x = self.downsample(x)
        return x

class AFPNDownsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFPNDownsample_x8, self).__init__()
        # 构建下采样模块，通过步长为8的卷积实现8倍降采样
        self.downsample = nn.Sequential(
            AFPNBasicConv(in_channels, out_channels, 8, 8, 0)  # 利用8x8卷积，步长设为8
        )

    def forward(self, x):
        # 经由下采样模块降低输入分辨率
        x = self.downsample(x)
        return x

class AFPNASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(AFPNASFF_2, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化两个通道压缩卷积层，用于特征降维
        self.weight_level_1 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建权重融合层，用以整合压缩后的特征
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        # 构造融合后处理卷积层，进一步细化融合特征
        self.conv = AFPNBasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        # 分别对两个输入特征图进行通道压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        # 拼接压缩后的特征图
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        # 利用融合层计算各级别特征的权重分布
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 根据计算得到的权重，对两个特征图进行加权合并
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]
        # 经过卷积层进一步整合融合后的特征
        out = self.conv(fused_out_reduced)
        return out

class AFPNASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(AFPNASFF_3, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化三个降维卷积层，对应三个输入特征图
        self.weight_level_1 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建融合权重层，将三个降维后的特征图整合在一起
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        # 构造后续处理卷积层，对加权融合后的特征进行优化
        self.conv = AFPNBasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        # 分别对三个输入进行通道压缩
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        # 拼接所有压缩后的特征图
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # 计算三个特征图的权重分布
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 按权重比例加权合并三个输入特征图
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
        # 经过卷积操作进一步融合特征
        out = self.conv(fused_out_reduced)
        return out

class AFPNASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(AFPNASFF_4, self).__init__()
        self.inter_dim = inter_dim
        compress_c = 8

        # 初始化四个降维卷积层，分别处理四个输入特征图
        self.weight_level_0 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = AFPNBasicConv(self.inter_dim, compress_c, 1, 1)

        # 构建融合权重计算层，用以整合四个降维特征
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        # 构造融合后特征处理卷积层
        self.conv = AFPNBasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        # 分别对四个输入特征图进行降维处理
        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        # 拼接所有降维后的特征图
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # 计算各层特征的权重
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 根据权重对四个特征图进行加权合并
        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]
        # 用卷积层进一步整合融合结果
        out = self.conv(fused_out_reduced)
        return out

import torch.nn as nn

class AFPNBlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        # 调用基类 nn.Module 的初始化函数
        super(AFPNBlockBody, self).__init__()
        # 设置四个1x1卷积模块，用于调整各层特征图的通道数量
        self.blocks_scalezero1 = nn.Sequential(
            AFPNBasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            AFPNBasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            AFPNBasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            AFPNBasicConv(channels[3], channels[3], 1),
        )

        # 构建2倍下采样和上采样模块
        self.downsample_scalezero1_2 = AFPNDownsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = AFPNUpsample(channels[1], channels[0], scale_factor=2)

        # 初始化两个ASFF_2模块用于特征融合
        self.asff_scalezero1 = AFPNASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = AFPNASFF_2(inter_dim=channels[1])

        # 构建两个残差块序列
        self.blocks_scalezero2 = nn.Sequential(
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
        )

        # 配置多个降采样和上采样操作模块
        self.downsample_scalezero2_2 = AFPNDownsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = AFPNDownsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = AFPNDownsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = AFPNUpsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = AFPNUpsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = AFPNUpsample(channels[2], channels[0], scale_factor=4)

        # 初始化三个ASFF_3模块用于多尺度特征融合
        self.asff_scalezero2 = AFPNASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = AFPNASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = AFPNASFF_3(inter_dim=channels[2])

        # 构造三个残差块序列
        self.blocks_scalezero3 = nn.Sequential(
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
        )

        # 配置更多的降采样和上采样模块以匹配不同尺度
        self.downsample_scalezero3_2 = AFPNDownsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = AFPNDownsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = AFPNDownsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = AFPNUpsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = AFPNDownsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = AFPNDownsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = AFPNUpsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = AFPNUpsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = AFPNDownsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = AFPNUpsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = AFPNUpsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = AFPNUpsample(channels[3], channels[2], scale_factor=2)

        # 初始化四个ASFF_4模块用于更高层次的特征融合
        self.asff_scalezero3 = AFPNASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = AFPNASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = AFPNASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = AFPNASFF_4(inter_dim=channels[3])

        # 构造四个残差块序列以进一步提炼融合特征
        self.blocks_scalezero4 = nn.Sequential(
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
            AFPNBasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
            AFPNBasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
            AFPNBasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            AFPNBasicBlock(channels[3], channels[3]),
            AFPNBasicBlock(channels[3], channels[3]),
            AFPNBasicBlock(channels[3], channels[3]),
            AFPNBasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        # 将输入的多尺度特征图依次拆分
        x0, x1, x2, x3 = x

        # 通过1x1卷积层对各尺度特征图进行通道调整
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        # ASFF_2特征融合：将x0与x1进行上下采样后融合
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        # 通过残差块序列进一步提炼特征
        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        # ASFF_3特征融合：将经过不同采样的x0、x1和x2融合
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        # 再次通过残差块序列进一步处理融合后的特征
        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        # ASFF_4特征融合：融合x0、x1、x2和x3经过不同尺度采样的结果
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)

        # 通过最后的残差块序列进一步优化融合结果
        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        # 输出最终融合后的多尺度特征图
        return scalezero, scaleone, scaletwo, scalethree

class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],  # 各输入特征图的通道数列表
                 out_channels=256):  # 最终输出特征图的通道数
        super(AFPN, self).__init__()

        # 配置是否启用半精度(fp16)运算
        self.fp16_enabled = False

        # 使用1x1卷积层对输入特征图通道数进行压缩
        self.conv0 = AFPNBasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = AFPNBasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = AFPNBasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = AFPNBasicConv(in_channels[3], in_channels[3] // 8, 1)

        # 利用BlockBody模块实现多尺度特征融合
        self.body = nn.Sequential(
            AFPNBlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        # 通过1x1卷积层将融合后的特征图调整至统一的out_channels
        self.conv00 = AFPNBasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = AFPNBasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = AFPNBasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = AFPNBasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)  # 利用池化生成额外的下采样特征图

        # 初始化各层权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 针对卷积层
                nn.init.xavier_normal_(m.weight, gain=0.02)  # 使用Xavier正态分布进行初始化
            elif isinstance(m, nn.BatchNorm2d):  # 针对批归一化层
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # 权重采用正态分布初始化
                torch.nn.init.constant_(m.bias.data, 0.0)  # 偏置初始化为0

    def forward(self, x):
        # 拆分输入的多尺度特征图
        x0, x1, x2, x3 = x

        # 通过1x1卷积层对输入特征图进行通道压缩
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # 经过BlockBody模块实现多尺度特征融合
        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        # 用1x1卷积层将融合后的特征图调整为统一的输出通道数
        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        # 利用池化层生成额外的下采样特征
        out4 = self.conv44(out3)

        # 返回最终的多尺度融合结果
        return out0, out1, out2, out3, out4


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet_SDD(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10, M=None):
        super(ResNet_SDD, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model

        if block_name.lower() == "basicblock":
            assert (
                depth - 2
            ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters
        # self.spp=SPP(M=M)
        self.class_num=num_classes
        #动态滤波器模块
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters
        self.class_num = num_classes

        #动态滤波器实现可插拔
        m_str = str(M).replace(" ", "") if M is not None else ""
        if m_str.startswith('dynamic'):
            # === 模式 A: 动态滤波器解耦 ===
            print(f"Using Dynamic Decoupling (M={M})")
            # 计算输入通道数: filters * expansion (BasicBlock=1, Bottleneck=4)
            final_channels = num_filters[3] * block.expansion
            # 默认 21 个区域 (对应 [1,2,4] 的总数)，也可以解析字符串 dynamic-10 来指定
            self.spp = DynamicDecoupling(in_channels=final_channels, num_regions=21)
            
        else:
            # === 模式 B: 原始 SDD (SPP) ===
            print(f"Using Standard SPP (M={M})")
            self.spp = SPP(M=M)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(
            block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1))
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        # x_spp,x_strength = self.spp(x)
        spp_out = self.spp(x)
        if len(spp_out) == 3:
            x_spp, x_strength, masks = spp_out
        else:
            x_spp, x_strength= spp_out
            masks = None

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = avg

        return out, patch_score, masks, x  #新增返回原始特征图

class ResNet_AFPN_SDD(nn.Module):
    """
    集成 AFPN 的 SDD ResNet 版本。
    """
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10, M=None):
        super(ResNet_AFPN_SDD, self).__init__()
        
        # 1. 初始化基础 ResNet 结构 (直接复用原有逻辑)
        if block_name.lower() == "basicblock":
            assert (depth - 2) % 6 == 0, "Depth error"
            n = (depth - 2) // 6
            block = BasicBlock # 这里使用的是 ResNet 原有的 BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (depth - 2) % 9 == 0, "Depth error"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name should be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        
        # 2. 初始化 AFPN
        # 获取各层通道数
        c1 = num_filters[1] * block.expansion
        c2 = num_filters[2] * block.expansion
        c3 = num_filters[3] * block.expansion
        
        # AFPN 将融合 f1, f2, f3 以及 f3的下采样(f4)
        # 输出维度我们将保持为 c3，以便后续分类器复用
        self.afpn = AFPN(in_channels=[c1, c2, c3, c3], out_channels=c3)
        
        # 3. 分类器与 SDD 模块
        # 注意：AFPN 输出通道数为 c3
        self.fc = nn.Linear(c3, num_classes) 
        self.stage_channels = num_filters
        self.spp = SPP(M=M)
        self.class_num = num_classes

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 直接复制 ResNet_SDD 或 ResNet 中的 _make_layer 代码
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 基础特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x, f1_pre = self.layer1(x) # f1
        f1 = x
        x, f2_pre = self.layer2(x) # f2
        f2 = x
        x, f3_pre = self.layer3(x) # f3
        f3 = x

        # === AFPN 特征融合 ===
        # 构造第4层输入 (f3 的下采样)
        f4 = F.avg_pool2d(f3, kernel_size=2, stride=2)
        # 进行融合，返回 [out0, out1, out2, out3, out4]
        # 我们主要使用与 f3 尺度相同的 out2 (索引为2)
        afpn_outs = self.afpn([f1, f2, f3, f4])
        feature_enhanced = afpn_outs[2] 
        # ===================

        # SDD 流程 (使用增强后的特征)
        x_spp, x_strength = self.spp(feature_enhanced)

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        # 全局分类 (也可以使用增强后的特征 feature_enhanced)
        x_global = self.avgpool(feature_enhanced)
        avg = x_global.reshape(x_global.size(0), -1)
        out = self.fc(avg)

        feats = {}
        # 这里你可以决定返回原始特征还是 AFPN 特征
        feats["feats"] = [f0, f1, f2, f3] 
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = avg
        # 也可以把增强特征存入，方便调试
        feats["afpn_feat"] = feature_enhanced

        return out, patch_score

class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10,M=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = avg

        return out, feats




def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], "basicblock", **kwargs)




def resnet8_sdd(**kwargs):
    return ResNet_SDD(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14_sdd(**kwargs):
    return ResNet_SDD(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20_sdd(**kwargs):
    return ResNet_SDD(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet32_sdd(**kwargs):
    return ResNet_SDD(32, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet44_sdd(**kwargs):
    return ResNet_SDD(44, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56_sdd(**kwargs):
    return ResNet_SDD(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet110_sdd(**kwargs):
    return ResNet_SDD(110, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8x4_sdd(**kwargs):
    return ResNet_SDD(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4_sdd(**kwargs):
    return ResNet_SDD(32, [32, 64, 128, 256], "basicblock", **kwargs)

# === 新增：AFPN 版本的工厂函数 ===
def resnet32x4_afpn_sdd(**kwargs):
    # 调用新的 ResNet_AFPN_SDD 类
    return ResNet_AFPN_SDD(32, [32, 64, 128, 256], "basicblock", **kwargs)

def resnet8x4_afpn_sdd(**kwargs):
    return ResNet_AFPN_SDD(8, [32, 64, 128, 256], "basicblock", **kwargs)



if __name__ == "__main__":
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnet8x4(num_classes=20)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)
