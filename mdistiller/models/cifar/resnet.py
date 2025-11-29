from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
from mdistiller.models.cifar.utils import SPP, DynamicDecoupling
import torch
from collections import OrderedDict


__all__ = ["resnet"]

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

        x_global = self.avgpool(f3)
        avg = x_global.reshape(x_global.size(0), -1)
        out = self.fc(avg)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = avg

        return out, patch_score, masks, f3  #新增返回原始特征图


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

# =========================================================================
# AFPN 模块实现 (已重命名解决冲突)
# 来源: 1、 渐进式非相邻特征融合(1区 2024).py
# =========================================================================
class AFPNBasicConv(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size, stride=1, pad=None):
        super().__init__()
        if pad is None:
            pad = (kernel_size - 1) // 2 if kernel_size else 0
        self.conv = nn.Conv2d(filter_in, filter_out, kernel_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(filter_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AFPNBasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        super().__init__()
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

class AFPN(nn.Module):
    """
    简化的 AFPN 核心模块，适配 CIFAR ResNet
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 通道对齐 (压缩到 1/8 或其他比例)
        self.conv0 = AFPNBasicConv(in_channels[0], in_channels[0], 1) # f1
        self.conv1 = AFPNBasicConv(in_channels[1], in_channels[1], 1) # f2
        self.conv2 = AFPNBasicConv(in_channels[2], in_channels[2], 1) # f3
        
        # 2. 融合层 (这里简化实现，使用 ASFF 思想)
        # 将 f1(32x), f2(16x) 下采样/上采样 对齐到 f3(8x) 的尺寸
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[2], 4, 4, 0, bias=False), # 32->8 (4x down)
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[2], 2, 2, 0, bias=False), # 16->8 (2x down)
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU()
        )
        
        # 3. 融合后的处理
        self.fusion_conv = nn.Sequential(
            AFPNBasicBlock(in_channels[2], in_channels[2]),
            AFPNBasicConv(in_channels[2], out_channels, 1)
        )

    def forward(self, feats):
        f1, f2, f3 = feats
        
        # 空间对齐：全部对齐到 f3 (8x8)
        f1_aligned = self.down1(f1)
        f2_aligned = self.down2(f2)
        
        # 加权融合 (简单相加，利用 BN 自动学习缩放)
        fused = f1_aligned + f2_aligned + f3
        
        # 进一步提炼
        out = self.fusion_conv(fused)
        return out

# =========================================================================
# 新模型：集成 AFPN 的 ResNet
# =========================================================================
class ResNet_AFPN_SDD(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10, M=None):
        super(ResNet_AFPN_SDD, self).__init__()
        # 复用 ResNet_SDD 的初始化逻辑
        # ... (这里省略重复代码，实际写的时候请复制 ResNet_SDD 的 __init__ 内容) ...
        # 注意：为了节省篇幅，建议直接继承 ResNet_SDD，或者完整复制一遍
        
        # --- 假设你完整复制了 ResNet_SDD 的内容，下面是新增部分 ---
        if block_name.lower() == "basicblock":
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            n = (depth - 2) // 9
            block = Bottleneck
            
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        self.stage_channels = num_filters
        self.class_num = num_classes
        
        # === 核心修改 1: 初始化 AFPN ===
        c1 = num_filters[1] * block.expansion
        c2 = num_filters[2] * block.expansion
        c3 = num_filters[3] * block.expansion
        
        # 将 f1, f2, f3 融合，输出维度保持 c3
        self.afpn = AFPN(in_channels=[c1, c2, c3], out_channels=c3)
        
        # 分类器和 SDD
        self.fc = nn.Linear(c3, num_classes)
        self.spp = SPP(M=M) # 使用标准 SPP，不用 Dynamic
        # =================================

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 复制 _make_layer ...
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # f0

        x, f1_pre = self.layer1(x) # f1
        f1 = x
        x, f2_pre = self.layer2(x) # f2
        f2 = x
        x, f3_pre = self.layer3(x) # f3
        f3 = x # 8x8

        # === 核心修改 2: 使用 AFPN 增强特征 ===
        # 将 f1, f2, f3 送入 AFPN，得到增强后的 8x8 特征
        feature_enhanced = self.afpn([f1, f2, f3])
        # ====================================

        # === 核心修改 3: SDD 使用增强特征 ===
        x_spp, x_strength = self.spp(feature_enhanced)

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        # 全局分类：也建议使用增强特征，提升主线性能
        x_global = self.avgpool(feature_enhanced)
        avg = x_global.reshape(x_global.size(0), -1)
        out = self.fc(avg)

        # 返回值保持与标准 SDD 一致 (不需要 masks)
        return out, patch_score, None, feature_enhanced

# 注册新模型
def resnet8x4_afpn_sdd(**kwargs):
    return ResNet_AFPN_SDD(8, [32, 64, 128, 256], "basicblock", **kwargs)


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
