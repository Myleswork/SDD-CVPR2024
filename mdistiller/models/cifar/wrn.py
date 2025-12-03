import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdistiller.models.cifar.utils import SPP
from .afpn import AFPN  # 引入公共 AFPN 模块

__all__ = ["wrn"]


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet_SDD(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, M=None):
        super(WideResNet_SDD, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.spp = SPP(M=M)
        self.class_num = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.stage_channels = nChannels

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def forward(self, x):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out); f1 = out
        out = self.block2(out); f2 = out
        out = self.block3(out); f3 = out
        
        # f3_relu: 8x8 特征图 (Feature Map)
        out = self.relu(self.bn1(out)) 
        f3_relu = out

        # SDD
        x_spp, x_strength = self.spp(out)

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        # Global
        out = F.avg_pool2d(out, 8)
        out = out.reshape(-1, self.nChannels)
        f4 = out
        out = self.fc(out)

        f1_pre = self.block2.layer[0].bn1(f1)
        f2_pre = self.block3.layer[0].bn1(f2)
        f3_pre = self.bn1(f3)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = f4

        # [修复] 返回 4 个值，其中 f3_relu 是 8x8 的特征图
        return out, patch_score, None, f3_relu


class WideResNet_AFPN_SDD(nn.Module):
    """
    集成 AFPN 的 WideResNet 版本 (纯净版)
    """
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, M=None):
        super(WideResNet_AFPN_SDD, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        
        # === 1. 初始化 AFPN ===
        # f1 (Block1): 32x32, Ch=nChannels[1]
        # f2 (Block2): 16x16, Ch=nChannels[2]
        # f3 (Block3): 8x8,   Ch=nChannels[3]
        # 注意: Block1 输出是 32x32 (因为 CIFAR Input 32x32, conv1 stride 1, block1 stride 1)
        # ResNet 的 f1 是 32x32, f2 16x16, f3 8x8。这里结构对齐。
        c1 = nChannels[1]
        c2 = nChannels[2]
        c3 = nChannels[3]
        
        print(f"[INFO] 🚀 WRN-AFPN Initialized: Inputs=[{c1}, {c2}, {c3}] -> Out={c3}")
        self.afpn = AFPN(in_channels=[c1, c2, c3], out_channels=c3)
        # ====================

        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        # === 2. 强制使用标准 SPP ===
        print(f"[INFO] 🔹 Using Standard SPP (M={M})")
        self.spp = SPP(M=M)
        self.class_num = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.block1(out); f1 = out # 32x32
        out = self.block2(out); f2 = out # 16x16
        out = self.block3(out); f3 = out # 8x8 (pre-bn)
        
        # 注意: f3 进入 AFPN 前不一定要过 BN/ReLU，AFPN 内部有处理
        # 但为了特征对齐，通常也可以传原始 Block 输出
        
        # === 3. AFPN 融合 ===
        # 将 [32x32, 16x16, 8x8] 融合为 8x8
        feature_enhanced = self.afpn([f1, f2, f3])
        # ====================
        
        # === 4. SDD ===
        x_spp, x_strength = self.spp(feature_enhanced)

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.class_num))
        patch_score = patch_score.permute((1, 2, 0))

        # Global
        out = F.avg_pool2d(feature_enhanced, 8)
        out = out.reshape(-1, self.nChannels)
        out = self.fc(out)

        # [修复] 返回 feature_enhanced
        return out, patch_score, None, feature_enhanced


# ==================== Factory Functions ====================

def wrn_sdd(**kwargs):
    model = WideResNet_SDD(**kwargs)
    return model

def wrn_40_2_sdd(**kwargs):
    model = WideResNet_SDD(depth=40, widen_factor=2, **kwargs)
    return model

def wrn_40_1_sdd(**kwargs):
    model = WideResNet_SDD(depth=40, widen_factor=1, **kwargs)
    return model

def wrn_16_2_sdd(**kwargs):
    model = WideResNet_SDD(depth=16, widen_factor=2, **kwargs)
    return model

def wrn_16_1_sdd(**kwargs):
    model = WideResNet_SDD(depth=16, widen_factor=1, **kwargs)
    return model

# === 注册 AFPN 版本 ===
def wrn_40_2_afpn_sdd(**kwargs):
    return WideResNet_AFPN_SDD(depth=40, widen_factor=2, **kwargs)

def wrn_16_2_afpn_sdd(**kwargs):
    return WideResNet_AFPN_SDD(depth=16, widen_factor=2, **kwargs)

def wrn_40_1_afpn_sdd(**kwargs):
    return WideResNet_AFPN_SDD(depth=40, widen_factor=1, **kwargs)


# ==================== Vanilla (Keep for compatibility) ====================

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.reshape(-1, self.nChannels)
        out = self.fc(out)
        return out, {}

def wrn(**kwargs): return WideResNet(**kwargs)
def wrn_40_2(**kwargs): return WideResNet(depth=40, widen_factor=2, **kwargs)
def wrn_40_1(**kwargs): return WideResNet(depth=40, widen_factor=1, **kwargs)
def wrn_16_2(**kwargs): return WideResNet(depth=16, widen_factor=2, **kwargs)
def wrn_16_1(**kwargs): return WideResNet(depth=16, widen_factor=1, **kwargs)


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 3, 32, 32)
    net = wrn_40_2(num_classes=100)
    logit, feats = net(x)
    print(logit.shape)