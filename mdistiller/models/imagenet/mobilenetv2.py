import torch
import torch.nn as nn
import torch.nn.functional as F
from mdistiller.models.imagenet.utils import SPP
from .afpn import AFPN

class MobileNetV2_SDD(nn.Module):
    def __init__(self,M=None,**kwargs):
        super(MobileNetV2_SDD, self).__init__()
        self.M=M

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)
        self.spp = SPP(M=self.M)

    def forward(self, x, is_feat=False):
        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](F.relu(feat3)))
        feat4=F.relu(feat4)

        x_spp, x_strength = self.spp(feat4)

        # feature_num = x_spp.shape[-1]
        # patch_score = torch.zeros(x_spp.shape[0], self.class_num, feature_num)
        # patch_strength = torch.zeros(x_spp.shape[0], feature_num)

        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, 1000))
        patch_score = patch_score.permute((1, 2, 0))

        feat5 = self.model[14](feat4)
        avg = feat5.reshape(-1, 1024)
        out = self.fc(avg)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [F.relu(feat1), F.relu(feat2), F.relu(feat3), F.relu(feat4)]
        feats["preact_feats"] = [feat1, feat2, feat3, feat4]
        return out, patch_score

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 1024]


class MobileNetV2(nn.Module):
    def __init__(self,**kwargs):
        super(MobileNetV2, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x, is_feat=False):
        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](F.relu(feat3)))
        feat4=F.relu(feat4)


        feat5 = self.model[14](feat4)
        avg = feat5.reshape(-1, 1024)
        out = self.fc(avg)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [F.relu(feat1), F.relu(feat2), F.relu(feat3), F.relu(feat4)]
        feats["preact_feats"] = [feat1, feat2, feat3, feat4]
        return out, feats

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 1024]

class MobileNetV2_AFPN_SDD(MobileNetV2_SDD):
    """
    ImageNet MobileNetV2 with AFPN (No Dynamic Decoupling)
    """
    def __init__(self, M=None, num_classes=1000, **kwargs):
        super(MobileNetV2_AFPN_SDD, self).__init__(M=M, **kwargs)
        
        # MobileNetV2 (ImageNet) 4 阶段通道
        c1, c2, c3, c4 = 128, 256, 512, 1024 # <--- 请修改为这个！
        self.afpn_out_dim = 1280
        
        # 初始化 AFPN
        self.afpn = AFPN(in_channels=[c1, c2, c3, c4], out_channels=self.afpn_out_dim)
        print(f"[INFO] 🚀 AFPN Initialized: Inputs=[{c1}, {c2}, {c3}, {c4}] -> Out={self.afpn_out_dim}")

        # 重置分类器
        self.fc = nn.Linear(self.afpn_out_dim, num_classes)
        self.class_num = num_classes
        
        # 初始化标准 SPP
        self.spp = SPP(M=M)

    def forward(self, x, is_feat=False):
        # 1. 提取 f1 (Stage 1: ~56x56)
        # 经过 idx 0, 1
        feat_0_1 = self.model[0:2](x) 
        # idx 2 是 stride=2 的 conv_dw，我们需要它的输出作为 f1
        # conv_dw 内部结构是 [pw, dw, pw-linear]，我们需要 dw 之后的特征吗？
        # 不，AFPN 通常接收的是每个 stage 最后的输出。
        # self.model[2] 的输出已经是下采样后的结果(56x56)
        f1 = self.model[2](feat_0_1) 
        
        # 2. 提取 f2 (Stage 2: ~28x28)
        # 经过 idx 3
        feat_3 = self.model[3](f1)
        # idx 4 是 stride=2 的下采样
        f2 = self.model[4](feat_3)
        
        # 3. 提取 f3 (Stage 3: ~14x14)
        # 经过 idx 5
        feat_5 = self.model[5](f2)
        # idx 6 是 stride=2 的下采样
        f3 = self.model[6](feat_5)
        
        # 4. 提取 f4 (Stage 4: ~7x7)
        # 经过 idx 7-11
        feat_7_11 = self.model[7:12](f3)
        # idx 12 是 stride=2 的下采样 (变 7x7)
        feat_12 = self.model[12](feat_7_11)
        # idx 13 是最后的卷积
        f4 = self.model[13](feat_12)
        
        # 激活后送入 AFPN
        in_f1 = F.relu(f1)
        in_f2 = F.relu(f2)
        in_f3 = F.relu(f3)
        in_f4 = F.relu(f4)

        # AFPN 融合
        feature_enhanced = self.afpn([in_f1, in_f2, in_f3, in_f4])

        # SDD 处理
        x_spp, x_strength = self.spp(feature_enhanced)

        # Patch Logits
        x_spp = x_spp.permute((2, 0, 1))
        K, B, C = x_spp.shape
        x_spp = x_spp.reshape(K * B, C)
        patch_score = self.fc(x_spp)
        patch_score = patch_score.reshape(K, B, -1).permute(1, 2, 0)

        # Global Logits (自适应池化)
        x_global = F.adaptive_avg_pool2d(feature_enhanced, (1, 1))
        x_global = x_global.view(x_global.size(0), -1)
        out = self.fc(x_global)

        return out, patch_score, None, feature_enhanced

def mobilenetv2_afpn_sdd(pretrained=False, **kwargs):
    return MobileNetV2_AFPN_SDD(**kwargs)