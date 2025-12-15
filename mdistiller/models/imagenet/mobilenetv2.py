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

class MobileNetV2_AFPN_SDD(nn.Module):
    def __init__(self, M=None, num_classes=1000, **kwargs):
        super(MobileNetV2_AFPN_SDD, self).__init__()
        self.M = M
        self.class_num = num_classes

        # 手动构建分层结构 (ImageNet配置)
        self.stem = conv_bn(3, 32, 2)
        
        # Stage 1: 32->16->24 (28x28 or 56x56 depending on stride)
        self.stage1 = nn.Sequential(
            conv_dw(32, 16, 1),
            conv_dw(16, 24, 2),
            conv_dw(24, 24, 1),
        ) # Out: 24ch
        
        self.stage2 = nn.Sequential(
            conv_dw(24, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 32, 1),
        ) # Out: 32ch
        
        self.stage3 = nn.Sequential(
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 96, 1),
            conv_dw(96, 96, 1),
            conv_dw(96, 96, 1),
        ) # Out: 96ch
        
        self.stage4 = nn.Sequential(
            conv_dw(96, 160, 2),
            conv_dw(160, 160, 1),
            conv_dw(160, 160, 1),
            conv_dw(160, 320, 1),
        ) # Out: 320ch
        
        # 最后的投影层
        self.last_conv = conv_bn(320, 1280, 1)

        # 初始化 AFPN
        # MobileNetV2 ImageNet 通道: [24, 32, 96, 320]
        # 注意：这里我们提取 stage 1,2,3,4 的输出
        c1, c2, c3, c4 = 24, 32, 96, 320
        self.afpn = AFPN(in_channels=[c1, c2, c3, c4], out_channels=1280)
        
        self.fc = nn.Linear(1280, num_classes)
        self.spp = SPP(M=M)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)  # 24
        f2 = self.stage2(f1) # 32
        f3 = self.stage3(f2) # 96
        f4 = self.stage4(f3) # 320
        
        # AFPN 融合
        feature_enhanced = self.afpn([f1, f2, f3, f4]) # Out: 1280
        
        # SDD 流程
        spp_out = self.spp(feature_enhanced)
        if len(spp_out) == 3:
            x_spp, x_strength, masks = spp_out
        else:
            x_spp, x_strength = spp_out
            masks = None

        x_spp = x_spp.permute((2, 0, 1))
        K, B, C = x_spp.shape
        x_spp = x_spp.reshape(K * B, C)
        patch_score = self.fc(x_spp)
        patch_score = patch_score.reshape(K, B, -1).permute(1, 2, 0)

        # 全局分类
        x_global = F.adaptive_avg_pool2d(feature_enhanced, (1, 1))
        x_global = x_global.view(x_global.size(0), -1)
        out = self.fc(x_global)

        return out, patch_score, masks, feature_enhanced


def mobilenetv2_afpn_sdd(pretrained=False, **kwargs):
    return MobileNetV2_AFPN_SDD(**kwargs)