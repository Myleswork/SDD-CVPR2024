import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, cub_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
import ast
import json


def conv_bn():
    return nn.Sequential(
        nn.Conv2d(3, 16, 7, 2, 3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2))
    )

def conv_1x1_bn(inp, oup=1280):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def modify_student_model_for_cub200(model, cfg,n_cls):
    if cfg.DISTILLER.STUDENT == 'resnet18_sdd':

        model.linear = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.STUDENT == 'resnet8x4_sdd':

        model.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(28)

        model.fc = nn.Linear(256, n_cls)
    elif cfg.DISTILLER.STUDENT == 'ShuffleV1_sdd':

        model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(14)

    elif cfg.DISTILLER.STUDENT == 'ShuffleV2_sdd':

        model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(14)


    elif cfg.DISTILLER.STUDENT == 'vgg8_sdd':
        model.classifier = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.STUDENT == 'MobileNetV2_sdd':
        model.conv1 = conv_bn()
        model.avgpool = nn.AvgPool2d(8, ceil_mode=True)
        # print(model_s)
    elif 'MobileNetV2_afpn_sdd' in cfg.DISTILLER.STUDENT:
        print(f"==> [CUB200] Modifying AFPN-MobileNetV2 for {n_cls} classes")
        # 1. 修改分类器 (classifier)
        # MobileNetV2 的 classifier 是 Sequential，最后一层是 Linear
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_cls)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_cls)
            
        # 2. 修改 class_num (防止 SDD reshape 报错)
        model.class_num = n_cls
        
        # 3. CUB200 图片较大，可能需要调整 avgpool (参考原 MobileNetV2_sdd 的做法)
        # 如果输入是 224x224，到这里是 7x7，可以直接用全局池化
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
    elif 'vgg8_afpn_sdd' in cfg.DISTILLER.STUDENT:
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                # 如果是 Sequential，修改最后一层
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, n_cls)
            else:
                # 如果直接是 Linear
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, n_cls)
<<<<<<< HEAD
    elif 'ShuffleV1_afpn_sdd' in cfg.DISTILLER.STUDENT:
        print(f"==> [CUB200] Modifying ShuffleNetV1 for {n_cls} classes")
        
        # 1. 适配输入层 (可选，针对 224x224 输入优化)
        # 如果直接用 CIFAR 的 conv1 (k=1)，第一层输出也是 224x224，计算量巨大且显存可能爆炸
        # 建议替换为类似 ResNet 的 7x7, stride=2
        if hasattr(model, 'conv1'):
            model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, bias=False)
            
        # 2. 修改分类器
        if hasattr(model, 'linear'):
            in_features = model.linear.in_features
            model.linear = nn.Linear(in_features, n_cls)
            
        # 3. 调整池化层 (根据输入尺寸变化)
        # 如果输入 224 -> conv1(s=2) -> 112 -> layer1(s=2) -> 56 -> layer2(s=2) -> 28 -> layer3(s=2) -> 14
        # 最后的特征图是 14x14。原版 avg_pool2d(4) 是给 32x32 输入用的。
        # 我们可以用 adaptive_avg_pool2d 替换，或者修改参数
        if hasattr(model, 'forward'): 
            # 注意：ShuffleNet 的 forward 函数里硬编码了 F.avg_pool2d(out, 4)
            # 这在 Python 动态修改比较麻烦，最好去修改模型定义的源码。
            # 如果不想改源码，可以在这里尝试 Monkey Patch (不推荐但可行)
            pass
    elif cfg.DISTILLER.STUDENT == 'resnet8x4_afpn_sdd':
        # 针对 CUB200 (224x224) 修改输入层，防止特征图过大
        model.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                                bias=False)
        # CIFAR版 ResNet 只有3个Layer，224输入经过 conv1(s=2) -> 112, 
        # layer1 -> 112, layer2(s=2) -> 56, layer3(s=2) -> 28
        # 所以需要 AvgPool(28) 才能得到 1x1 输出
        model.avgpool = nn.AvgPool2d(28)

        # 修改全连接层以适配 CUB200 的类别数 (n_cls=200)
        # resnet8x4 的输出通道数是 256
        model.fc = nn.Linear(256, n_cls)
=======
    elif 'Shuffle' in cfg.DISTILLER.STUDENT and 'afpn' in cfg.DISTILLER.STUDENT:
        if hasattr(model, 'conv1'):
            model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, bias=False)
        if hasattr(model, 'avgpool'):
            model.avgpool = nn.AvgPool2d(14)
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_cls)
>>>>>>> aef97b0a4093ef133f65379c937de303c479d387
    else:
        raise EOFError

    return model


def modify_teacher_model_for_cub200(model,cfg,n_cls):
    if cfg.DISTILLER.TEACHER == 'resnet32x4_sdd':
        model.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(28)

        model.fc = nn.Linear(256, n_cls)
    elif cfg.DISTILLER.TEACHER == 'vgg13_sdd':
        model.classifier = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.TEACHER == 'ResNet50_sdd':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.linear = nn.Linear(2048, n_cls)
    else:
        raise NotImplementedError

    return model


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True,M=cfg.M)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False,M=cfg.M)

        elif cfg.DATASET.TYPE == "cub200":

            net, pretrain_model_path = cub_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes, M=cfg.M)

            model_teacher = modify_teacher_model_for_cub200(model_teacher, cfg,n_cls=num_classes)

            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cub_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes, M=cfg.M
            )
            model_student = modify_student_model_for_cub200(model_student, cfg,n_cls=num_classes)

        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            # model_teacher = net(num_classes=num_classes, M=cfg.M)
            model_teacher = net(num_classes=num_classes, M='[1,2,4]')
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes, M=args.M
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--warmup", type=float, default=20.0)
    parser.add_argument("--M", default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.distributation = False
    cfg.warmup = args.warmup
    cfg.M = args.M
    print(type(cfg.M))
    cfg.freeze()
    main(cfg, args.resume, args.opts)
