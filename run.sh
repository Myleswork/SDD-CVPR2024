# python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_shuv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/res32x4_shuv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_shuv1.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/res32x4_shuv1.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/wrn40_2_wrn_16_2.yaml --M [1,2,4]  #没结果
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/wrn40_2_wrn_16_2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/wrn40_2_wrn_40_1.yaml --M [1,2,4]  #没结果
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/wrn40_2_wrn_40_1.yaml --M [1,2,4]

#要跑的
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/res32x4_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res32x4_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res32x4_mv2.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/wrn40_2_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/wrn40_2_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/wrn40_2_vgg8.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/wrn40_2_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/wrn40_2_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/wrn40_2_mv2.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res32x4_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res32x4_shuv1.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/wrn40_2_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/wrn40_2_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/wrn40_2_shuv1.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res50_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/res50_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res50_mv2.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/vgg13_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/vgg13_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/vgg13_mv2.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res32x4_shuv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res32x4_shuv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/res32x4_shuv2.yaml --M [1,2,4]

python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res50_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/res50_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res50_vgg8.yaml --M [1,2,4]