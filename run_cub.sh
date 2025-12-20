# python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_shuv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/res32x4_shuv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_shuv1.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/res32x4_shuv1.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/wrn40_2_wrn_16_2.yaml --M [1,2,4]  #没结果
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/wrn40_2_wrn_16_2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/sdd_dkd/wrn40_2_wrn_40_1.yaml --M [1,2,4]  #没结果
# python train_origin.py --cfg configs/cifar100/AFPN_SDD/wrn40_2_wrn_40_1.yaml --M [1,2,4]

#要跑的
# python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/res32x4_mv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/res32x4_mv2.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/res32x4_mv2.yaml --M [1,2,4]

# python train_origin.py --cfg configs/cifar100/AFPN_SDD_KD/wrn40_2_vgg8.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD_DKD/wrn40_2_vgg8.yaml --M [1,2,4]
# python train_origin.py --cfg configs/cifar100/AFPN_SDD_NKD/wrn40_2_vgg8.yaml --M [1,2,4]

#上面改过了
# res32x4_mv2
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res32x4_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res32x4_mv2.yaml --M [1,2]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res32x4_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res32x4_mv2.yaml --M [1,2]



# res32x4_shuv1
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res32x4_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res32x4_shuv1.yaml --M [1,2]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res32x4_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res32x4_shuv1.yaml --M [1,2]



# vgg13_mv2
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/vgg13_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/vgg13_mv2.yaml --M [1,2]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/vgg13_mv2.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/vgg13_mv2.yaml --M [1,2]

# vgg13_vgg8
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/vgg13_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/vgg13_vgg8.yaml --M [1,2]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/vgg13_vgg8.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/vgg13_vgg8.yaml --M [1,2]

#res50_shuv1
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res50_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_kd_afpn/res50_shuv1.yaml --M [1,2]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res50_shuv1.yaml --M [1,2,4]
python train_origin.py --cfg configs/cub200/sdd_nkd_afpn/res50_shuv1.yaml --M [1,2]