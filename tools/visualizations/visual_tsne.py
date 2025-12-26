import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 引入项目依赖
import sys
sys.path.append(".") # 确保能找到 mdistiller 包

from mdistiller.engine.cfg import CFG as cfg
from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint

def plot_features(features, labels, num_classes, save_path):
    print(f"Plotting t-SNE to {save_path}...")
    plt.figure(figsize=(10, 10))
    
    # 使用 tab10 颜色映射 (适合 10 类)，如果是 CIFAR100，颜色会重复使用
    # s=5 控制点的大小，alpha=0.6 控制透明度
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    
    # 如果类别少于 20，可以显示图例
    if num_classes <= 20:
        plt.legend(*scatter.legend_elements(), title="Classes")
        
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE Visualization")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

def main(args):
    # 1. 加载配置
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    
    # 强制设置
    cfg.DATASET.TYPE = 'cifar100' # 或者是你实际的数据集
    
    # 2. 准备数据 (只用验证集)
    _, val_loader, _, num_classes = get_dataset(cfg)
    
    # 3. 初始化模型
    print(f"Loading student model: {cfg.DISTILLER.STUDENT}")
    
    # 构建参数字典
    model_kwargs = {'num_classes': num_classes}
    
    # 如果命令行传入了 M 参数 (例如 SDD 模型需要)，则解析并传入
    if args.M:
        try:
            # 将字符串 "[1,2]" 转为列表 [1,2]
            import ast
            model_kwargs['M'] = ast.literal_eval(args.M)
            print(f"Model using M={model_kwargs['M']}")
        except:
            print("Warning: Failed to parse M argument, using default.")

    # 某些模型可能需要 use_afpn 参数，视你的注册函数而定
    # 如果你的注册函数写死了 use_afpn=True，这里不用管
    # 否则可能需要: model_kwargs['use_afpn'] = True 
    
    # 实例化模型
    try:
        model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](**model_kwargs)
    except KeyError:
        print(f"Error: Model '{cfg.DISTILLER.STUDENT}' not found in cifar_model_dict.")
        return

    # 4. 加载权重
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        # load_checkpoint 可能会返回 dict 或直接是 state_dict，视你的 utils 实现而定
        # 通常 mdistiller 的 load_checkpoint 返回包含 "model" key 的 dict
        ckpt = load_checkpoint(args.resume)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Warning: No checkpoint loaded! Analyzing random weights.")

    model = model.cuda()
    model.eval()

    # 5. 提取特征
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(val_loader)):
            data = data.cuda()
            
            # 前向传播
            ret = model(data)
            
            # --- 关键：适配不同的返回值 ---
            pooled_feat = None
            
            if isinstance(ret, tuple):
                if len(ret) == 4: 
                    # 新接口: (logits, patch_score, masks, feature_final)
                    feature_map = ret[3] 
                    # 手动进行全局平均池化得到特征向量 [B, C, H, W] -> [B, C]
                    pooled_feat = F.adaptive_avg_pool2d(feature_map, (1, 1)).view(feature_map.size(0), -1)
                    
                elif len(ret) == 2:
                    # 旧接口: (logits, feature_dict)
                    # 尝试从字典中获取 'pooled_feat'
                    if isinstance(ret[1], dict) and 'pooled_feat' in ret[1]:
                        pooled_feat = ret[1]['pooled_feat']
                    else:
                        # 如果没有字典，可能就是 (logits, features)
                        # 假设 ret[1] 是特征
                        pooled_feat = ret[1]
                        if len(pooled_feat.shape) > 2: # 如果是特征图，池化它
                             pooled_feat = F.adaptive_avg_pool2d(pooled_feat, (1, 1)).view(pooled_feat.size(0), -1)
            else:
                # 只有 logits? t-SNE 效果可能不好，但也只能用这个了
                pooled_feat = ret

            if pooled_feat is not None:
                all_features.append(pooled_feat.cpu().numpy())
                all_labels.append(labels.numpy())

    # 拼接
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)

    # 6. 运行 t-SNE
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    # 为了速度，可以先只取一部分数据，例如前2000个
    # all_features_tsne = tsne.fit_transform(all_features[:2000])
    # labels_tsne = all_labels[:2000]
    
    # 跑全量数据:
    all_features_tsne = tsne.fit_transform(all_features)
    
    # 7. 绘图
    output_filename = f"tsne_{cfg.DISTILLER.STUDENT}.png"
    plot_features(all_features_tsne, all_labels, num_classes, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("t-SNE Visualization")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint file (e.g., student_best)")
    parser.add_argument("--M", type=str, default=None, help="M parameter for SDD models, e.g. '[1,2]'")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main(args)