import torch
import open_clip
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
import os
import argparse
import yaml
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from zori.data.datasets.register_isaid_zsi_11_4 import register_all_isaid11_4_instance_val_all
import numpy as np

RESISC45_PROMPT = [
    'satellite imagery of {}.',
    'aerial imagery of {}.',
    'satellite photo of {}.',
    'aerial photo of {}.',
    'satellite view of {}.',
    'aerial view of {}.',
    'satellite imagery of a {}.',
    'aerial imagery of a {}.',
    'satellite photo of a {}.',
    'aerial photo of a {}.',
    'satellite view of a {}.',
    'aerial view of a {}.',
    'satellite imagery of the {}.',
    'aerial imagery of the {}.',
    'satellite photo of the {}.',
    'aerial photo of the {}.',
    'satellite view of the {}.',
    'aerial view of the {}.',
]


def get_text_embeddings(cfg, class_names, clip_model, text_tokenizer, template):
    with torch.no_grad():
        clip_weights = []
        for classname in class_names:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            template_texts = [t.format(classname) for t in template]
            texts_token = text_tokenizer(template_texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()  # [768,11]
    return clip_weights


def compute_class_weights(cfg, dataset_name):
    """
    计算类别权重来处理类别不平衡问题
    权重定义为: w_i = 1/n_i / sum(1/n_k for all k)
    """
    # 获取数据集统计信息
    if 'CLASS_SAMPLES' in cfg and cfg['CLASS_SAMPLES'] is not None:
        # 如果配置中提供了每个类别的样本数量
        class_samples = torch.tensor(cfg['CLASS_SAMPLES'], dtype=torch.float32).cuda()
    else:
        # 如果没有提供，尝试从数据集中获取或使用均匀分布
        try:
            dataset_dicts = get_detection_dataset_dicts([dataset_name])
            metadata = MetadataCatalog.get(dataset_name)
            num_classes = len(metadata.thing_classes)

            # 统计每个类别的样本数量
            class_counts = torch.zeros(num_classes)
            for dataset_dict in dataset_dicts:
                for ann in dataset_dict.get("annotations", []):
                    class_counts[ann["category_id"]] += 1

            class_samples = class_counts.cuda()
        except:
            # 如果无法获取统计信息，使用均匀分布
            metadata = MetadataCatalog.get(dataset_name)
            num_classes = len(metadata.thing_classes)
            class_samples = torch.ones(num_classes).cuda()

    # 计算权重: w_i = 1/n_i，然后归一化
    inverse_freq = 1.0 / (class_samples + 1e-8)  # 加小常数避免除零
    weights = inverse_freq / inverse_freq.sum()

    return weights


def get_indices(cfg, clip_weights, class_weights):
    """
    使用加权方法计算特征通道选择指标
    """
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t()  # [N, feat_dim]

    # 1. 计算加权相似度 S_weighted
    weighted_sim_sum = torch.zeros((feat_dim)).cuda()
    weight_sum = 0

    for i in range(cate_num):
        for j in range(cate_num):
            if i != j:
                # 加权相似度计算
                sim_ij = text_feat[i, :] * text_feat[j, :]
                weight_ij = class_weights[i] * class_weights[j]
                weighted_sim_sum += weight_ij * sim_ij
                weight_sum += weight_ij

    # 归一化加权相似度
    S_weighted = weighted_sim_sum / (weight_sum + 1e-8)

    # 2. 计算加权方差 V_weighted
    # 首先计算加权均值
    weighted_mean = torch.zeros(feat_dim).cuda()
    for i in range(cate_num):
        weighted_mean += class_weights[i] * text_feat[i, :]

    # 计算加权方差
    weighted_var = torch.zeros(feat_dim).cuda()
    for i in range(cate_num):
        diff = text_feat[i, :] - weighted_mean
        weighted_var += class_weights[i] * (diff ** 2)

    V_weighted = weighted_var

    # 3. 计算加权目标函数
    # O_weighted = -λ * S_weighted + (1-λ) * V_weighted
    lambda_param = cfg.get('LAMBDA', 0.5)  # 默认λ=0.5
    criterion = (-1) * lambda_param * S_weighted + (1 - lambda_param) * V_weighted

    # 4. 选择top-k通道
    for channel_num in cfg['CHANNEL_NUM']:
        _, indices = torch.topk(criterion, k=channel_num)
        savefile = "{}/refined_channel_{}.pt".format(cfg['INDICES_DIR'], channel_num)
        torch.save(indices, savefile)

        # 保存权重信息用于分析
        weight_info = {
            'indices': indices,
            'criterion_values': criterion[indices],
            'class_weights': class_weights,
            'weighted_similarity': S_weighted[indices],
            'weighted_variance': V_weighted[indices]
        }
        weight_info_file = "{}/channel_info_{}.pt".format(cfg['INDICES_DIR'], channel_num)
        torch.save(weight_info, weight_info_file)

    return indices


def main():
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='settings in yaml format')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cfg['INDICES_DIR'] = '{}/text_indices'.format(cfg['OUTPUT_DIR'])
    os.makedirs(cfg['INDICES_DIR'], exist_ok=True)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)

    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_isaid11_4_instance_val_all(_root)
    dataset_name = cfg['DATASET']
    metadata = MetadataCatalog.get(dataset_name)
    model, _, _ = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])
    text_tokenizer = open_clip.get_tokenizer(cfg['MODEL_NAME'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_templates = RESISC45_PROMPT
    text_embeddings = get_text_embeddings(cfg, metadata.thing_classes, model, text_tokenizer, prompt_templates)

    # 计算类别权重并使用加权方法
    class_weights = compute_class_weights(cfg, dataset_name)
    print("Class weights:", class_weights)
    indices = get_indices(cfg, text_embeddings, class_weights)
    print("Channel selection completed using weighted method")


if __name__ == '__main__':
    main()