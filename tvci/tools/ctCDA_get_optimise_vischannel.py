from tqdm import tqdm
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import open_clip
import os
import random
import argparse
import yaml
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg
from collections import defaultdict
import itertools
import cv2
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from zori.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper


def get_text_embeddings(model, class_names, device, cfg):
    """
    获取类别名称的文本嵌入
    """
    with torch.no_grad():
        # 构建文本提示，例如 "a photo of a {class_name}"
        if 'TEXT_TEMPLATE' in cfg:
            texts = [cfg['TEXT_TEMPLATE'].format(name) for name in class_names]
        else:
            texts = [f"a photo of a {name}" for name in class_names]

        # 对文本进行tokenize
        text_tokens = open_clip.tokenize(texts).to(device)

        # 获取文本特征
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # [num_classes, text_embed_dim]


def compute_text_visual_alignment(visual_feats, text_feats, cfg):
    """
    计算文本嵌入和视觉嵌入的对齐分数

    Args:
        visual_feats: [num_classes, prop_num, channels, H, W]
        text_feats: [num_classes, text_embed_dim]

    Returns:
        alignment_score: [channels] 每个通道的对齐分数
    """
    num_classes = visual_feats.shape[0]
    prop_num = visual_feats.shape[1]
    channels = visual_feats.shape[2]

    # 对每个通道计算对齐分数
    alignment_scores = torch.zeros(channels).cuda()

    for c in range(channels):
        channel_alignment = 0.0
        count = 0

        # 对每个类别和原型
        for i in range(num_classes):
            for p in range(prop_num):
                # 获取该通道的视觉特征 [H, W]
                visual_feat_channel = visual_feats[i, p, c, :, :]

                # 全局平均池化得到一个标量特征
                visual_feat_pooled = visual_feat_channel.mean()

                # 计算与对应类别文本特征的相似度
                # 这里简化处理：使用视觉特征的均值与文本特征的均值的乘积
                text_feat_mean = text_feats[i].mean()

                # 对齐分数：视觉特征和文本特征的相似度
                alignment = visual_feat_pooled * text_feat_mean
                channel_alignment += alignment
                count += 1

        alignment_scores[c] = channel_alignment / count

    return alignment_scores


# def get_indices(cfg, loader, metadata):
#     # CLIP
#     model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     # 获取类别名称
#     # class_names = [metadata.thing_classes[i] for i in metadata.seen_index]
#     # 修改为：使用映射将 category ID 转换为连续索引
#     if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
#         # 使用映射将原始 category ID 转换为连续索引
#         class_names = [
#             metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[cat_id]]
#             for cat_id in metadata.seen_index
#         ]
#     else:
#         # 如果没有映射，假设 seen_index 已经是连续索引
#         class_names = [metadata.thing_classes[i] for i in metadata.seen_index]
#     print(f"Class names: {class_names}")
#
#     # 获取文本嵌入
#     print("Computing text embeddings...")
#     text_embeddings = get_text_embeddings(model, class_names, device, cfg)
#     print(f"Text embeddings shape: {text_embeddings.shape}")
#
#     for prop_num in cfg['PROTOTYPE_NUM']:
#         with torch.no_grad():
#             feats = [[] for _ in range(cfg['CATE_NUM'])]
#             instance_count = {str(class_id): 0 for class_id in range(cfg['CATE_NUM'])}
#             for i, batched_inputs in enumerate(tqdm(loader)):
#                 images = [x["image"].to(device) for x in batched_inputs]
#                 images = [
#                     (x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(
#                         -1, 1, 1).cuda() for x in images]
#                 instances = [x["instances"].to(device) for x in batched_inputs]
#                 for (image, instance) in zip(images, instances):
#                     h, w = 320, 320
#                     gt_boxes = instance.gt_boxes
#                     for i, gt_box in enumerate(gt_boxes):
#                         class_id = instance.gt_classes[i].item()
#                         instance_count[str(class_id)] = instance_count.get(str(class_id), 0) + 1
#                         if instance_count[str(class_id)] <= prop_num:
#                             x_min, y_min, x_max, y_max = gt_box.int().tolist()
#
#                             # Crop the image using the GT box coordinates
#                             cropped_image = image[:, y_min:y_max, x_min:x_max]
#                             resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))
#                             feat = model.visual.trunk.stem(resized_image)
#                             feats[class_id].append(feat)
#                 if all(count >= prop_num for count in instance_count.values()):
#                     break
#
#             feats = list(itertools.chain(*feats))
#             feats = torch.cat(feats, dim=0)
#             feats /= feats.norm(dim=-1, keepdim=True)  # [176, 192, 80, 80]
#
#         feats = feats.reshape(cfg['CATE_NUM'], prop_num, 192, 80, 80)
#
#         # 计算文本-视觉对齐分数（替代类间相似度）
#         print("Computing text-visual alignment scores...")
#         alignment_score = compute_text_visual_alignment(feats, text_embeddings, cfg)
#
#         # 新的准则：最大化文本-视觉对齐 + 保持类内方差
#         # 注意：这里对齐分数越高越好，所以使用正权重
#         criterion = cfg['W'][0] * alignment_score + cfg['W'][1] * torch.var(feats, dim=(0, 1, 3, 4))
#
#         print(f"Alignment score range: [{alignment_score.min():.4f}, {alignment_score.max():.4f}]")
#         print(
#             f"Variance range: [{torch.var(feats, dim=(0, 1, 3, 4)).min():.4f}, {torch.var(feats, dim=(0, 1, 3, 4)).max():.4f}]")
#
#         for channel_num in cfg['VIS_CHANNEL_NUM']:
#             _, indices = torch.topk(criterion, k=channel_num)
#             savefile = "{}/refined_channel_{}_{}_{}.pt".format(cfg['INDICES_DIR'], cfg['CATE_NUM'], prop_num,
#                                                                channel_num)
#             torch.save(indices, savefile)
#             print(f"Saved indices to {savefile}")
#
#     return indices
# def get_indices(cfg, loader, metadata):
#     # CLIP
#     model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     # 获取类别名称
#     if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
#         class_names = [
#             metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[cat_id]]
#             for cat_id in metadata.seen_index
#         ]
#     else:
#         class_names = [metadata.thing_classes[i] for i in range(len(metadata.thing_classes))]
#
#     print(f"Class names: {class_names}")
#
#     # 获取文本嵌入
#     print("Computing text embeddings...")
#     text_embeddings = get_text_embeddings(model, class_names, device, cfg)
#     print(f"Text embeddings shape: {text_embeddings.shape}")
#
#     # 创建 ID 映射：原始 category ID -> 连续索引 (0 到 CATE_NUM-1)
#     # 只考虑 seen 类别
#     id_mapping = {}
#     for idx, cat_id in enumerate(metadata.seen_index):
#         id_mapping[cat_id] = idx
#     print(f"ID mapping: {id_mapping}")
#
#     for prop_num in cfg['PROTOTYPE_NUM']:
#         with torch.no_grad():
#             feats = [[] for _ in range(cfg['CATE_NUM'])]
#             instance_count = {idx: 0 for idx in range(cfg['CATE_NUM'])}  # 使用连续索引
#
#             for i, batched_inputs in enumerate(tqdm(loader)):
#                 images = [x["image"].to(device) for x in batched_inputs]
#                 images = [
#                     (x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(
#                         -1, 1, 1).cuda() for x in images]
#                 instances = [x["instances"].to(device) for x in batched_inputs]
#
#                 for (image, instance) in zip(images, instances):
#                     h, w = 320, 320
#                     gt_boxes = instance.gt_boxes
#
#                     for i, gt_box in enumerate(gt_boxes):
#                         orig_class_id = instance.gt_classes[i].item()  # 原始 category ID
#
#                         # 跳过不在 seen 类别中的实例
#                         if orig_class_id not in id_mapping:
#                             continue
#
#                         # 转换为连续索引
#                         class_id = id_mapping[orig_class_id]
#
#                         instance_count[class_id] = instance_count.get(class_id, 0) + 1
#
#                         if instance_count[class_id] <= prop_num:
#                             x_min, y_min, x_max, y_max = gt_box.int().tolist()
#
#                             # Crop the image using the GT box coordinates
#                             cropped_image = image[:, y_min:y_max, x_min:x_max]
#                             resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))
#                             feat = model.visual.trunk.stem(resized_image)
#                             feats[class_id].append(feat)
#
#                 if all(count >= prop_num for count in instance_count.values()):
#                     break
#
#             feats = list(itertools.chain(*feats))
#             feats = torch.cat(feats, dim=0)
#             feats /= feats.norm(dim=-1, keepdim=True)  # [176, 192, 80, 80]
#
#         feats = feats.reshape(cfg['CATE_NUM'], prop_num, 192, 80, 80)
#
#         # 计算文本-视觉对齐分数
#         print("Computing text-visual alignment scores...")
#         alignment_score = compute_text_visual_alignment(feats, text_embeddings, cfg)
#
#         # 新的准则：最大化文本-视觉对齐 + 保持类内方差
#         criterion = cfg['W'][0] * alignment_score + cfg['W'][1] * torch.var(feats, dim=(0, 1, 3, 4))
#
#         print(f"Alignment score range: [{alignment_score.min():.4f}, {alignment_score.max():.4f}]")
#         print(
#             f"Variance range: [{torch.var(feats, dim=(0, 1, 3, 4)).min():.4f}, {torch.var(feats, dim=(0, 1, 3, 4)).max():.4f}]")
#
#         for channel_num in cfg['VIS_CHANNEL_NUM']:
#             _, indices = torch.topk(criterion, k=channel_num)
#             savefile = "{}/refined_channel_{}_{}_{}.pt".format(cfg['INDICES_DIR'], cfg['CATE_NUM'], prop_num,
#                                                                channel_num)
#             torch.save(indices, savefile)
#             print(f"Saved indices to {savefile}")
#
#     return indices
def get_indices(cfg, loader, metadata):
    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取类别名称
    if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
        class_names = [
            metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[cat_id]]
            for cat_id in metadata.seen_index
        ]
    else:
        class_names = [metadata.thing_classes[i] for i in range(len(metadata.thing_classes))]

    print(f"Class names: {class_names}")

    # 获取文本嵌入
    print("Computing text embeddings...")
    text_embeddings = get_text_embeddings(model, class_names, device, cfg)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # 创建 ID 映射
    id_mapping = {}
    for idx, cat_id in enumerate(metadata.seen_index):
        id_mapping[cat_id] = idx
    print(f"ID mapping: {id_mapping}")

    for prop_num in cfg['PROTOTYPE_NUM']:
        with torch.no_grad():
            feats = [[] for _ in range(cfg['CATE_NUM'])]
            instance_count = {idx: 0 for idx in range(cfg['CATE_NUM'])}

            for i, batched_inputs in enumerate(tqdm(loader)):
                images = [x["image"].to(device) for x in batched_inputs]
                images = [
                    (x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(
                        -1, 1, 1).cuda() for x in images]
                instances = [x["instances"].to(device) for x in batched_inputs]

                for (image, instance) in zip(images, instances):
                    h, w = 320, 320
                    gt_boxes = instance.gt_boxes

                    for i, gt_box in enumerate(gt_boxes):
                        orig_class_id = instance.gt_classes[i].item()

                        if orig_class_id not in id_mapping:
                            continue

                        class_id = id_mapping[orig_class_id]

                        instance_count[class_id] = instance_count.get(class_id, 0) + 1

                        if instance_count[class_id] <= prop_num:
                            x_min, y_min, x_max, y_max = gt_box.int().tolist()

                            # 检查边界框是否有效
                            if x_max <= x_min or y_max <= y_min:
                                continue

                            cropped_image = image[:, y_min:y_max, x_min:x_max]
                            resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))
                            feat = model.visual.trunk.stem(resized_image)
                            feats[class_id].append(feat)

                if all(count >= prop_num for count in instance_count.values()):
                    break

            # 检查每个类别收集到的样本数量
            print("\nSamples collected per class:")
            for class_id in range(cfg['CATE_NUM']):
                print(f"  Class {class_id} ({class_names[class_id]}): {len(feats[class_id])} samples")

            # 确保每个类别都有足够的样本
            # 如果某个类别样本不足，用该类别已有的样本重复填充
            for class_id in range(cfg['CATE_NUM']):
                if len(feats[class_id]) < prop_num:
                    print(
                        f"Warning: Class {class_id} ({class_names[class_id]}) only has {len(feats[class_id])} samples, padding to {prop_num}")
                    # 重复最后一个样本来填充
                    while len(feats[class_id]) < prop_num:
                        if len(feats[class_id]) > 0:
                            feats[class_id].append(feats[class_id][-1])
                        else:
                            # 如果一个样本都没有，创建一个零张量
                            print(f"Error: Class {class_id} ({class_names[class_id]}) has no samples at all!")
                            dummy_feat = torch.zeros(1, 192, 80, 80).to(device)
                            feats[class_id].append(dummy_feat)
                elif len(feats[class_id]) > prop_num:
                    # 如果样本过多，只保留前 prop_num 个
                    feats[class_id] = feats[class_id][:prop_num]

            # 展平并拼接
            feats = list(itertools.chain(*feats))
            feats = torch.cat(feats, dim=0)

            print(f"Total features shape before normalization: {feats.shape}")
            print(f"Expected shape: [{cfg['CATE_NUM'] * prop_num}, 192, 80, 80]")

            feats /= feats.norm(dim=-1, keepdim=True)

        feats = feats.reshape(cfg['CATE_NUM'], prop_num, 192, 80, 80)
        print(f"Features reshaped to: {feats.shape}")

        # 计算文本-视觉对齐分数
        print("Computing text-visual alignment scores...")
        alignment_score = compute_text_visual_alignment(feats, text_embeddings, cfg)

        criterion = cfg['W'][0] * alignment_score + cfg['W'][1] * torch.var(feats, dim=(0, 1, 3, 4))

        print(f"Alignment score range: [{alignment_score.min():.4f}, {alignment_score.max():.4f}]")
        print(
            f"Variance range: [{torch.var(feats, dim=(0, 1, 3, 4)).min():.4f}, {torch.var(feats, dim=(0, 1, 3, 4)).max():.4f}]")

        for channel_num in cfg['VIS_CHANNEL_NUM']:
            _, indices = torch.topk(criterion, k=channel_num)
            savefile = "{}/refined_channel_{}_{}_{}.pt".format(cfg['INDICES_DIR'], cfg['CATE_NUM'], prop_num,
                                                               channel_num)
            torch.save(indices, savefile)
            print(f"Saved indices to {savefile}")

    return indices

def main():
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='settings in yaml format')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cfg['INDICES_DIR'] = '{}/vis_indices'.format(cfg['OUTPUT_DIR'])
    os.makedirs(cfg['INDICES_DIR'], exist_ok=True)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)

    print("Preparing dataset.")

    dataset_name = cfg['DATASET']
    # Get metadata for the dataset
    metadata = MetadataCatalog.get(dataset_name)
    cfg['CATE_NUM'] = len(metadata.seen_index)
    # Get dataset dicts
    dataset_dicts = get_detection_dataset_dicts(dataset_name)
    train_loader = build_detection_test_loader(dataset=dataset_dicts,
                                               mapper=COCOInstanceNewBaselineDatasetMapper(cfg, image_format='RGB',
                                                                                           tfm_gens=[]), batch_size=2)

    print("\nLoading visual features from train set.")
    indices = get_indices(cfg, train_loader, metadata)


if __name__ == '__main__':
    main()