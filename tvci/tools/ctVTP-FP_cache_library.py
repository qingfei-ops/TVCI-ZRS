
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

import cv2
from zori.data.datasets.register_isaid_zsi_11_4 import register_all_isaid11_4_instance_seen
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from zori.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

import matplotlib.pyplot as plt
import numpy as np
from mmengine.visualization import Visualizer
import mmcv

from detectron2.modeling import build_backbone
from detectron2.config import CfgNode


# ============= 文本缓存库模块 =============
class TextBank(nn.Module):
    """
    文本缓存库：用多个文本描述增强类别的文本嵌入
    """

    def __init__(self, clip_model, text_tokenizer, class_descriptions, device='cuda'):
        super().__init__()
        self.clip_model = clip_model
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.class_descriptions = class_descriptions
        self.text_banks = {}
        self.num_texts_per_class = {}
        self._build_text_bank()

    def _build_text_bank(self):
        """构建文本缓存库"""
        print("\n" + "=" * 60)
        print("Building Text Bank...")
        print("=" * 60)
        with torch.no_grad():
            for class_id, descriptions in self.class_descriptions.items():
                text_features = []
                for desc in descriptions:
                    text_tokens = self.text_tokenizer([desc]).to(self.device)
                    text_feat = self.encode_text(text_tokens)
                    text_features.append(text_feat)

                self.text_banks[class_id] = torch.cat(text_features, dim=0)
                self.num_texts_per_class[class_id] = len(descriptions)
                print(f"  Class {class_id}: {len(descriptions)} text descriptions")

        print("✓ Text Bank built successfully\n")

    def encode_text(self, text_tokens):
        """使用 CLIP 编码文本"""
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)

    def strengthen_text_embedding(self, class_id, visual_feature=None):
        """
        增强文本嵌入 (公式 21-22)
        """
        if class_id not in self.text_banks:
            # 如果类别不在文本库中，返回零向量
            return torch.zeros(1, self.clip_model.text_projection.shape[-1], device=self.device)

        text_features = self.text_banks[class_id]
        k = text_features.shape[0]

        if visual_feature is not None and k > 1:
            # 使用视觉特征计算权重
            similarities = (visual_feature @ text_features.T).squeeze(0)
            weights = F.softmax(similarities, dim=0)
        else:
            # 均等权重
            weights = torch.ones(k, device=self.device) / k

        # 加权融合
        strengthened_text = (weights.unsqueeze(1) * text_features).sum(dim=0, keepdim=True)
        strengthened_text = F.normalize(strengthened_text, dim=-1)

        return strengthened_text


# ============= 视觉特征融合模块 =============
class VisualFeatureFusion(nn.Module):
    """融合 V₁ 和 V₂ (公式 18)"""

    def __init__(self, feature_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feature_dim) * 0.5)
        self.beta = nn.Parameter(torch.ones(feature_dim))
        self.e = nn.Parameter(torch.zeros(feature_dim))
        self.g = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, V1, V2):
        """V₃ = (α⊙V₁ + (1-α)⊙V₂)⊙β + e⊙(V₁+V₂) + g⊙V₁"""
        alpha = torch.sigmoid(self.alpha)
        term1 = (alpha.view(1, -1, 1, 1) * V1 + (1 - alpha).view(1, -1, 1, 1) * V2) * self.beta.view(1, -1, 1, 1)
        term2 = self.e.view(1, -1, 1, 1) * (V1 + V2)
        term3 = self.g.view(1, -1, 1, 1) * V1
        return term1 + term2 + term3


# ============= CLIP_CDA 模型构建 =============
def build_ipan_clip_model(cfg):
    """构建带 IPAN 的 CLIP 模型"""
    d2_cfg = get_cfg()
    d2_cfg.MODEL.BACKBONE.NAME = "CLIP_KMA"
    d2_cfg.MODEL.ZORI = CfgNode()
    d2_cfg.MODEL.ZORI.CLIP_MODEL_NAME = cfg['MODEL_NAME']
    d2_cfg.MODEL.ZORI.CLIP_PRETRAINED_WEIGHTS = cfg['PRETRAIN']
    d2_cfg.MODEL.VIS_CHANNEL_INDICES = cfg['VIS_CHANNEL_INDICES']

    backbone = build_backbone(d2_cfg, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()
    return backbone


def load_class_descriptions(cfg):
    """加载类别文本描述"""
    if 'CLASS_DESCRIPTIONS' in cfg:
        return cfg['CLASS_DESCRIPTIONS']

    # 默认：为每个类别生成多样化描述
    class_names = cfg.get('CLASS_NAMES', [])
    class_descriptions = {}

    for idx, class_name in enumerate(class_names):
        descriptions = [
            f"{class_name}",
            f"a photo of a {class_name}",
            f"an image containing {class_name}",
            f"a satellite image of {class_name}",
            f"{class_name} in aerial view",
        ]
        class_descriptions[idx] = descriptions

    return class_descriptions


def extract_v1_v2_features(backbone, cropped_image, text_embedding=None):
    """提取 V₁ (原始CLIP) 和 V₂ (IPAN增强)"""
    with torch.no_grad():
        # V₁: 原始 CLIP 特征
        if backbone.model_type == 'convnext':
            x = backbone.clip_model.visual.trunk.stem(cropped_image)
            for i in range(4):
                x = backbone.clip_model.visual.trunk.stages[i](x)
            V1 = backbone.clip_model.visual.trunk.norm_pre(x)
        elif backbone.model_type == 'resnet':
            x = backbone.clip_model.visual.conv1(cropped_image)
            x = backbone.clip_model.visual.bn1(x)
            x = backbone.clip_model.visual.relu1(x)
            x = backbone.clip_model.visual.conv2(x)
            x = backbone.clip_model.visual.bn2(x)
            x = backbone.clip_model.visual.relu2(x)
            x = backbone.clip_model.visual.conv3(x)
            x = backbone.clip_model.visual.bn3(x)
            x = backbone.clip_model.visual.relu3(x)
            x = backbone.clip_model.visual.avgpool(x)
            x = backbone.clip_model.visual.layer1(x)
            x = backbone.clip_model.visual.layer2(x)
            x = backbone.clip_model.visual.layer3(x)
            V1 = backbone.clip_model.visual.layer4(x)

        # V₂: IPAN 处理后的特征
        features_dict = backbone.extract_features(cropped_image)
        V2 = features_dict['clip_vis_dense']

        return V1, V2


def extract_enhanced_feature(backbone, fusion_module, cropped_image, text_embedding=None):
    """
    提取增强后的特征向量
    ⚠️ 修正：确保输出维度与 CLIP.encode_image() 一致
    """
    with torch.no_grad():
        # 提取 V₁ 和 V₂
        V1, V2 = extract_v1_v2_features(backbone, cropped_image, text_embedding)

        # 融合得到 V₃
        V3 = fusion_module(V1, V2)

        # 全局平均池化
        feature = F.adaptive_avg_pool2d(V3, (1, 1)).flatten(1)  # [1, 1536]

        # ============= 关键修正：应用 CLIP 的投影层 =============
        if hasattr(backbone, 'clip_model') and hasattr(backbone.clip_model.visual, 'head'):
            # 对于 ConvNeXt：通过 head 投影到 768 维
            feature = backbone.clip_model.visual.head(feature)  # [1, 768]
        elif hasattr(backbone, 'clip_model') and hasattr(backbone.clip_model.visual, 'attnpool'):
            # 对于 ResNet：通过 attnpool 投影
            feature = backbone.clip_model.visual.attnpool(V3).flatten(1)

        # 归一化
        feature = F.normalize(feature, dim=-1)

        return feature

def build_cache_model(cfg, loader, use_text_bank=True):
    """
    构建缓存库

    Args:
        cfg: 配置字典
        loader: 数据加载器
        use_text_bank: 是否使用文本缓存库增强
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============= 初始化模型 =============
    if use_text_bank and cfg.get('USE_TEXT_BANK', False):
        print("\n" + "=" * 60)
        print("Mode: Enhanced with Text Bank + IPAN + Fusion")
        print("=" * 60)

        # 1. 加载 CLIP 用于文本编码
        clip_model, _, _ = open_clip.create_model_and_transforms(
            cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN']
        )
        text_tokenizer = open_clip.get_tokenizer(cfg['MODEL_NAME'])
        clip_model.to(device)
        clip_model.eval()

        # 2. 构建文本缓存库
        class_descriptions = load_class_descriptions(cfg)
        text_bank = TextBank(clip_model, text_tokenizer, class_descriptions, device)

        # 3. 加载 IPAN 增强的 backbone
        backbone = build_ipan_clip_model(cfg)

        # 4. 初始化融合模块
        if 'convnext_base' in cfg['MODEL_NAME'].lower():
            feature_dim = 1024
        elif 'convnext_large' in cfg['MODEL_NAME'].lower():
            feature_dim = 1536
        elif 'rn50' in cfg['MODEL_NAME'].lower():
            feature_dim = 2048
        else:
            feature_dim = 1024

        fusion_module = VisualFeatureFusion(feature_dim)
        fusion_module.to(device)
        fusion_module.eval()

        if cfg.get('FUSION_WEIGHTS_PATH') and os.path.exists(cfg['FUSION_WEIGHTS_PATH']):
            fusion_module.load_state_dict(torch.load(cfg['FUSION_WEIGHTS_PATH']))
            print(f"✓ Loaded fusion weights")

        feature_extractor = lambda img, text_emb: extract_enhanced_feature(
            backbone, fusion_module, img, text_emb
        )
    else:
        print("\n" + "=" * 60)
        print("Mode: Original CLIP (Baseline)")
        print("=" * 60)

        # 原始 CLIP
        model, _, _ = open_clip.create_model_and_transforms(
            cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN']
        )
        model.to(device)
        model.eval()

        text_bank = None
        feature_extractor = lambda img, text_emb: model.encode_image(img)

    # ============= 构建缓存 =============
    for num in cfg['PROTOTYPE_NUM']:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            sampled_instances = {str(class_id): [] for class_id in cfg['SEEN_IDS']}
            instance_count = {str(class_id): 0 for class_id in cfg['SEEN_IDS']}

            print(f"\nBuilding cache with {num} prototypes per class...")

            for i, batched_inputs in enumerate(tqdm(loader, desc=f"Processing")):
                image_ids = [x["image_id"] for x in batched_inputs]
                images_ = [x["image"].to(device) for x in batched_inputs]
                images = [(x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) /
                          torch.Tensor(cfg['PIXEL_STD']).view(-1, 1, 1).cuda() for x in images_]
                instances = [x["instances"].to(device) for x in batched_inputs]

                for (image_, image_id, image, instance) in zip(images_, image_ids, images, instances):
                    h, w = 320, 320
                    image_ = torch.permute(image_, (1, 2, 0))
                    gt_boxes = instance.gt_boxes  # XYXY

                    for i, gt_box in enumerate(gt_boxes):
                        class_id = instance.gt_classes[i].item()

                        if class_id in cfg['SEEN_IDS']:
                            if instance_count[str(class_id)] < num:
                                if image_id not in sampled_instances[str(class_id)]:
                                    instance_count[str(class_id)] += 1
                                    sampled_instances[str(class_id)].append(image_id)
                                    x_min, y_min, x_max, y_max = gt_box.int().tolist()

                                    # 裁剪图像
                                    cropped_image = image[:, y_min:y_max, x_min:x_max]
                                    image_box = image_[y_min:y_max, x_min:x_max, :]

                                    # 可视化（与原始代码一致）
                                    if num == 32:
                                        img_dir = '{}/imgs_{}'.format(cfg['OUTPUT_DIR'], num)
                                        os.makedirs(img_dir, exist_ok=True)
                                        fig, axs = plt.subplots(1, 2)
                                        axs[0].imshow(image_.int().cpu().numpy())
                                        axs[0].axis('off')
                                        axs[0].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                                       linewidth=2, edgecolor='r', facecolor='none'))
                                        axs[1].imshow(image_box.int().cpu().numpy())
                                        axs[1].axis('off')
                                        plt.tight_layout()
                                        output_filename = f'img_{image_id}_class_{class_id}.jpg'
                                        plt.savefig("{}/{}".format(img_dir, output_filename))
                                        plt.close()

                                    # 调整大小
                                    resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))

                                    # ============= 提取特征（增强或原始）=============
                                    if text_bank is not None:
                                        # 使用 backbone 提取视觉特征用于计算权重
                                        with torch.no_grad():
                                            if hasattr(backbone, 'clip_model'):
                                                temp_vis = backbone.clip_model.encode_image(resized_image)
                                            else:
                                                V1, V2 = extract_v1_v2_features(backbone, resized_image, None)
                                                temp_vis = F.adaptive_avg_pool2d(V1, (1, 1)).flatten(1)

                                            temp_vis = F.normalize(temp_vis, dim=-1)

                                        strengthened_text = text_bank.strengthen_text_embedding(
                                            class_id, temp_vis
                                        )
                                        cache_key = feature_extractor(resized_image, strengthened_text)
                                    else:
                                        cache_key = feature_extractor(resized_image, None)

                                    cache_keys.append(cache_key)
                                    cache_values.append(instance.gt_classes[i].unsqueeze(0))

            # ============= ⚠️ 关键修正：与右边代码完全一致的后处理 =============
            cache_keys = torch.cat(cache_keys, dim=0)
            # 归一化（右边代码有这一步）
            cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
            cache_values = F.one_hot(torch.cat(cache_values, dim=0))

            # 保存（文件名与右边代码一致）
            torch.save(cache_keys, cfg['OUTPUT_DIR'] + '/keys_' + str(num) + ".pt")
            torch.save(cache_values, cfg['OUTPUT_DIR'] + '/values_' + str(num) + ".pt")

            print(f"✓ Saved: keys_{num}.pt {cache_keys.shape}, values_{num}.pt {cache_values.shape}")

    return


def main():
    # 加载配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='settings in yaml format')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)

    print("Preparing dataset.")

    dataset_name = cfg['DATASET']
    metadata = MetadataCatalog.get(dataset_name)
    cfg['SEEN_IDS'] = [i - 1 for i in metadata.seen_index]

    dataset_dicts = get_detection_dataset_dicts(dataset_name)
    train_loader = build_detection_test_loader(
        dataset=dataset_dicts,
        mapper=COCOInstanceNewBaselineDatasetMapper(cfg, image_format='RGB', tfm_gens=[]),
        batch_size=2
    )

    cfg['OUTPUT_DIR'] = '{}/seen_{}'.format(cfg['OUTPUT_DIR'], len(cfg['SEEN_IDS']))
    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)

    print("\nLoading visual features from train set.")

    # 根据配置选择是否使用文本缓存库增强
    use_enhancement = cfg.get('USE_TEXT_BANK', False)
    build_cache_model(cfg, train_loader, use_text_bank=use_enhancement)

    print("\n" + "=" * 60)
    print("Cache building completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()