"""
TVCI: Complete Model Integration
基于ZORI架构的完整TVCI模型实现
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, List

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

# 导入之前定义的核心模块
from .TVCI_modules import (
    SeleTVCIeEnhancedClassifier,
    ChannelDiscriminativeAdapter,
    BidirectionalVTPathUpdate,
    VisualTextPriorFusion
)

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


@META_ARCH_REGISTRY.register()
class TVCI(nn.Module):
    """
    TVCI: Channel-discriminative Text-visual Interaction for
    Zero-shot Remote Sensing Instance Segmentation

    主要改进：
    1. SeleTVCIe Enhanced Classifier (SEC)
    2. Channel Discriminative Adaptation (CDA)
    3. Bidirectional Visual-Text Path Update (BVTPU)
    4. Visual-Text Prior Fusion Prediction (VTPFP)
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            sem_seg_head,
            criterion,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            train_metadata,
            test_metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            test_topk_per_image: int,
            generalized: bool,
            # cache bank
            cache: bool,
            seen_key,
            seen_value,
            unseen_key,
            unseen_value,
            alpha: float,
            # TVCI specific parameters
            num_prototypes: int = 100,
            adapter_ratio: float = 0.5,
            selection_ratio: float = 0.7,
            bvtpu_layers: int = 3,
            ensemble_weight_visual: float = 0.4,
            ensemble_weight_text: float = 0.3,
            ensemble_weight_cache: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata

        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.generalized = generalized
        self.cache = cache
        self.alpha = alpha

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.void_embedding = nn.Embedding(1, backbone.dim_latent)

        # 准备类别名称
        _, self.train_num_templates, self.train_class_names = \
            self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = \
            self.prepare_class_names_from_metadata(test_metadata, train_metadata)

        # 加载缓存库
        self.seen_cache_keys = torch.load(seen_key).cuda()
        self.seen_cache_values = torch.load(seen_value).cuda()
        self.unseen_cache_keys = torch.load(unseen_key).cuda()
        self.unseen_cache_values = torch.load(unseen_value).cuda()

        # 合并缓存
        self.cache_values = torch.cat([self.seen_cache_values, self.unseen_cache_values], dim=0)
        self.cache_keys = torch.cat([self.seen_cache_keys, self.unseen_cache_keys], dim=0)

        # ============ TVCI核心模块初始化 ============
        visual_dim = backbone.dim_latent
        text_dim = backbone.dim_latent

        # 1. 选择增强分类器
        self.seleTVCIe_classifier = SeleTVCIeEnhancedClassifier(
            text_dim=text_dim,
            num_channels=16,
            selection_ratio=selection_ratio
        )

        # 2. 通道判别自适应
        self.channel_adapter = ChannelDiscriminativeAdapter(
            visual_dim=visual_dim,
            adapter_ratio=adapter_ratio
        )

        # 3. 双向视觉文本路径更新
        self.bvt_updater = BidirectionalVTPathUpdate(
            dim=visual_dim,
            num_heads=8,
            num_layers=bvtpu_layers
        )

        # 4. 视觉文本先验融合
        self.prior_fusion = VisualTextPriorFusion(
            dim=visual_dim,
            num_prototypes=num_prototypes
        )

        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.tensor([
            ensemble_weight_visual,
            ensemble_weight_text,
            ensemble_weight_cache
        ]))

        # 掩码池化（用于提取区域特征）
        from .modeling.transformer_decoder.zori_transformer_decoder import MaskPooling
        self.mask_pooling = MaskPooling()

        self.train_text_classifier = None
        self.test_text_classifier = None

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        """准备类别名称（与原ZORI相同）"""

        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')
                res.append(x_)
            return res

        try:
            class_names = split_labels(metadata.stuff_classes)
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)

        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in RESISC45_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(RESISC45_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)
        class_names = templated_class_names

        return category_overlapping_mask, num_templates, class_names

    def get_text_classifier(self):
        """获取文本分类器（支持训练和测试模式）"""
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.train_class_names[idx:idx + bs],
                            self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                # 平均并归一化
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(RESISC45_PROMPT),
                    len(RESISC45_PROMPT),
                    text_classifier.shape[-1]
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.test_class_names[idx:idx + bs],
                            self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(RESISC45_PROMPT),
                    len(RESISC45_PROMPT),
                    text_classifier.shape[-1]
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        """从配置文件构建模型"""
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # Loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # Building criterion
        from .modeling.matcher import HungarianMatcher
        from .modeling.criterion import SetCriterion

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                    or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "generalized": cfg.MODEL.GENERALIZED,
            "cache": cfg.MODEL.CACHE_BANK.CACHE,
            "seen_key": cfg.MODEL.CACHE_BANK.SEEN_KEY,
            "seen_value": cfg.MODEL.CACHE_BANK.SEEN_VALUE,
            "unseen_key": cfg.MODEL.CACHE_BANK.UNSEEN_KEY,
            "unseen_value": cfg.MODEL.CACHE_BANK.UNSEEN_VALUE,
            "alpha": cfg.MODEL.CACHE_BANK.ALPHA,
            # TVCI specific
            "num_prototypes": cfg.MODEL.TVCI.NUM_PROTOTYPES,
            "adapter_ratio": cfg.MODEL.TVCI.ADAPTER_RATIO,
            "selection_ratio": cfg.MODEL.TVCI.SELECTION_RATIO,
            "bvtpu_layers": cfg.MODEL.TVCI.BVTPU_LAYERS,
            "ensemble_weight_visual": cfg.MODEL.TVCI.ENSEMBLE_WEIGHT_VISUAL,
            "ensemble_weight_text": cfg.MODEL.TVCI.ENSEMBLE_WEIGHT_TEXT,
            "ensemble_weight_cache": cfg.MODEL.TVCI.ENSEMBLE_WEIGHT_CACHE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        TVCI前向传播

        主要流程：
        1. 图像编码 + 文本编码
        2. 应用选择增强分类器(SEC)
        3. 应用通道判别自适应(CDA)
        4. 双向视觉文本更新(BVTPU)
        5. 视觉文本先验融合预测(VTPFP)
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # 提取视觉特征
        features = self.backbone(images.tensor)

        # 获取文本分类器
        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat([
            text_classifier,
            F.normalize(self.void_embedding.weight, dim=-1)
        ], dim=0)

        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        # 通过分割头
        outputs = self.sem_seg_head(features)

        if self.training:
            # 训练模式：计算损失
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses

        else:
            # 推理模式：应用TVCI模块
            return self.inference_with_TVCI(batched_inputs, images, features, outputs)

    def inference_with_TVCI(self, batched_inputs, images, features, outputs):
        """使用TVCI模块进行推理"""
        mask_cls_results = outputs["pred_logits"]  # [B, Q, C+1]
        mask_pred_results = outputs["pred_masks"]  # [B, Q, H, W]

        # 获取CLIP视觉特征
        clip_feature = features["clip_vis_dense"]  # [B, C, H', W']
        text_classifier = features['text_classifier']  # [C+1, D]
        num_templates = features['num_templates']

        B, Q = mask_pred_results.shape[:2]
        C = text_classifier.shape[0] - 1  # 去除void类

        # 上采样mask用于池化
        mask_for_pooling = F.interpolate(
            mask_pred_results,
            size=clip_feature.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 提取池化后的区域特征
        if "convnext" in self.backbone.model_name.lower():
            pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
            pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
        elif "rn" in self.backbone.model_name.lower():
            pooled_clip_feature = self.backbone.visual_prediction_forward(
                clip_feature, mask_for_pooling
            )
        else:
            raise NotImplementedError

        # ============ 应用TVCI模块 ============

        # 1. 选择增强分类器
        enhanced_text, channel_mask = self.seleTVCIe_classifier(text_classifier[:-1])  # 不包含void

        # 2. 通道判别自适应
        adapted_visual = self.channel_adapter(pooled_clip_feature)  # [B*Q, D]
        adapted_visual = adapted_visual.view(B, Q, -1)

        # 3. 双向视觉文本路径更新
        enhanced_text_batch = enhanced_text.unsqueeze(0).expand(B, -1, -1)
        updated_visual, updated_text = self.bvt_updater(adapted_visual, enhanced_text_batch)

        # 4. 视觉文本先验融合预测
        TVCI_logits = self.prior_fusion(
            updated_visual,
            updated_text,
            self.cache_keys,
            self.cache_values
        )  # [B, Q, C]

        # ============ 多路径集成 ============

        # 原始分类器预测（去除void）
        in_vocab_cls = mask_cls_results[..., :-1].softmax(-1)  # [B, Q, C]

        # 标准CLIP相似度预测
        standard_clip_logits = torch.einsum(
            'bqd,cd->bqc',
            F.normalize(pooled_clip_feature.view(B, Q, -1), dim=-1),
            F.normalize(text_classifier[:-1], dim=-1)
        )
        standard_clip_probs = (standard_clip_logits * self.backbone.clip_model.logit_scale.exp()).softmax(-1)

        # 归一化集成权重
        weights = F.softmax(self.ensemble_weights, dim=0)
        w_visual, w_text, w_prior = weights[0], weights[1], weights[2]

        # 加权集成
        ensemble_probs = (
                w_visual * in_vocab_cls +
                w_text * standard_clip_probs +
                w_prior * TVCI_logits.softmax(-1)
        )

        # 考虑seen/unseen类别的几何集成
        category_overlapping_mask = self.category_overlapping_mask.to(self.device)

        # Seen类别：更多依赖训练的分类器
        alpha_seen = 0.3
        cls_logits_seen = (
                (in_vocab_cls ** (1 - alpha_seen) * ensemble_probs ** alpha_seen).log()
                * category_overlapping_mask
        )

        # Unseen类别：更多依赖TVCI融合
        alpha_unseen = 0.7
        cls_logits_unseen = (
                (in_vocab_cls ** (1 - alpha_unseen) * ensemble_probs ** alpha_unseen).log()
                * (1 - category_overlapping_mask)
        )

        cls_results = cls_logits_seen + cls_logits_unseen

        # 添加void概率
        is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
        mask_cls_probs = torch.cat([
            cls_results.softmax(-1) * (1.0 - is_void_prob),
            is_void_prob
        ], dim=-1)
        mask_cls_results = torch.log(mask_cls_probs + 1e-8)

        # 上采样mask到原始分辨率
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        # 后处理
        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # 语义分割推理
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # 全景分割推理
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # 实例分割推理
            if self.instance_on:
                if self.generalized:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                else:
                    instance_r = retry_if_cuda_oom(self.instance_inference_zsis)(
                        mask_cls_result, mask_pred_result
                    )
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def prepare_targets(self, targets, images):
        """准备训练目标（与原ZORI相同）"""
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        """语义分割推理"""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        """全景分割推理（与原ZORI相同）"""
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    })

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        """通用实例分割推理"""
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)

        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1
        ).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]