# -*- coding: utf-8 -*-
"""
TVCI: Context-aware Instance Vision for Zero-shot Remote Sensing Instance Segmentation
基于FC-clip,ZoRI改进的零样本遥感实例分割模型配置

主要创新点:
1. 选择增强分类器 (Channel Selection Enhanced Classifier)
2. 视觉特征解耦的通道判别自适应 (Channel Discriminative Adaptation, CDA)
3. 双向视觉文本路径更新聚合 (Bidirectional Visual-Text Path, VTP-FP)
4. 视觉文本先验融合预测 (Visual-Text Prior Fusion Prediction)
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER (保持与ZoRI一致)
    """
    # data config
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # maskformer2 extra configs
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_TVCI_config(cfg):
    """
    TVCI模型配置 - 针对零样本遥感实例分割的创新
    """
    # ==================== TVCI主配置 ====================
    cfg.MODEL.TVCI = CN()

    # 模型基本信息
    cfg.MODEL.TVCI.VERSION = "1.0"
    cfg.MODEL.TVCI.TASK = "zero_shot_remote_sensing_instance_segmentation"

    # ==================== CLIP编码器配置 (继承ZoRI) ====================
    cfg.MODEL.TVCI.CLIP = CN()
    cfg.MODEL.TVCI.CLIP.MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.TVCI.CLIP.PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.TVCI.CLIP.EMBED_DIM = 768
    cfg.MODEL.TVCI.CLIP.IMAGE_ENCODER_FROZEN = False  # 是否冻结图像编码器
    cfg.MODEL.TVCI.CLIP.TEXT_ENCODER_FROZEN = True  # 文本编码器保持冻结

    # ==================== 创新点1: 选择增强分类器 (SEC) ====================
    # Selection-Enhanced Classifier
    cfg.MODEL.TVCI.SEC = CN()
    cfg.MODEL.TVCI.SEC.ENABLED = True

    # 文本通道选择配置
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION = CN()
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.ENABLED = True
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.NUM_CHANNELS = 300  # 选择判别性文本通道数量
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.SELECTION_METHOD = "discriminative"  # discriminative, attention, learnable
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.INDICES_PATH = "./indices/TVCI_text_channels_300.pt"
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.DYNAMIC = True  # 动态选择vs静态索引
    cfg.MODEL.TVCI.SEC.TEXT_CHANNEL_SELECTION.LEARNABLE_THRESHOLD = True  # 学习通道选择阈值

    # 文本嵌入增强
    cfg.MODEL.TVCI.SEC.TEXT_ENHANCEMENT = CN()
    cfg.MODEL.TVCI.SEC.TEXT_ENHANCEMENT.REFINEMENT_LAYERS = 2  # 文本嵌入refinement层数
    cfg.MODEL.TVCI.SEC.TEXT_ENHANCEMENT.HIDDEN_DIM = 512
    cfg.MODEL.TVCI.SEC.TEXT_ENHANCEMENT.DROPOUT = 0.1

    # 对齐能力增强
    cfg.MODEL.TVCI.SEC.ALIGNMENT_BOOST = CN()
    cfg.MODEL.TVCI.SEC.ALIGNMENT_BOOST.ENABLED = True
    cfg.MODEL.TVCI.SEC.ALIGNMENT_BOOST.TEMPERATURE = 0.07
    cfg.MODEL.TVCI.SEC.ALIGNMENT_BOOST.CONTRASTIVE_WEIGHT = 1.0

    # ==================== 创新点2: 通道判别自适应 (CDA) ====================
    # Channel Discriminative Adaptation
    cfg.MODEL.TVCI.CDA = CN()
    cfg.MODEL.TVCI.CDA.ENABLED = True

    # 视觉特征解耦配置
    cfg.MODEL.TVCI.CDA.VISUAL_DECOUPLING = CN()
    cfg.MODEL.TVCI.CDA.VISUAL_DECOUPLING.ENABLED = True
    cfg.MODEL.TVCI.CDA.VISUAL_DECOUPLING.NUM_GROUPS = 3  # 将视觉通道分为3组
    cfg.MODEL.TVCI.CDA.VISUAL_DECOUPLING.GROUP_STRATEGY = "adaptive"  # adaptive, fixed, learnable

    # CLIP知识保留配置
    cfg.MODEL.TVCI.CDA.KNOWLEDGE_PRESERVATION = CN()
    cfg.MODEL.TVCI.CDA.KNOWLEDGE_PRESERVATION.ENABLED = True
    cfg.MODEL.TVCI.CDA.KNOWLEDGE_PRESERVATION.FROZEN_RATIO = 0.5  # 冻结50%通道保留预训练知识
    cfg.MODEL.TVCI.CDA.KNOWLEDGE_PRESERVATION.FREEZE_STRATEGY = "importance"  # importance, random, first_half

    # 遥感领域适配配置
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION = CN()
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION.ENABLED = True
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION.ADAPTER_TYPE = "lightweight"  # lightweight, bottleneck, parallel
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION.ADAPTER_DIM = 256
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION.NUM_ADAPTER_LAYERS = 2
    cfg.MODEL.TVCI.CDA.DOMAIN_ADAPTATION.REDUCTION_RATIO = 4  # adapter降维比例

    # 视觉通道选择（对应图中的Visual embeddings部分）
    cfg.MODEL.TVCI.CDA.VISUAL_CHANNEL_SELECTION = CN()
    cfg.MODEL.TVCI.CDA.VISUAL_CHANNEL_SELECTION.ENABLED = True
    cfg.MODEL.TVCI.CDA.VISUAL_CHANNEL_SELECTION.NUM_CHANNELS = 160  # 对应refined_channel_11_1_160
    cfg.MODEL.TVCI.CDA.VISUAL_CHANNEL_SELECTION.INDICES_PATH = "./indices/TVCI_vis_channels_160.pt"
    cfg.MODEL.TVCI.CDA.VISUAL_CHANNEL_SELECTION.SELECTION_METHOD = "discriminative"

    # ==================== 创新点3: 双向视觉文本路径更新聚合 (VTP-FP) ====================
    # Bidirectional Visual-Text Path with Feature Propagation
    cfg.MODEL.TVCI.VTP_FP = CN()
    cfg.MODEL.TVCI.VTP_FP.ENABLED = True

    # 双向路径配置
    cfg.MODEL.TVCI.VTP_FP.BIDIRECTIONAL = CN()
    cfg.MODEL.TVCI.VTP_FP.BIDIRECTIONAL.ENABLED = True
    cfg.MODEL.TVCI.VTP_FP.BIDIRECTIONAL.NUM_ITERATIONS = 3  # 双向更新迭代次数
    cfg.MODEL.TVCI.VTP_FP.BIDIRECTIONAL.UPDATE_STRATEGY = "progressive"  # progressive, simultaneous

    # Text Bank配置（对应图中的Text Bank）
    cfg.MODEL.TVCI.VTP_FP.TEXT_BANK = CN()
    cfg.MODEL.TVCI.VTP_FP.TEXT_BANK.ENABLED = True
    cfg.MODEL.TVCI.VTP_FP.TEXT_BANK.CAPACITY = 1000  # 文本库容量
    cfg.MODEL.TVCI.VTP_FP.TEXT_BANK.UPDATE_MOMENTUM = 0.9
    cfg.MODEL.TVCI.VTP_FP.TEXT_BANK.SAVE_PATH = "./cache/TVCI_text_bank.pt"

    # Visual Bank配置（对应图中的Visual Bank）
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK = CN()
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK.ENABLED = True
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK.CAPACITY = 1000  # 视觉库容量
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK.UPDATE_MOMENTUM = 0.9
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK.SAVE_PATH = "./cache/TVCI_visual_bank.pt"
    cfg.MODEL.TVCI.VTP_FP.VISUAL_BANK.PROTOTYPE_LEARNING = True  # 学习航空视觉原型

    # 跨模态对齐配置
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT = CN()
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT.ENABLED = True
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT.ALIGNMENT_TYPE = "bidirectional_attention"
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT.NUM_HEADS = 8
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT.HIDDEN_DIM = 512
    cfg.MODEL.TVCI.VTP_FP.CROSS_MODAL_ALIGNMENT.NUM_LAYERS = 3

    # 特征传播配置（对应图中的multiplication和summation操作）
    cfg.MODEL.TVCI.VTP_FP.FEATURE_PROPAGATION = CN()
    cfg.MODEL.TVCI.VTP_FP.FEATURE_PROPAGATION.ENABLED = True
    cfg.MODEL.TVCI.VTP_FP.FEATURE_PROPAGATION.FUSION_TYPE = "adaptive"  # adaptive, multiply, add, concat
    cfg.MODEL.TVCI.VTP_FP.FEATURE_PROPAGATION.LEARNABLE_WEIGHTS = True

    # ==================== 创新点4: 视觉文本先验融合预测 ====================
    # Visual-Text Prior Fusion Prediction
    cfg.MODEL.TVCI.PRIOR_FUSION = CN()
    cfg.MODEL.TVCI.PRIOR_FUSION.ENABLED = True

    # 航空视觉原型配置
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE = CN()
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE.ENABLED = True
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE.NUM_PROTOTYPES = 100  # 每个类别的原型数量
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE.UPDATE_STRATEGY = "ema"  # ema, kmeans, online
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE.EMA_DECAY = 0.999
    cfg.MODEL.TVCI.PRIOR_FUSION.VISUAL_PROTOTYPE.PROTOTYPE_DIM = 768

    # 文本缓存库配置
    cfg.MODEL.TVCI.PRIOR_FUSION.TEXT_CACHE = CN()
    cfg.MODEL.TVCI.PRIOR_FUSION.TEXT_CACHE.ENABLED = True
    cfg.MODEL.TVCI.PRIOR_FUSION.TEXT_CACHE.CACHE_SIZE = 1000
    cfg.MODEL.TVCI.PRIOR_FUSION.TEXT_CACHE.SAVE_PATH = "./cache/TVCI_text_cache.pt"
    cfg.MODEL.TVCI.PRIOR_FUSION.TEXT_CACHE.SEMANTIC_ENHANCEMENT = True  # 语义丰富度提升

    # 先验融合策略
    cfg.MODEL.TVCI.PRIOR_FUSION.FUSION_STRATEGY = CN()
    cfg.MODEL.TVCI.PRIOR_FUSION.FUSION_STRATEGY.TYPE = "weighted_sum"  # weighted_sum, attention, gated
    cfg.MODEL.TVCI.PRIOR_FUSION.FUSION_STRATEGY.VISUAL_WEIGHT = 0.6
    cfg.MODEL.TVCI.PRIOR_FUSION.FUSION_STRATEGY.TEXT_WEIGHT = 0.4
    cfg.MODEL.TVCI.PRIOR_FUSION.FUSION_STRATEGY.LEARNABLE_WEIGHTS = True

    # 表征差距缩小
    cfg.MODEL.TVCI.PRIOR_FUSION.GAP_REDUCTION = CN()
    cfg.MODEL.TVCI.PRIOR_FUSION.GAP_REDUCTION.ENABLED = True
    cfg.MODEL.TVCI.PRIOR_FUSION.GAP_REDUCTION.PROJECTION_DIM = 512
    cfg.MODEL.TVCI.PRIOR_FUSION.GAP_REDUCTION.NUM_PROJECTION_LAYERS = 2

    # ==================== Mask生成配置 (对应IPAN部分) ====================
    cfg.MODEL.TVCI.MASK_GENERATION = CN()

    # IPAN配置 (Instance Prediction Aggregation Network)
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN = CN()
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.ENABLED = True
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.POGP = CN()  # Point-wise Object Generation Prediction
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.POGP.ENABLED = True
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.POGP.NUM_POINTS = 112 * 112

    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.MILP = CN()  # Mask-level Instance Localization Prediction
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.MILP.ENABLED = True
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.MILP.POOLING_SIZE = 7

    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.FEEN = CN()  # Feature Enhancement Network
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.FEEN.ENABLED = True
    cfg.MODEL.TVCI.MASK_GENERATION.IPAN.FEEN.NUM_LAYERS = 3

    # ==================== 几何集成 (继承ZoRI优点) ====================
    cfg.MODEL.TVCI.GEOMETRIC_ENSEMBLE = CN()
    cfg.MODEL.TVCI.GEOMETRIC_ENSEMBLE.ENABLED = True
    cfg.MODEL.TVCI.GEOMETRIC_ENSEMBLE.ALPHA = 0.4  # 保持ZoRI的参数
    cfg.MODEL.TVCI.GEOMETRIC_ENSEMBLE.BETA = 0.8
    cfg.MODEL.TVCI.GEOMETRIC_ENSEMBLE.ENSEMBLE_ON_VALID_MASK = True  # 改进：在有效mask上集成

    # ==================== 损失函数配置 ====================
    cfg.MODEL.TVCI.LOSS = CN()

    # 对比学习损失（用于跨模态对齐）
    cfg.MODEL.TVCI.LOSS.CONTRASTIVE = CN()
    cfg.MODEL.TVCI.LOSS.CONTRASTIVE.ENABLED = True
    cfg.MODEL.TVCI.LOSS.CONTRASTIVE.WEIGHT = 1.0
    cfg.MODEL.TVCI.LOSS.CONTRASTIVE.TEMPERATURE = 0.07
    cfg.MODEL.TVCI.LOSS.CONTRASTIVE.TYPE = "infonce"  # infonce, triplet, nce

    # 通道选择损失
    cfg.MODEL.TVCI.LOSS.CHANNEL_SELECTION = CN()
    cfg.MODEL.TVCI.LOSS.CHANNEL_SELECTION.ENABLED = True
    cfg.MODEL.TVCI.LOSS.CHANNEL_SELECTION.WEIGHT = 0.5
    cfg.MODEL.TVCI.LOSS.CHANNEL_SELECTION.SPARSITY_WEIGHT = 0.1  # 稀疏性正则化

    # 原型学习损失
    cfg.MODEL.TVCI.LOSS.PROTOTYPE = CN()
    cfg.MODEL.TVCI.LOSS.PROTOTYPE.ENABLED = True
    cfg.MODEL.TVCI.LOSS.PROTOTYPE.WEIGHT = 0.3
    cfg.MODEL.TVCI.LOSS.PROTOTYPE.MARGIN = 0.5

    # 领域适配损失
    cfg.MODEL.TVCI.LOSS.DOMAIN_ADAPTATION = CN()
    cfg.MODEL.TVCI.LOSS.DOMAIN_ADAPTATION.ENABLED = True
    cfg.MODEL.TVCI.LOSS.DOMAIN_ADAPTATION.WEIGHT = 0.2
    cfg.MODEL.TVCI.LOSS.DOMAIN_ADAPTATION.TYPE = "mmd"  # mmd, coral, dann

    # ==================== 推理配置 ====================
    cfg.MODEL.TVCI.TEST = CN()
    cfg.MODEL.TVCI.TEST.CONFIDENCE_THRESHOLD = 0.5
    cfg.MODEL.TVCI.TEST.NMS_THRESHOLD = 0.5
    cfg.MODEL.TVCI.TEST.TOP_K_INSTANCES = 100

    # 测试时使用的库
    cfg.MODEL.TVCI.TEST.USE_TEXT_BANK = True
    cfg.MODEL.TVCI.TEST.USE_VISUAL_BANK = True
    cfg.MODEL.TVCI.TEST.USE_PROTOTYPE = True

    # ==================== 数据集配置 (遥感特定) ====================
    cfg.MODEL.TVCI.DATASET = CN()
    cfg.MODEL.TVCI.DATASET.TYPE = "remote_sensing"
    cfg.MODEL.TVCI.DATASET.SEEN_CLASSES = []  # 在yaml中指定
    cfg.MODEL.TVCI.DATASET.UNSEEN_CLASSES = []  # 在yaml中指定
    cfg.MODEL.TVCI.DATASET.IMAGE_RESOLUTION = 1024

    # ==================== 通用性开关 ====================
    cfg.MODEL.GENERALIZED = True  # 保持泛化零样本设置

    # ==================== 解码器配置 ====================
    cfg.MODEL.DEC = True  # 保持与ZoRI一致


# 向后兼容：保留ZoRI配置用于消融实验
def add_zori_config(cfg):
    """
    保留原始ZoRI配置以便进行消融实验和对比
    """
    cfg.MODEL.ZORI = CN()
    cfg.MODEL.ZORI.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.ZORI.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.ZORI.EMBED_DIM = 768
    cfg.MODEL.ZORI.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.ZORI.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.ZORI.ENSEMBLE_ON_VALID_MASK = False

    cfg.MODEL.GENERALIZED = True

    cfg.MODEL.CACHE_BANK = CN()
    cfg.MODEL.CACHE_BANK.CACHE = True
    cfg.MODEL.CACHE_BANK.SEEN_KEY = './out/seen_11/keys_2.pt'
    cfg.MODEL.CACHE_BANK.SEEN_VALUE = './out/seen_11/values_2.pt'
    cfg.MODEL.CACHE_BANK.UNSEEN_KEY = './out/seen_11/keys_2.pt'
    cfg.MODEL.CACHE_BANK.UNSEEN_VALUE = './out/seen_11/values_2.pt'
    cfg.MODEL.CACHE_BANK.ALPHA = 0.5

    cfg.MODEL.CHANNEL_INDICES = "./out/text_indices/refined_channel_300.pt"
    cfg.MODEL.VIS_CHANNEL_INDICES = "./out/vis_indices/refined_channel_11_1_160.pt"

    cfg.MODEL.DEC = True