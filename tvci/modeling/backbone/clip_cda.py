
"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


# ============= IPAN Components =============

class AttentionWeightMLP(nn.Module):
    """注意力权重计算网络 (图3)"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, g_combined):
        """
        Args:
            g_combined: [B, H, W, D'] 组合特征 [g_avg, g_max, g_std]
        Returns:
            Q: [B, H, W, 1] 注意力权重
        """
        x = self.relu(self.fc1(g_combined))
        x = self.relu(self.fc2(x))
        Q = torch.sigmoid(self.fc3(x))
        return Q


class VisualGuidedAdaptiveLayer(nn.Module):
    """视觉引导自适应层 G_Layer (图4)"""

    def __init__(self, channel_dim):
        super().__init__()
        # MLP for generating attention weights
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, channel_dim),
            nn.ReLU(),
            nn.Linear(channel_dim, channel_dim)
        )

    def forward(self, V_freeze, V_train):
        """
        Args:
            V_freeze: [B, C, H, W] 冻结分支特征
            V_train: [B, C, H, W] 训练分支特征
        Returns:
            V_freeze_updated: [B, C, H, W] 更新后的冻结特征
            V_train_updated: [B, C, H, W] 更新后的训练特征
        """
        B, C, H, W = V_freeze.shape

        # Compute attention weights
        V_freeze_flat = V_freeze.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]
        Q_train = self.mlp(V_freeze_flat)  # [BHW, C]
        Q_train = Q_train.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        V_train_flat = V_train.permute(0, 2, 3, 1).reshape(B * H * W, C)
        Q_freeze = self.mlp(V_train_flat)
        Q_freeze = Q_freeze.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Update features according to equations (12) and (13)
        # V_freeze^(1) = V_freeze ⊙ (1 + β · Q_train)
        # V_train^(1) = V_train ⊙ (1 + α · Q_freeze)
        beta = 0.1  # 可调节超参数
        alpha = 0.1

        V_freeze_updated = V_freeze * (1 + beta * torch.sigmoid(Q_train))
        V_train_updated = V_train * (1 + alpha * torch.sigmoid(Q_freeze))

        return V_freeze_updated, V_train_updated


class VisionTextPoolingLayer(nn.Module):
    """视觉-文本池化层 VP_MHCA (图5)"""

    def __init__(self, visual_dim, text_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # Multi-head cross-attention
        self.mhca = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Project text embeddings to visual dimension if needed
        if text_dim != visual_dim:
            self.text_proj = nn.Linear(text_dim, visual_dim)
        else:
            self.text_proj = nn.Identity()

    def forward(self, V_freeze_updated, V_train_updated, text_embedding):
        """
        Args:
            V_freeze_updated: [B, C, H, W] 更新后的冻结特征
            V_train_updated: [B, C, H, W] 更新后的训练特征
            text_embedding: [B, text_dim] or [B, N, text_dim] 文本嵌入
        Returns:
            V_freeze_final: [B, C, H, W] 最终冻结特征
            V_train_final: [B, C, H, W] 最终训练特征
        """
        B, C, H, W = V_freeze_updated.shape

        # Flatten spatial dimensions
        V_freeze_flat = V_freeze_updated.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        V_train_flat = V_train_updated.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # Project text embedding
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1)  # [B, 1, text_dim]
        text_key = self.text_proj(text_embedding)  # [B, N, visual_dim]

        # Apply MHCA: equations (14) and (15)
        # V'_freeze = V_freeze + MHCA(V_freeze^(1), V_train, T_ext)
        attn_output_freeze, _ = self.mhca(
            query=V_freeze_flat,
            key=torch.cat([V_train_flat, text_key], dim=1),
            value=torch.cat([V_train_flat, text_key], dim=1)
        )
        V_freeze_final = V_freeze_updated + attn_output_freeze.permute(0, 2, 1).reshape(B, C, H, W)

        # V'_train = V_train + MHCA(V_train^(1), V_freeze, T_ext)
        attn_output_train, _ = self.mhca(
            query=V_train_flat,
            key=torch.cat([V_freeze_flat, text_key], dim=1),
            value=torch.cat([V_freeze_flat, text_key], dim=1)
        )
        V_train_final = V_train_updated + attn_output_train.permute(0, 2, 1).reshape(B, C, H, W)

        return V_freeze_final, V_train_final


class IPAN(nn.Module):
    """Instance-level Prompt Adapter Network 完整模块"""

    def __init__(self, channel_dim, text_dim=512, num_heads=8):
        super().__init__()
        self.channel_dim = channel_dim

        # 三个子模块
        self.attention_mlp = AttentionWeightMLP(
            input_dim=channel_dim * 3,  # [g_avg, g_max, g_std]
            hidden_dim1=channel_dim * 2,
            hidden_dim2=channel_dim
        )
        self.g_layer = VisualGuidedAdaptiveLayer(channel_dim)
        self.vp_mhca = VisionTextPoolingLayer(channel_dim, text_dim, num_heads)

    def compute_statistics(self, V):
        """计算特征统计量 (公式8-10)"""
        B, C, H, W = V.shape
        V_flat = V.view(B, C, H * W)

        # g_avg: 全局平均
        g_avg = V_flat.mean(dim=2)  # [B, C]

        # g_max: 最大激活
        g_max, _ = V_flat.max(dim=2)  # [B, C]

        # g_std: 标准化后的标准差
        g_std = torch.sqrt(((V_flat - g_avg.unsqueeze(2)) ** 2).sum(dim=2) / (H * W))  # [B, C]

        return g_avg, g_max, g_std

    def forward(self, V_freeze, V_train, text_embedding=None):
        """
        Args:
            V_freeze: [B, C, H, W] 冻结分支特征
            V_train: [B, C, H, W] 训练分支特征
            text_embedding: [B, text_dim] 文本嵌入 (可选)
        Returns:
            V_freeze_final: [B, C, H, W] 最终冻结特征
            V_train_final: [B, C, H, W] 最终训练特征
        """
        # Step 1: 计算统计量和注意力权重 (当前版本简化，直接使用G_Layer)
        # 可以选择性地使用 attention_mlp 进一步优化

        # Step 2: 视觉引导自适应 (G_Layer)
        V_freeze_updated, V_train_updated = self.g_layer(V_freeze, V_train)

        # Step 3: 如果有文本嵌入，应用视觉-文本池化
        if text_embedding is not None:
            V_freeze_final, V_train_final = self.vp_mhca(
                V_freeze_updated, V_train_updated, text_embedding
            )
        else:
            V_freeze_final = V_freeze_updated
            V_train_final = V_train_updated

        return V_freeze_final, V_train_final


# ============= Modified CLIP_KMA with IPAN =============

@BACKBONE_REGISTRY.register()
class CLIP_KMA(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.ZORI.CLIP_MODEL_NAME
        pretrained = cfg.MODEL.ZORI.CLIP_PRETRAINED_WEIGHTS
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained

        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        # 在CLIP_KMA类的__init__中，修改模型初始化部分
        # self.clip_model, _, _ = open_clip.create_model_and_transforms(
        #     model_name,
        #     pretrained=pretrained,
        #     force_context_length=128  # 用这个参数指定文本长度（替代text_max_length）
        # )
        #
        # # 重建与128长度匹配的注意力掩码（下三角掩码，防止未来信息泄露）
        # max_length = 128
        # self.clip_model.attn_mask = torch.full((max_length, max_length), -float("inf"))
        # self.clip_model.attn_mask = torch.triu(self.clip_model.attn_mask, diagonal=1).to(self.clip_model.device)
        #
        # # 同步修改文本tokenize的长度
        # def tokenize_text(self, text):
        #     return self.text_tokenizer(text, max_length=128, truncate=True, padding="max_length")

        model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]

        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        vis_channel_indices = cfg.MODEL.VIS_CHANNEL_INDICES
        self.indices = torch.load(vis_channel_indices)
        all_indices = torch.arange(self.output_channels[0])
        top_indices_set = set(self.indices.tolist())
        self.rest_indices = [idx for idx in all_indices.tolist() if idx not in top_indices_set]
        # vis_channel_indices = cfg.MODEL.VIS_CHANNEL_INDICES
        #
        # # 1. 加载索引文件（CPU上加载）
        # indices_cpu = torch.load(vis_channel_indices)
        #
        # # 2. 转移到当前进程的GPU设备（关键：多GPU下每个进程会自动对应一张GPU）
        # #    用模型参数的设备作为基准（自动适配当前进程的GPU，如cuda:0~3）
        # self.indices = indices_cpu.to(next(self.parameters()).device)
        #
        # # 3. 生成all_indices时直接在当前设备上创建
        # all_indices = torch.arange(self.output_channels[0], device=self.indices.device)
        #
        # # 4. 计算rest_indices并绑定到当前设备
        # top_indices_set = set(self.indices.tolist())  # 临时转为CPU列表做判断
        # self.rest_indices = torch.tensor(
        #     [idx for idx in all_indices.tolist() if idx not in top_indices_set],
        #     device=self.indices.device  # 与indices同设备（当前进程的GPU）
        # )

        stem_layer_conv2d = self.clip_model.visual.trunk.stem[
            0] if self.model_type == 'convnext' else self.clip_model.visual.conv1
        out_channel = stem_layer_conv2d.out_channels
        in_channel = stem_layer_conv2d.in_channels
        kernel_size = stem_layer_conv2d.kernel_size
        stride = stem_layer_conv2d.stride

        self.stem_train = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.stem_freeze = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.stem_freeze.load_state_dict(stem_layer_conv2d.state_dict())
        self.stem_train.load_state_dict(stem_layer_conv2d.state_dict())

        for param in self.stem_freeze.parameters():
            param.requires_grad = False

        # ============= 添加 IPAN 模块 =============
        self.ipan = IPAN(
            channel_dim=self.output_channels[0],
            text_dim=self.dim_latent,
            num_heads=8
        )
        # =========================================

        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x, text_embedding=None):
        """添加 text_embedding 参数以支持 IPAN"""
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
        }[self.model_type](x, text_embedding)

    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
            'resnet': self.visual_prediction_forward_resnet,
        }[self.model_type](x, masks)

    def extract_features_convnext(self, x, text_embedding=None):
        out = {}

        # 分别提取冻结和训练分支
        x_train = self.stem_train(x)
        x_freeze = self.stem_freeze(x)

        # ============= 应用 IPAN 更新特征 =============
        x_freeze_updated, x_train_updated = self.ipan(
            V_freeze=x_freeze,
            V_train=x_train,
            text_embedding=text_embedding
        )
        # ===========================================

        # 按照原始逻辑组合特征
        x = torch.zeros_like(x_train_updated)
        x[:, self.rest_indices, :, :] = x_train_updated[:, self.rest_indices, :, :]
        x[:, self.indices.tolist(), :, :] = x_freeze_updated[:, self.indices.tolist(), :, :]

        x = self.clip_model.visual.trunk.stem[1](x)
        out['stem'] = x.contiguous()

        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i + 2}'] = x.contiguous()

        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out

    def extract_features_resnet(self, x, text_embedding=None):
        out = {}

        # ResNet stem (3个卷积层)
        x_train = self.stem_train(x)
        x_freeze = self.stem_freeze(x)

        # ============= 应用 IPAN 更新特征 =============
        x_freeze_updated, x_train_updated = self.ipan(
            V_freeze=x_freeze,
            V_train=x_train,
            text_embedding=text_embedding
        )
        # ===========================================

        # 组合特征
        x = torch.zeros_like(x_train_updated)
        x[:, self.rest_indices, :, :] = x_train_updated[:, self.rest_indices, :, :]
        x[:, self.indices.tolist(), :, :] = x_freeze_updated[:, self.indices.tolist(), :, :]

        # 继续 ResNet 的后续层
        x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        out['stem'] = x.contiguous()

        x = self.clip_model.visual.avgpool(x)
        x = self.clip_model.visual.layer1(x)
        out['res2'] = x.contiguous()
        x = self.clip_model.visual.layer2(x)
        out['res3'] = x.contiguous()
        x = self.clip_model.visual.layer3(x)
        out['res4'] = x.contiguous()
        x = self.clip_model.visual.layer4(x)
        out['res5'] = x.contiguous()
        out['clip_vis_dense'] = x
        return out

    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch * num_query, channel, 1, 1)
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1])

    def visual_prediction_forward_resnet(self, x, masks):
        batch, channel, height, width = x.shape
        if masks.shape[-2] != height or masks.shape[-1] != width:
            masks = F.interpolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        num_masks = masks.shape[1]

        positional_embedding = self.clip_model.visual.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :]
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear', align_corners=False)
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height * width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)
        key_value = x + spatial_pos_embed

        masks = masks.reshape(batch, num_masks, height * width)
        masks = (masks > 0).to(masks.dtype)
        query = x.mean(0, keepdim=True) + positional_embedding[:1, None, :]
        query = query.repeat_interleave(num_masks, dim=0)

        attn_mask = masks < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.clip_model.visual.attnpool.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch * self.clip_model.visual.attnpool.num_heads,
                                      query.shape[0], key_value.shape[0])

        x = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=key_value.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias,
                                    self.clip_model.visual.attnpool.k_proj.bias,
                                    self.clip_model.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.visual.attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )[0].permute(1, 0, 2)

        return x

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x, text_embedding=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            text_embedding: 可选的文本嵌入 [B, text_dim] 用于 IPAN
        """
        self.eval()
        with torch.no_grad():
            return self.extract_features(x, text_embedding)

    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1
