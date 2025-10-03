# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
'''
https://monica.im/share/chat?shareId=r44FnEXHICr1c0Sy
    model = dict(
        type='DINOScaleAware',
        num_queries=900,
        
        # ✅ 归一化尺度范围（相对于图像短边）
        # 小目标：边长 < 4% 短边
        # 中等目标：4% ~ 12% 短边
        # 大目标：> 12% 短边
        scale_ranges=((0.0, 0.04), (0.04, 0.12), (0.12, 1.0)),
        
        query_scale_ratios=(0.5, 0.35, 0.15),
        default_img_shape=(800, 1333),  # 仅用于推理时的 fallback
        # ...
    )
'''

@MODELS.register_module()
class DINOScaleAware(DeformableDETR):
    r"""Scale-Aware DINO: DINO with scale-specific query embeddings
    for improved small object detection.
    
    This implementation modifies the original DINO by:
    1. Using separate query embeddings for different object scales
    2. Scale-aware top-k proposal selection
    3. Scale-specific initialization strategies
    
    Args:
        scale_ranges (tuple): Scale ranges for small, medium, large objects.
            Default: ((0, 32), (32, 96), (96, 1e5))
        query_scale_ratios (tuple): Ratio of queries for each scale.
            Default: (0.5, 0.35, 0.15) for (small, medium, large)
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, 
                 *args, 
                 scale_ranges=((0.0, 0.04), (0.04, 0.12), (0.12, 1.0)),  # 相对比例
                 query_scale_ratios=(0.5, 0.35, 0.15),
                 default_img_shape=(800, 1333),
                 dn_cfg: OptConfigType = None, 
                 **kwargs) -> None:
        self.scale_ranges = scale_ranges
        self.query_scale_ratios = query_scale_ratios
        super().__init__(*args, **kwargs)
        
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers with scale-aware query embeddings."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        
        # ============ 核心改动：尺度感知的 Query Embedding ============
        # 计算每个尺度的 query 数量
        self.num_small_queries = int(self.num_queries * self.query_scale_ratios[0])
        self.num_medium_queries = int(self.num_queries * self.query_scale_ratios[1])
        self.num_large_queries = self.num_queries - self.num_small_queries - self.num_medium_queries
        
        # 为不同尺度创建独立的 embedding
        self.small_query_embed = nn.Embedding(self.num_small_queries, self.embed_dims)
        self.medium_query_embed = nn.Embedding(self.num_medium_queries, self.embed_dims)
        self.large_query_embed = nn.Embedding(self.num_large_queries, self.embed_dims)
        
        # 保留原始接口（用于兼容性）
        self.query_embedding = None  # 标记为不使用统一的 embedding
        # ============ 改动结束 ============

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights with scale-specific strategies."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        
        # ============ 核心改动：尺度特定的初始化 ============
        # 小目标：更小的初始化方差，更密集的特征
        nn.init.normal_(self.small_query_embed.weight, std=0.01)
        
        # 中等目标：标准初始化
        nn.init.normal_(self.medium_query_embed.weight, std=0.02)
        
        # 大目标：更大的初始化方差，更稀疏的特征
        nn.init.normal_(self.large_query_embed.weight, std=0.05)
        # ============ 改动结束 ============
        
        normal_(self.level_embed)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer."""
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def _compute_bbox_areas(
      self, 
      bboxes: Tensor, 
      batch_data_samples: OptSampleList = None,
      default_img_shape: tuple = (800, 1333)
  ) -> Tensor:
        """计算 bbox 的边长（支持 2D 和 3D 输入）"""
        # 统一处理维度
        if bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        bs = bboxes.shape[0]
        areas_list = []
        
        for b in range(bs):
            # 获取图像尺寸
            if batch_data_samples and len(batch_data_samples) > b:
                metainfo = batch_data_samples[b].metainfo
                img_h = metainfo.get('img_shape', self.default_img_shape)[0]
                img_w = metainfo.get('img_shape', self.default_img_shape)[1]
            else:
                img_h, img_w = self.default_img_shape
            
            # 计算边长（从 normalized 坐标转换）
            widths = bboxes[b, :, 2] * img_w
            heights = bboxes[b, :, 3] * img_h
            areas = (widths * heights).sqrt().clamp(min=1e-6)
            areas_list.append(areas)
        
        result = torch.stack(areas_list, dim=0)
        return result.squeeze(0) if squeeze_output else result
    
    def _compute_bbox_areas_normalized(
        self, 
        bboxes: Tensor,
        batch_data_samples: OptSampleList = None
    ) -> Tensor:
        """计算归一化的 bbox 边长（相对于图像短边）"""
        if bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        bs = bboxes.shape[0]
        areas_list = []
        
        for b in range(bs):
            # 获取图像尺寸
            if batch_data_samples and len(batch_data_samples) > b:
                metainfo = batch_data_samples[b].metainfo
                img_h = metainfo.get('img_shape', self.default_img_shape)[0]
                img_w = metainfo.get('img_shape', self.default_img_shape)[1]
            else:
                img_h, img_w = self.default_img_shape
            
            # ✅ 关键：归一化到短边
            min_size = min(img_h, img_w)
            
            # 计算归一化边长
            widths = bboxes[b, :, 2] * img_w / min_size
            heights = bboxes[b, :, 3] * img_h / min_size
            areas = (widths * heights).sqrt().clamp(min=1e-6)
            areas_list.append(areas)
        
        result = torch.stack(areas_list, dim=0)
        return result.squeeze(0) if squeeze_output else result

    def _scale_aware_topk_selection(
        self,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_size: int,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """尺度感知的 top-k 选择策略
        
        Args:
            enc_outputs_class (Tensor): (bs, num_proposals, num_classes)
            enc_outputs_coord (Tensor): (bs, num_proposals, 4)
            batch_size (int): batch size
            
        Returns:
            tuple: (topk_indices, topk_scores, topk_coords)
        """
        cls_out_features = enc_outputs_class.shape[-1]
        
        # 计算每个 proposal 的置信度和尺度
        proposal_scores = enc_outputs_class.max(-1)[0]  # (bs, num_proposals)
        
        # 为每个 batch 分别处理
        all_topk_indices = []
        all_topk_scores = []
        all_topk_coords = []
        
        for b in range(batch_size):
            scores_b = proposal_scores[b]  # (num_proposals,)
            coords_b = enc_outputs_coord[b]  # (num_proposals, 4)
            class_b = enc_outputs_class[b]  # (num_proposals, num_classes)
            
            # 计算每个 proposal 的尺度
            areas = self._compute_bbox_areas_normalized(
                coords_b, 
                batch_data_samples[b:b+1] if batch_data_samples else None
            )
            
            # ✅ 使用归一化的尺度范围
            valid_mask = torch.isfinite(areas) & (areas > 0)
            small_mask = valid_mask & (areas >= self.scale_ranges[0][0]) & (areas < self.scale_ranges[0][1])
            medium_mask = valid_mask & (areas >= self.scale_ranges[1][0]) & (areas < self.scale_ranges[1][1])
            large_mask = valid_mask & (areas >= self.scale_ranges[2][0])
            
            # ✅ 修复：避免重复选择
            def safe_topk(mask, k, exclude_indices=None):
                masked_scores = scores_b.clone()
                masked_scores[~mask] = -1e10
                if exclude_indices is not None and len(exclude_indices) > 0:
                    masked_scores[exclude_indices] = -1e10
                
                available = (masked_scores > -1e9).sum().item()
                k_actual = min(k, available, len(masked_scores))
                
                if k_actual > 0:
                    return torch.topk(masked_scores, k=k_actual)[1]
                return torch.tensor([], dtype=torch.long, device=scores_b.device)
            
            selected = []
            for mask, num_q in [(small_mask, self.num_small_queries),
                                (medium_mask, self.num_medium_queries),
                                (large_mask, self.num_large_queries)]:
                indices = safe_topk(mask, num_q, 
                                   torch.cat(selected) if selected else None)
                selected.append(indices)
            
            topk_indices_b = torch.cat(selected)
            
            # 收集结果
            topk_scores_b = class_b[topk_indices_b]
            topk_coords_b = coords_b[topk_indices_b]
            
            all_topk_indices.append(topk_indices_b)
            all_topk_scores.append(topk_scores_b)
            all_topk_coords.append(topk_coords_b)
        
        topk_indices = torch.stack(all_topk_indices, dim=0)  # (bs, num_queries)
        topk_scores = torch.stack(all_topk_scores, dim=0)    # (bs, num_queries, num_classes)
        topk_coords = torch.stack(all_topk_coords, dim=0)    # (bs, num_queries, 4)
        
        return topk_indices, topk_scores, topk_coords

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder."""
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # ============ 核心改动：尺度感知的 top-k 选择 ============
        topk_indices, topk_score, topk_coords_unact = \
            self._scale_aware_topk_selection(
                enc_outputs_class, enc_outputs_coord_unact, bs, batch_data_samples=batch_data_samples)
        # ============ 改动结束 ============
        
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # ============ 核心改动：组装尺度感知的 query embedding ============
        # 拼接不同尺度的 query embeddings
        small_query = self.small_query_embed.weight    # (num_small, dim)
        medium_query = self.medium_query_embed.weight  # (num_medium, dim)
        large_query = self.large_query_embed.weight    # (num_large, dim)
        
        query_embed = torch.cat([small_query, medium_query, large_query], dim=0)  # (num_queries, dim)
        query = query_embed[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # (bs, num_queries, dim)
        # ============ 改动结束 ============
        
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder."""
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
