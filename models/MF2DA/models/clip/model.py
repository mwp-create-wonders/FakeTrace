from collections import OrderedDict
from typing import Tuple, Union, List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .SFSResnet import create_SFSResnet_extractor


# ==================== 基础组件 ====================
class LayerNorm(nn.LayerNorm):
    """解决fp16训练时数值不稳定的问题"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """残差注意力块"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out[f'layer{idx}'] = x[0]  # shape:LND. choose cls token feature
        return out, x


class VisionTransformer(nn.Module):
    """视觉Transformer"""
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [*, grid ** 2, width]
        
        # 添加CLS和位置编码
        cls_embed = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_embed, x], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer编码
        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        
        out['before_projection'] = x  
        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x 

        return x


# ==================== 基础CLIP模型 ====================
class CLIP(nn.Module):
    """基础CLIP模型"""
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()

        self.context_length = context_length

        # 视觉编码器
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        # 文本编码器
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self._build_attention_mask()
        )

        # 文本编码参数
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def _build_attention_mask(self):
        """构建因果注意力掩码"""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 下三角为零，上三角为负无穷
        return mask

    def initialize_parameters(self):
        """参数初始化"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 处理Transformer的输出，它返回元组 (out, x)
        out, x = self.transformer(x)  # 直接解包元组
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 特征归一化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


# ==================== 融合模块 ====================
class CrossModalAttentionFusion(nn.Module):
    """跨模态注意力融合"""
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            QuickGELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        visual_proj = self.visual_proj(visual_features.to(self.visual_proj.weight.dtype)).unsqueeze(1)
        text_proj = self.text_proj(text_features.to(self.text_proj.weight.dtype)).unsqueeze(1)
        
        # 跨模态注意力
        attended_visual, attention_weights = self.cross_attention(
            query=visual_proj, 
            key=text_proj, 
            value=text_proj,
            need_weights=True
        )
        
        # 残差连接
        visual_enhanced = self.ln1(visual_proj + attended_visual)
        mlp_output = self.mlp(visual_enhanced)
        visual_enhanced = self.ln2(visual_enhanced + mlp_output)
        
        fused_features = self.output_proj(visual_enhanced.squeeze(1))
        return fused_features, attention_weights


class HierarchicalGatedFusion(nn.Module):
    """层次化门控融合"""
    def __init__(self, visual_dim: int, text_dim: int, num_gates: int = 3, reduction_ratio: int = 4):
        super().__init__()
        self.num_gates = num_gates
        
        self.gates = nn.ModuleList()
        self.visual_transformers = nn.ModuleList()

        for i in range(num_gates):
            gate_net = nn.Sequential(
                nn.Linear(visual_dim + text_dim, (visual_dim + text_dim) // reduction_ratio),
                nn.ReLU(),
                nn.Linear((visual_dim + text_dim) // reduction_ratio, visual_dim),
                nn.Sigmoid()
            )
            self.gates.append(gate_net)
            
            visual_transformer = nn.Sequential(
                nn.Linear(visual_dim, visual_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.visual_transformers.append(visual_transformer)
        
        self.feature_aggregation = nn.Sequential(
            nn.Linear(visual_dim * num_gates, visual_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        batch_size = visual_features.size(0)
        fused_features_list = []
        current_visual = visual_features
        
        for i in range(self.num_gates):
            gate_input = torch.cat([current_visual, text_features], dim=1)
            gate_weights = self.gates[i](gate_input)
            transformed_visual = self.visual_transformers[i](current_visual)
            gated_visual = gate_weights * transformed_visual + (1 - gate_weights) * current_visual
            fused_features_list.append(gated_visual)
            current_visual = gated_visual
        
        all_features = torch.cat(fused_features_list, dim=1)
        final_fused = self.feature_aggregation(all_features)
        return final_fused


class MultiModalFusionNetwork(nn.Module):
    """多模态融合网络"""
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 多种融合策略
        self.cross_modal_fusion = CrossModalAttentionFusion(self.visual_dim, self.text_dim, self.hidden_dim)
        self.hierarchical_fusion = HierarchicalGatedFusion(self.visual_dim, self.text_dim)
        
        # 融合策略选择器
        self.fusion_selector = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 2种融合策略
            nn.Softmax(dim=1)
        )
        
        # 特征归一化
        self.visual_norm = LayerNorm(visual_dim)
        self.text_norm = LayerNorm(text_dim)

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        # 特征归一化
        visual_norm = self.visual_norm(visual_features)
        text_norm = self.text_norm(text_features)
        
        # 计算融合策略权重
        selector_input = torch.cat([visual_norm, text_norm], dim=1)
        fusion_weights = self.fusion_selector(selector_input)
        
        # 各种融合策略
        cross_modal_feat, attention_weights = self.cross_modal_fusion(visual_norm, text_norm)
        hierarchical_feat = self.hierarchical_fusion(visual_norm, text_norm)

        # 加权融合
        fused_features = (fusion_weights[:, 0:1] * cross_modal_feat +
                         fusion_weights[:, 1:2] * hierarchical_feat)
        
        # 与原始视觉特征拼接
        final_features = torch.cat([fused_features, visual_norm], dim=1)
        
        return final_features


# ==================== 文本特征管理器 ====================
class TextFeatureManager:
    """文本特征管理器"""
    def __init__(self, clip_model: nn.Module, context_length: int = 77):
        self.clip = clip_model
        self.context_length = context_length
        self.tokenizer = _Tokenizer()
        
        # 预定义真伪鉴别提示词
        # self.authenticity_prompts = [
        #     "a authentic real photograph with natural lighting",
        #     "a digitally manipulated fake image with artifacts",
        #     "a genuine image with consistent shadows and textures", 
        #     "a forged picture showing editing traces and inconsistencies",
        #     "a real photo with smooth gradients and natural colors",
        #     "a synthetic image with artificial patterns and repetitions"
        # ]

        self.authenticity_prompts = [
            # 真实图像
            "natural lighting and shadows",
            "realistic texture details",
            "authentic color gradation",
            "subtle film grain noise",
            "imperfect human features",
            "believable skin textures",
            "consistent object proportions",
            "natural depth of field",
            "real-world environmental details",
            "spontaneous moment captured",
            "organic material appearance",
            "convincing hair strands",
            "natural posture and expression",
            "authentic wear and tear",
            "plausible spatial relationships",
        
            # AI生成/伪造图像
            "unnatural smoothness",
            "illogical shadow direction",
            "blurred background artifacts",
            "repetitive pattern details",
            "distorted facial features",
            "inconsistent texture scaling",
            "oversaturated color palette",
            "floating object anomalies",
            "impossible object physics",
            "misaligned edges and seams",
            "watercolor-like artifacts",
            "extra or missing fingers",
            "nonsensical text elements",
            "glowing contour lines",
            "mismatched lighting sources"
        ]
        
        # TODO 不使用文本引导
        # self.authenticity_prompts = ["a picture"]
        
        self.cached_text_features = None
        self._init_text_features()

    def _tokenize(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Tokenize文本"""
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)
        
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def _init_text_features(self):
        """初始化文本特征"""
        with torch.no_grad():
            device = next(self.clip.parameters()).device
            prompt_tokens = self._tokenize(self.authenticity_prompts).to(device)
            text_features = self.clip.encode_text(prompt_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            self.cached_text_features = text_features

    def get_dynamic_text_features(self, image_features: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        """动态选择最相关的文本提示"""
        if self.cached_text_features is None:
            raise ValueError("Text features not initialized")
        
        # 确保文本特征与图像特征在同一设备上
        device = image_features.device
        cached_text_features = self.cached_text_features.to(device)
        
        similarities = torch.matmul(image_features, cached_text_features.T)  # [batch, num_prompts]
        topk_similarities, topk_indices = torch.topk(similarities, k=top_k, dim=1)
        
        batch_size = image_features.size(0)
        selected_text_features = []
        
        for i in range(batch_size):
            weights = F.softmax(topk_similarities[i], dim=0)
            weighted_features = torch.zeros_like(cached_text_features[0])
            
            for j, idx in enumerate(topk_indices[i]):
                weighted_features += weights[j] * cached_text_features[idx]
            
            selected_text_features.append(weighted_features)
        
        return torch.stack(selected_text_features)

    def get_default_text_features(self, batch_size: int) -> torch.Tensor:
        """获取默认文本特征"""
        if self.cached_text_features is None:
            raise ValueError("Text features not initialized")
        
        # 获取模型当前设备
        device = next(self.clip.parameters()).device
        cached_text_features = self.cached_text_features.to(device)
        
        text_features = cached_text_features.mean(dim=0, keepdim=True)
        text_features = text_features.expand(batch_size, -1)
        return text_features


# ==================== 双分支CLIP模型 ====================
class DualBranchCLIPModel(nn.Module):
    """
    双分支CLIP模型：融合CLIP-ViT和SFSResNet特征
    """
    def __init__(self, 
                 clip_model: nn.Module,
                 resnet_output_dim: int = 768,
                 num_classes: int = 1,
                 fusion_hidden_dim: int = 768,
                 use_attention_pool: bool = True):
        super().__init__()
        
        # 第一分支：CLIP模型
        self.clip_branch = clip_model

        # 查询clip属性
        # print("CLIP branch attributes:", dir(self.clip_branch))
        # print("CLIP branch visual attributes:", dir(self.clip_branch.visual) if hasattr(self.clip_branch, 'visual') else "No visual")
        # print("CLIP branch text attributes:", dir(self.clip_branch.text) if hasattr(self.clip_branch, 'text') else "No text")

        # 获取clip.visual输出维度
        clip_visual_dim = self.clip_branch.visual.output_dim
        clip_text_dim = self.clip_branch.text_projection.shape[1]

        # 获取输入分辨率信息
        if hasattr(self.clip_branch.visual, 'input_resolution'):
            self.input_resolution = self.clip_branch.visual.input_resolution
        else:
            self.input_resolution = 224  # CLIP默认输入尺寸
        
        # 第二分支：SFSResNet特征提取器
        # 独立的
        self.resnet_branch = create_SFSResnet_extractor(
            "resnet50", 
            output_dim=resnet_output_dim,
            use_attention_pool=use_attention_pool,
            input_resolution=self.input_resolution
        )
        
        # 文本特征管理器
        self.text_manager = TextFeatureManager(self.clip_branch)
        
        # 特征融合网络
        self.fusion_network = MultiModalFusionNetwork(
            visual_dim=clip_visual_dim + resnet_output_dim,
            text_dim=clip_text_dim,
            hidden_dim=fusion_hidden_dim,
        )
        

    def forward(self, 
                image: torch.Tensor, 
                use_dynamic_prompts: bool = True, 
                # TODO 不使用文本引导，选择1，使用文本引导，使用其他
                top_k: int = 2):
        """
        Args:
            image: 输入图像 [batch_size, 3, H, W]
            use_dynamic_prompts: 是否使用动态提示选择
            top_k: 动态选择时使用的top-k数量
        
        Returns:
            fused_features: 融合后的特征 [batch_size, fusion_hidden_dim + clip_output_dim]
            feature_dict: 包含所有中间特征的字典（用于分析和可视化）
        """
        batch_size = image.size(0)
        
        # 第一分支：CLIP特征提取
        clip_features = self.clip_branch.encode_image(image)
        clip_features = F.normalize(clip_features, p=2, dim=1)
        
        # 第二分支：ResNet特征提取（融合后的特征）
        resnet_features_dict = self.resnet_branch(image)

        # 原始版本
        # resnet_features = resnet_features_dict['multi_layer_fused']

        # TODO 消融实验部分设计，无金字塔
        # resnet_features = resnet_features_dict['final_pooled']

        # TODO 消融实验部分设计，传统resnet
        resnet_features = resnet_features_dict['Resnet_feature']
        
        
        # 获取文本特征
        if use_dynamic_prompts:
            text_features = self.text_manager.get_dynamic_text_features(clip_features, top_k=top_k)
        else:
            text_features = self.text_manager.get_default_text_features(batch_size)
        
        # 融合两个视觉分支的特征
        combined_visual_features = torch.cat([clip_features, resnet_features], dim=1)

        # print(f"combined_visual_features的形状 - combined_visual_features: {combined_visual_features.shape}")

        # print(f"text_features的形状 - text_features: {text_features.shape}")
        
        # 文本特征引导融合
        fused_features = self.fusion_network(combined_visual_features, text_features)
        
        # 返回融合特征和所有中间特征
        feature_dict = {
            'clip_features': clip_features,
            'resnet_features': resnet_features,
            'combined_visual_features': combined_visual_features,
            'text_features': text_features,
            'fused_features': fused_features
        }
        
        return fused_features, feature_dict

    def get_feature_maps(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各层的特征图"""
        clip_features = self.clip_branch.encode_image(image)
        resnet_features = self.resnet_branch(image)
        
        all_features = {
            'clip_final': clip_features,
            **resnet_features
        }
        return all_features


# ==================== 模型构建函数 ====================
def convert_weights_to_fp16(model: nn.Module):
    """将模型参数转换为fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_base_clip(state_dict: dict) -> CLIP:
    """构建基础CLIP模型"""
    assert "visual.proj" in state_dict, "Only ViT models are supported"
    
    # 视觉编码器参数
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    # 文本编码器参数
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 构建模型
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )

    # 加载权重
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def build_dual_branch_model(state_dict: dict,
                           resnet_output_dim: int = 768,
                           fusion_hidden_dim: int = 768) -> DualBranchCLIPModel:
    """构建双分支CLIP模型"""
    # 构建基础CLIP
    base_clip = build_base_clip(state_dict)
    
    # 构建双分支模型
    model = DualBranchCLIPModel(
        clip_model=base_clip,
        resnet_output_dim=resnet_output_dim,
        fusion_hidden_dim=fusion_hidden_dim
    )
    
    return model