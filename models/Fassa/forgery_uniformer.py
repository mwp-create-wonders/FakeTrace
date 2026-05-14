import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from resnet_branch import ResNetBranch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupedSparseAttention(nn.Module):
    """基于分数排序的分组稀疏自注意力"""
    def __init__(self, dim, num_heads=8, num_groups=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, sorted_indices):
        """
        Args:
            x: [B, N, C] 输入特征
            sorted_indices: [B, N] 按分数降序排列的索引
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads
        
        # 计算 q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # 初始化注意力矩阵
        attn = torch.zeros(B, self.num_heads, N, N, device=x.device)
        group_size = N // self.num_groups
        
        # 计算分组稀疏注意力
        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < self.num_groups - 1 else N
            
            group_indices = sorted_indices[:, start_idx:end_idx]
            
            for b in range(B):
                for h in range(self.num_heads):
                    # 获取当前组的索引
                    idx = group_indices[b]  # [group_size]
                    
                    # 计算当前组的注意力
                    q_g = q[b, h, idx]  # [group_size, head_dim]
                    k_g = k[b, h, :]    # [N, head_dim]
                    
                    # 注意力分数: [group_size, N]
                    attn_g = (q_g @ k_g.transpose(-2, -1)) * self.scale
                    attn_g = attn_g.softmax(dim=-1)
                    
                    # 赋值到注意力矩阵
                    attn[b, h, idx, :] = attn_g
        
        # 应用 dropout
        attn = self.attn_drop(attn)
        
        # 计算注意力加权和
        # attn: [B, num_heads, N, N]
        # v: [B, num_heads, N, head_dim]
        # 矩阵乘法: [B, num_heads, N, N] @ [B, num_heads, N, head_dim] = [B, num_heads, N, head_dim]
        x = attn @ v
        
        # 重组维度
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, num_heads * head_dim] = [B, N, C]
        
        # 投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SparseAttentionBlock(nn.Module):
    """稀疏自注意力块"""
    def __init__(self, dim, num_heads, num_groups=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GroupedSparseAttention(
            dim, num_heads=num_heads, num_groups=num_groups, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.num_groups = num_groups

    def forward(self, x, sorted_indices=None):
        B, N, H, W = x.shape
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        if sorted_indices is not None and self.num_groups > 1:
            x = x + self.drop_path(self.attn(self.norm1(x), sorted_indices))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), sorted_indices))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class GlobalAttentionBlock(nn.Module):
    """全局自注意力块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, H, W = x.shape
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ForgeryUniFormer(nn.Module):
    def __init__(self, layers=[5, 8, 20, 7], img_size=512, in_chans=3, 
                 embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 resnet_pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        
        # 初始化ResNet分支
        self.resnet_branch = ResNetBranch(pretrained=resnet_pretrained)
        
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        num_heads = [dim // head_dim for dim in embed_dim]

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(layers[0])])
        self.norm1 = norm_layer(embed_dim[0])
        
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]], norm_layer=norm_layer)
            for i in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])

        self.blocks3_stage_groups = nn.ModuleList([
            nn.ModuleList([
                GlobalAttentionBlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]+j], norm_layer=norm_layer)
                if j == 4 else SparseAttentionBlock(
                    dim=embed_dim[2], num_heads=num_heads[2], num_groups=16>>j, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+layers[0]+layers[1]+j], norm_layer=norm_layer)
                for j in range(5)
            ])
            for i in range(4)
        ])
        self.norm3 = norm_layer(embed_dim[2])

        self.blocks4_stage_groups = nn.ModuleList([
            nn.ModuleList([
                GlobalAttentionBlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layers[0]+layers[1]+layers[2]+j], norm_layer=norm_layer)
                if j == 3 else SparseAttentionBlock(
                    dim=embed_dim[3], num_heads=num_heads[3], num_groups=8>>j, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layers[0]+layers[1]+layers[2]+j], norm_layer=norm_layer)
                for j in range(4)
            ]),
            nn.ModuleList([
                GlobalAttentionBlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layers[0]+layers[1]+layers[2]+4+j], norm_layer=norm_layer)
                if j == 2 else SparseAttentionBlock(
                    dim=embed_dim[3], num_heads=num_heads[3], num_groups=4>>j, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[layers[0]+layers[1]+layers[2]+4+j], norm_layer=norm_layer)
                for j in range(3)
            ])
        ])
        self.norm4 = norm_layer(embed_dim[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_patch_scores_and_indices(self, resnet_out, group_type, target_size=None):
        """
        根据ResNet输出计算patch分数并排序返回索引
        """
        x = resnet_out.mean(dim=1, keepdim=True)  # 保持通道维度，得到 [B, 1, H, W]
        
        if group_type == 0:
            pass
        elif group_type == 1:
            x = F.adaptive_avg_pool2d(x, (64, 64))
        elif group_type == 2:
            x = F.adaptive_avg_pool2d(x, (32, 32))
        else:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        # 移除通道维度，得到 [B, H, W]
        x = x.squeeze(1)
        
        if group_type in [0, 1]:
            ws = 4 if group_type == 0 else 2
            B, H, W = x.shape
            x = x.unfold(1, ws, ws).unfold(2, ws, ws)
            x = x.mean(dim=[3, 4])
        
        scores_flat = x.flatten(1)
        sorted_indices = torch.argsort(scores_flat, dim=1, descending=True)
        
        return sorted_indices

    def compute_stage4_indices(self, resnet_out, group_type):
        """
        计算Stage4的patch分数和索引
        """
        x = resnet_out.mean(dim=1, keepdim=True)  # 保持通道维度，得到 [B, 1, H, W]
        
        if group_type == 0:
            x = F.adaptive_avg_pool2d(x, (32, 32))
            # 移除通道维度，得到 [B, H, W]
            x = x.squeeze(1)
            B, H, W = x.shape
            ws = 2
            x = x.unfold(1, ws, ws).unfold(2, ws, ws)
            x = x.mean(dim=[3, 4])
        else:
            x = F.adaptive_avg_pool2d(x, (16, 16))
            # 移除通道维度，得到 [B, H, W]
            x = x.squeeze(1)
        
        scores_flat = x.flatten(1)
        sorted_indices = torch.argsort(scores_flat, dim=1, descending=True)
        
        return sorted_indices

    def forward_features(self, x, resnet_feature):
        """
        Args:
            x: 输入图像 [B, 3, 512, 512]
            resnet_feature: ResNet输入特征 [B, 64, 512, 512]
        """
        resnet_outs = {}
        
        # 获取ResNet分支的输出
        out_1, out_2, out_3, out_4 = self.resnet_branch(resnet_feature)
        resnet_outs['out_1'] = out_1
        resnet_outs['out_2'] = out_2
        resnet_outs['out_3'] = out_3
        resnet_outs['out_4'] = out_4
        
        # UniFormer前两个stage
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x_out1 = self.norm1(x.permute(0, 2, 3, 1))
        resnet_outs['stage1'] = x_out1.permute(0, 3, 1, 2).contiguous()
        
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x_out2 = self.norm2(x.permute(0, 2, 3, 1))
        resnet_outs['stage2'] = x_out2.permute(0, 3, 1, 2).contiguous()
        
        # Stage 3: 使用ResNet分支的输出来指导稀疏注意力
        x = self.patch_embed3(x)
        
        stage3_sorted_indices = [
            self.compute_patch_scores_and_indices(out_1, 0),  # out_1: [B, 256, 128, 128]
            self.compute_patch_scores_and_indices(out_2, 1),  # out_2: [B, 512, 64, 64]
            self.compute_patch_scores_and_indices(out_3, 2),  # out_3: [B, 1024, 32, 32]
            self.compute_patch_scores_and_indices(out_4, 3)   # out_4: [B, 2048, 16, 16]
        ]
        
        for group_idx, group_blocks in enumerate(self.blocks3_stage_groups):
            sorted_indices = stage3_sorted_indices[group_idx]
            for block_idx, blk in enumerate(group_blocks):
                if isinstance(blk, SparseAttentionBlock):
                    x = blk(x, sorted_indices)
                else:
                    x = blk(x)
        
        x_out3 = self.norm3(x.permute(0, 2, 3, 1))
        resnet_outs['stage3'] = x_out3.permute(0, 3, 1, 2).contiguous()
        
        # Stage 4: 使用ResNet分支的输出来指导稀疏注意力
        x = self.patch_embed4(x)
        
        stage4_sorted_indices = [
            self.compute_stage4_indices(out_3, 0),  # out_3: [B, 1024, 32, 32]
            self.compute_stage4_indices(out_4, 1)   # out_4: [B, 2048, 16, 16]
        ]
        
        for group_idx, group_blocks in enumerate(self.blocks4_stage_groups):
            sorted_indices = stage4_sorted_indices[group_idx]
            for block_idx, blk in enumerate(group_blocks):
                if isinstance(blk, SparseAttentionBlock):
                    x = blk(x, sorted_indices)
                else:
                    x = blk(x)
        
        x_out4 = self.norm4(x.permute(0, 2, 3, 1))
        resnet_outs['stage4'] = x_out4.permute(0, 3, 1, 2).contiguous()
        
        return resnet_outs

    def forward(self, x, resnet_feature):
        return self.forward_features(x, resnet_feature)


class UPerHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], channels=512, num_classes=19, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        
        self.psp_modules = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in pool_scales
        ])
        
        self.psp_conv = nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.bn_psp = nn.BatchNorm2d(channels)
        self.relu_psp = nn.ReLU(inplace=True)
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, channels, 1) for in_ch in in_channels[:-1]
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1) for _ in in_channels[:-1]
        ])
        
        self.fpn_bottleneck = nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1)
        self.bn_fpn = nn.BatchNorm2d(channels)
        self.relu_fpn = nn.ReLU(inplace=True)
        
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def psp_forward(self, x):
        input_size = x.size()[2:]
        psp_outs = [x]
        for psp in self.psp_modules:
            psp_outs.append(F.interpolate(psp(x), size=input_size, mode='bilinear', align_corners=False))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.relu_psp(self.bn_psp(self.psp_conv(psp_outs)))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs[-1]))
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False)
        
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        
        output = self.relu_fpn(self.bn_fpn(self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))))
        output = self.cls_seg(output)
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        
        return output


class ForgeryUniformerSegmentation(nn.Module):
    def __init__(self, img_size=512, in_chans=3, num_classes=2, embed_dim=[64, 128, 320, 512],
                 layers=[5, 8, 20, 7], resnet_pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.encoder = ForgeryUniFormer(
            layers=layers, 
            img_size=img_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            resnet_pretrained=resnet_pretrained
        )
        
        self.decoder = UPerHead(
            in_channels=embed_dim,
            channels=embed_dim[-1],
            num_classes=num_classes
        )

    def forward(self, x, resnet_feature):
        """
        Args:
            x: 输入图像 [B, 3, 512, 512]
            resnet_feature: ResNet输入特征 [B, 64, 512, 512]
        """
        encoder_outputs = self.encoder(x, resnet_feature)
        outputs = self.decoder([
            encoder_outputs['stage1'],
            encoder_outputs['stage2'],
            encoder_outputs['stage3'],
            encoder_outputs['stage4']
        ])
        return outputs


if __name__ == '__main__':
    model = ForgeryUniformerSegmentation(img_size=512, num_classes=2)
    x = torch.randn(1, 3, 512, 512)  # 输入图像
    resnet_feature = torch.randn(1, 64, 512, 512)  # 预先提取的特征
    output = model(x, resnet_feature)
    print(f"Output shape: {output.shape}")