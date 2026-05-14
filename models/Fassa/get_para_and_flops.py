import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forgery_uniformer import ForgeryUniformerSegmentation, Attention, GroupedSparseAttention


def count_parameters(model: nn.Module) -> int:
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_hook(module: nn.Module, input: Tensor, output: Tensor, flops_dict: dict):
    """用于注册 hook 来计算 FLOPs"""
    module_name = module.__class__.__name__
    
    if isinstance(module, nn.Conv2d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        groups = module.groups
        
        if hasattr(module, 'output_size') and module.output_size is not None:
            output_size = module.output_size
        else:
            output_size = output.shape[2] * output.shape[3] if len(output.shape) == 4 else 1
        
        active_channels = in_channels // groups
        flops = active_channels * out_channels * kernel_size * output_size
        flops_dict['total'] += flops
        
    elif isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        flops = in_features * out_features
        if module.bias is not None:
            flops += out_features
        flops_dict['total'] += flops
        
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if hasattr(module, 'num_features'):
            flops = module.num_features * 2
            flops_dict['total'] += flops
            
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        if len(output.shape) == 4:
            output_size = output.shape[2] * output.shape[3]
            flops_dict['total'] += output_size


def compute_attention_flops(module: nn.Module, input: Tensor, output: Tensor, flops_dict: dict):
    """计算注意力机制的 FLOPs"""
    if hasattr(module, 'num_heads'):
        B, N, C = input[0].shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        
        if isinstance(module, GroupedSparseAttention):
            num_groups = module.num_groups
            group_size = N // num_groups
            
            qkv_flops = 3 * B * N * C * C
            attention_flops = 0
            for g in range(num_groups):
                attention_flops += B * group_size * N * head_dim
            attention_flops = attention_flops * num_heads * 2
            
            output_flops = B * N * C * C
            total_flops = qkv_flops + attention_flops + output_flops
        else:
            qkv_flops = 3 * B * N * C * C
            attention_flops = B * N * N * C
            output_flops = B * N * C * C
            total_flops = qkv_flops + attention_flops + output_flops
            
        flops_dict['total'] += total_flops


def compute_flops(model: nn.Module, input_size: tuple = (512, 512), batch_size: int = 1) -> dict:
    """计算模型的 FLOPs"""
    model.eval()
    flops_dict = {'total': 0}
    
    hooks = []
    
    def conv_flops_hook(module, input, output):
        if isinstance(module, nn.Conv2d) and module.in_channels > 0:
            in_h, in_w = input[0].shape[2:]
            out_h, out_w = output.shape[2:]
            kernel_h, kernel_w = module.kernel_size
            out_channels = module.out_channels
            
            if module.groups > 1:
                in_channels = module.in_channels // module.groups
            else:
                in_channels = module.in_channels
            
            flops = in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
            flops_dict['total'] += flops
            
    def linear_flops_hook(module, input, output):
        in_features = module.in_features
        out_features = module.out_features
        flops = in_features * out_features
        if module.bias is not None:
            flops += out_features
        flops_dict['total'] += flops
            
    def bn_flops_hook(module, input, output):
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            flops = input[0].numel() * 2
            flops_dict['total'] += flops
            
    def attention_flops_hook(module, input, output):
        if isinstance(module, (Attention, GroupedSparseAttention)):
            B, N, C = input[0].shape
            if isinstance(module, GroupedSparseAttention):
                num_groups = module.num_groups
                group_size = N // num_groups
                
                qkv_flops = 3 * B * N * C * (C // module.num_heads) * module.num_heads
                attention_flops = B * module.num_heads * num_groups * group_size * N * (C // module.num_heads)
                output_flops = B * N * C * C
                total_flops = qkv_flops + attention_flops + output_flops
            else:
                qkv_flops = 3 * B * N * C * (C // module.num_heads) * module.num_heads
                attention_flops = B * module.num_heads * N * N * (C // module.num_heads)
                output_flops = B * N * C * C
                total_flops = qkv_flops + attention_flops + output_flops
            flops_dict['total'] += total_flops
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'resnet' not in name.lower():
                hooks.append(module.register_forward_hook(conv_flops_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            hooks.append(module.register_forward_hook(bn_flops_hook))
        elif isinstance(module, (Attention, GroupedSparseAttention)):
            hooks.append(module.register_forward_hook(attention_flops_hook))
    
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    dummy_resnet_feature = torch.randn(batch_size, 64, input_size[0], input_size[1])
    
    with torch.no_grad():
        _ = model(dummy_input, dummy_resnet_feature)
    
    for hook in hooks:
        hook.remove()
    
    return flops_dict['total']


def print_model_summary(model: nn.Module, input_size: tuple = (512, 512), batch_size: int = 1):
    """打印模型参数量和 FLOPs"""
    print("=" * 60)
    print("ForgeryUniFormer 模型分析")
    print("=" * 60)
    print(f"\n输入图像尺寸: {input_size}")
    print(f"批次大小: {batch_size}")
    print(f"\n{'='*60}")
    print("参数量统计")
    print("=" * 60)
    
    total_params = count_parameters(model)
    print(f"总参数量: {total_params:,}")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"总参数量: {total_params / 1e9:.2f}B")
    
    encoder_params = 0
    decoder_params = 0
    resnet_params = 0
    
    for name, param in model.named_parameters():
        if 'encoder' in name and 'resnet' not in name.lower():
            encoder_params += param.numel()
        elif 'decoder' in name:
            decoder_params += param.numel()
        elif 'resnet' in name.lower():
            resnet_params += param.numel()
    
    print(f"\n编码器 (UniFormer) 参数量: {encoder_params:,} ({encoder_params / 1e6:.2f}M)")
    print(f"解码器 (UPerHead) 参数量: {decoder_params:,} ({decoder_params / 1e6:.2f}M)")
    print(f"ResNet分支 参数量: {resnet_params:,} ({resnet_params / 1e6:.2f}M)")
    
    print(f"\n{'='*60}")
    print("FLOPs 统计")
    print("=" * 60)
    
    flops = compute_flops(model, input_size, batch_size)
    print(f"总 FLOPs: {flops:,}")
    print(f"总 FLOPs: {flops / 1e9:.2f}G")
    
    print(f"\n{'='*60}")
    print("各模块参数量分布")
    print("=" * 60)
    
    module_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                module_name = name.split('.')[-1] if name else module.__class__.__name__
                if module_name not in module_params:
                    module_params[module_name] = 0
                module_params[module_name] += params
    
    sorted_params = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for name, params in sorted_params[:20]:
        print(f"  {name}: {params:,} ({params / 1e6:.2f}M)")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    model = ForgeryUniformerSegmentation(
        img_size=512,
        num_classes=2,
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        resnet_pretrained=False
    )
    
    print_model_summary(model, input_size=(512, 512), batch_size=1)
