import torch
import torch.nn as nn
import torchvision.models as models
from spatial_spectrum_compression import SACCModule


class ResNetBranch(nn.Module):
    """
    ResNet分支模块
    
    功能：
    1. 接收512×512×64的特征张量
    2. 使用SACC模块压缩为512×512×16
    3. 输入到预训练的ResNet50
    4. 输出4个stage的特征张量
    """
    
    def __init__(self, pretrained=True):
        super(ResNetBranch, self).__init__()
        
        # 空间频谱压缩模块
        self.sacc_module = SACCModule(
            in_channels=64,  # 输入通道数
            out_channels=16,  # 输出通道数
            num_scales=3  # 多尺度分支数
        )
        
        # 使用torchvision的ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 替换第一层卷积，适配16通道输入
        # 原始ResNet50的第一层是7x7卷积，输入3通道
        # 这里修改为16通道输入
        self.resnet.conv1 = nn.Conv2d(
            in_channels=16,  # 输入通道数
            out_channels=64,  # 输出通道数
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 注册钩子用于获取中间特征
        self.stage_outputs = {}
        
        # 为每个stage注册钩子
        def get_stage_output(name):
            def hook(module, input, output):
                self.stage_outputs[name] = output
            return hook
        
        # Stage 1: layer1
        self.resnet.layer1.register_forward_hook(get_stage_output('stage1'))
        # Stage 2: layer2
        self.resnet.layer2.register_forward_hook(get_stage_output('stage2'))
        # Stage 3: layer3
        self.resnet.layer3.register_forward_hook(get_stage_output('stage3'))
        # Stage 4: layer4
        self.resnet.layer4.register_forward_hook(get_stage_output('stage4'))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量，形状为 [B, 64, 512, 512]
        
        Returns:
            out_1: Stage 1输出，形状为 [B, 256, 128, 128]
            out_2: Stage 2输出，形状为 [B, 512, 64, 64]
            out_3: Stage 3输出，形状为 [B, 1024, 32, 32]
            out_4: Stage 4输出，形状为 [B, 2048, 16, 16]
        """
        # 1. 使用SACC模块压缩特征
        x_compressed = self.sacc_module(x)
        
        # 2. 输入到ResNet50
        _ = self.resnet(x_compressed)
        
        # 3. 获取4个stage的输出
        out_1 = self.stage_outputs['stage1']
        out_2 = self.stage_outputs['stage2']
        out_3 = self.stage_outputs['stage3']
        out_4 = self.stage_outputs['stage4']
        
        return out_1, out_2, out_3, out_4


if __name__ == '__main__':
    # 测试代码
    model = ResNetBranch(pretrained=False)
    
    # 创建测试输入
    input_tensor = torch.randn(1, 64, 512, 512)
    
    # 前向传播
    out1, out2, out3, out4 = model(input_tensor)
    
    # 打印输出形状
    print(f"Stage 1 output shape: {out1.shape}")
    print(f"Stage 2 output shape: {out2.shape}")
    print(f"Stage 3 output shape: {out3.shape}")
    print(f"Stage 4 output shape: {out4.shape}")