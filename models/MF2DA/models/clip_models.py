from .clip import clip


from PIL import Image
import torch
import torch.nn as nn


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, resnet_output_dim=768, fusion_hidden_dim=768):
        super(CLIPModel, self).__init__()
        
        # 加载特征提取模型
        self.model, self.preprocess = clip.load(name)
        
        # 延迟初始化分类器，在第一次前向传播时确定特征维度
        self.num_classes = num_classes
        self.fc = None
        self.feature_dim = None

    def _init_fc(self, feature_dim):
        """根据特征维度初始化分类器"""
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, self.num_classes)
        )
        
        # 如果模型已经在某个设备上，将分类器移到相同设备
        if next(self.model.parameters()).is_cuda:
            self.fc = self.fc.cuda()

    def forward(self, x, return_feature=False):
        # 获取融合特征
        fused_features, feature_dict = self.model(x)
        
        # 如果分类器尚未初始化，根据特征维度初始化
        if self.fc is None:
            self._init_fc(fused_features.shape[1])
        
        # 分类
        logits = self.fc(fused_features)
        
        if return_feature:
            # 返回分类结果和所有特征
            feature_dict['logits'] = logits
            feature_dict['fused_features'] = fused_features
            return logits, feature_dict
        
        # 只返回分类结果
        return logits
        

    def extract_features(self, x):
        """只提取特征，不进行分类"""
        fused_features, feature_dict = self.model(x)
        return fused_features, feature_dict

