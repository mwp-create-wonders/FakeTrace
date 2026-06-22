import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt

        # 返回一个基于CLIP的模型
        self.model = get_model(opt.arch)

        # 固定backbone
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                # 冻结CLIP分支的所有参数
                if name.startswith('model.clip_branch.'):
                    p.requires_grad = False
                    # print(f"冻结: {name}")
                else:
                    # 其他参数保持可训练
                    p.requires_grad = True
                    params.append(p)
                    # print(f"可训练: {name}")
            
            # 如果没有找到可训练参数，使用所有参数作为后备
            if len(params) == 0:
                print("警告：没有找到可训练参数，将使用所有参数")
                params = self.model.parameters()
                for p in params:
                    p.requires_grad = True
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        # 确保模型至少有一次前向传播，以初始化动态分类器
        if hasattr(self.model, 'classifier') and self.model.classifier is None:
            print("执行一次前向传播以初始化分类器...")
            dummy_input = torch.randn(1, 3, 224, 224).to(opt.gpu_ids[0])
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            # 如果分类器已初始化，确保其参数可训练
            if self.model.classifier is not None:
                for name, param in self.model.classifier.named_parameters():
                    param.requires_grad = True
                    params.append(param)
                    print(f"分类器可训练: {name}")
        
        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {frozen_params:,}")
        print(f"可训练比例: {trainable_params/total_params*100:.2f}%")
        
        # 挑选激活函数
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        # 在训练时已经固定参数了
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



