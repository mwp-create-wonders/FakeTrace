import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import argparse

from forgery_uniformer import ForgeryUniformerSegmentation


class TamperingDataset(Dataset):
    """伪造图像检测数据集"""
    def __init__(self, tampered_dir, mask_dir, tensor_dir):
        self.tampered_dir = tampered_dir
        self.mask_dir = mask_dir
        self.tensor_dir = tensor_dir
        
        self.file_list = [f.replace('.npz', '') for f in os.listdir(tensor_dir) if f.endswith('.npz')]
        self.file_list.sort()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        tampered_path = os.path.join(self.tampered_dir, f"{file_name}.png")
        mask_path = os.path.join(self.mask_dir, f"{file_name}_mask.png")
        tensor_path = os.path.join(self.tensor_dir, f"{file_name}.npz")
        
        tampered_img = Image.open(tampered_path).convert('RGB')
        tampered_img = transforms.ToTensor()(tampered_img)
        tampered_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tampered_img)
        
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor = transforms.ToTensor()(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()
        
        tensor_data = np.load(tensor_path)
        feature_tensor = torch.from_numpy(tensor_data['tensor']).permute(2, 0, 1).float()
        
        return {
            'image': tampered_img,
            'mask': mask_tensor,
            'feature': feature_tensor,
            'name': file_name
        }


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        if pred.shape[1] > 1:
            pred = pred[:, 0:1, :, :]
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class EdgeExtractor(nn.Module):
    """边缘提取模块：生成掩码的边缘信息（用于边缘损失计算）"""
    def __init__(self):
        super(EdgeExtractor, self).__init__()
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)
        self.kernel = torch.tensor(laplacian_kernel).unsqueeze(0).unsqueeze(0)
    
    def forward(self, mask):
        kernel = self.kernel.to(mask.device)
        edge = F.conv2d(mask, kernel, padding=1)
        edge = torch.abs(edge)
        edge = (edge > 0).float()   # (edge > 0)是一个bool张量
        return edge


class EdgeLoss(nn.Module):
    """边缘损失 - 使用拉普拉斯算子计算边缘"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.edge_extractor = EdgeExtractor()
    
    def forward(self, pred, target):
        # 输入的pred是sigmoid之后的概率，pred_binary转化为二值掩码，再计算边缘，这样是否正确？
        # pred_binary = (pred > 0.5).float()
        pred_edge = self.edge_extractor(pred)
        target_edge = self.edge_extractor(target)
        return F.mse_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    """组合损失：Dice + BCE + Edge"""
    def __init__(self, dice_weight=1.0, bce_weight=1.0, edge_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.edge_loss = EdgeLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        pred_for_loss = pred
        if pred.shape[1] > 1:
            pred_for_loss = pred[:, 0:1, :, :]
        
        dice = self.dice_loss(pred_for_loss, target)
        bce = self.bce_loss(pred_for_loss, target)
        edge = self.edge_loss(torch.sigmoid(pred_for_loss), target) 
        # 这里是传入了sigmoid之后的连续型概率张量，而LDB-Net中用的是pred_mask_binary = (pred_mask_sigmoid > 0.5).float()，也就是转成了0-1硬张量来处理，这是否会造成不一样的结果？
        total = self.dice_weight * dice + self.bce_weight * bce + self.edge_weight * edge
        return total, dice.item(), bce.item(), edge.item()


def compute_f1_iou(pred, target, threshold=0.5):
    """计算F1和IoU"""
    pred = (pred >= threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    iou = tp / (tp + fp + fn + 1e-7)
    
    return f1.item(), iou.item()


def train():
    """训练函数"""
    parser = argparse.ArgumentParser(description='Train Forgery UniFormer')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--freeze_resnet', action='store_true', default=False,
                        help='Whether to freeze ResNet and SACC')
    parser.add_argument('--freeze_uniformer', action='store_true', default=False,
                        help='Whether to freeze UniFormer')
    parser.add_argument('--checkpoint_dir', type=str, default='autodl-tmp/fassa_checkpoint',
                        help='Directory to save checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = 'autodl-tmp/data'
    tampered_dir = os.path.join(data_dir, 'tampered')
    mask_dir = os.path.join(data_dir, 'masks')
    tensor_dir = 'autodl-tmp/data_tensor'
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    train_ratio = 0.9
    start_epoch = 0
    
    print("Loading dataset...")
    full_dataset = TamperingDataset(
        tampered_dir=tampered_dir,
        mask_dir=mask_dir,
        tensor_dir=tensor_dir
    )
    
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(dataset_size, generator=generator)
    
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.tolist())
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices.tolist())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {dataset_size}, Train size: {train_size}, Test size: {test_size}")
    
    model = ForgeryUniformerSegmentation(
        img_size=512,
        num_classes=2,
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        resnet_pretrained=True
    ).to(device)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from checkpoint: {args.resume}")
        print(f"Starting from epoch: {start_epoch + 1}")
    
    if args.freeze_resnet:
        print("Freezing ResNet branch and SACC module...")
        for param in model.encoder.resnet_branch.parameters():
            param.requires_grad = False
    
    if args.freeze_uniformer:
        print("Freezing UniFormer encoder...")
        for param in model.encoder.parameters():
            if 'resnet_branch' not in str(param):
                param.requires_grad = False
    
    criterion = CombinedLoss(dice_weight=0.8, bce_weight=0.8, edge_weight=1.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_f1 = 0.0
        epoch_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            features = batch['feature'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, features)
            loss, dice_val, bce_val, edge_val = criterion(outputs, masks)
            
            loss.backward()

            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: Gradient is None for parameter {name}")
                    
            exit(1)
            
            optimizer.step()
            
            outputs_for_metric = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
            f1, iou = compute_f1_iou(torch.sigmoid(outputs_for_metric), masks)
            
            epoch_loss += loss.item()
            epoch_dice_loss += dice_val
            epoch_bce_loss += bce_val
            epoch_edge_loss += edge_val
            epoch_f1 += f1
            epoch_iou += iou
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_val:.4f}',
                'bce': f'{bce_val:.4f}',
                'edge': f'{edge_val:.4f}',
                'f1': f'{f1:.4f}',
                'iou': f'{iou:.4f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_dice_loss = epoch_dice_loss / num_batches
        avg_bce_loss = epoch_bce_loss / num_batches
        avg_edge_loss = epoch_edge_loss / num_batches
        avg_f1 = epoch_f1 / num_batches
        avg_iou = epoch_iou / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        print(f"  Dice Loss: {avg_dice_loss:.4f}, BCE Loss: {avg_bce_loss:.4f}, Edge Loss: {avg_edge_loss:.4f}")
        print(f"  Train F1: {avg_f1:.4f}, Train IoU: {avg_iou:.4f}")
        
        model.eval()
        test_loss = 0.0
        test_dice_loss = 0.0
        test_bce_loss = 0.0
        test_edge_loss = 0.0
        test_f1 = 0.0
        test_iou = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                features = batch['feature'].to(device)
                
                outputs = model(images, features)
                
                loss, dice_val, bce_val, edge_val = criterion(outputs, masks)
                outputs_for_metric = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
                f1, iou = compute_f1_iou(torch.sigmoid(outputs_for_metric), masks)
                
                test_loss += loss.item()
                test_dice_loss += dice_val
                test_bce_loss += bce_val
                test_edge_loss += edge_val
                test_f1 += f1
                test_iou += iou
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        avg_test_dice_loss = test_dice_loss / test_batches
        avg_test_bce_loss = test_bce_loss / test_batches
        avg_test_edge_loss = test_edge_loss / test_batches
        avg_test_f1 = test_f1 / test_batches
        avg_test_iou = test_iou / test_batches
        
        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {avg_test_loss:.4f}")
        print(f"  Dice Loss: {avg_test_dice_loss:.4f}, BCE Loss: {avg_test_bce_loss:.4f}, Edge Loss: {avg_test_edge_loss:.4f}")
        print(f"  Test F1: {avg_test_f1:.4f}, Test IoU: {avg_test_iou:.4f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'f1': avg_f1,
            'iou': avg_iou
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Training completed!")


if __name__ == '__main__':
    train()
