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

from resnet_branch import ResNetBranch

class ResNetBranchWithHead(nn.Module):
    """带分割头的ResNetBranch，用于预训练"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet_branch = ResNetBranch(pretrained=pretrained)
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 1, 1)
        )
    
    def forward(self, x):
        _, _, _, out_4 = self.resnet_branch(x)
        logits = self.segmentation_head(out_4)
        return logits

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
        
        mask_path = os.path.join(self.mask_dir, f"{file_name}_mask.png")
        tensor_path = os.path.join(self.tensor_dir, f"{file_name}.npz")
        
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor = transforms.ToTensor()(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()
        
        tensor_data = np.load(tensor_path)
        feature_tensor = torch.from_numpy(tensor_data['tensor']).permute(2, 0, 1).float()
        
        return feature_tensor, mask_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

def compute_f1_iou(pred, target):
    pred = (torch.sigmoid(pred) >= 0.5).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return f1.item(), iou.item()

def main():
    parser = argparse.ArgumentParser(description='Pretrain ResNetBranch')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='RB_ckpt')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = 'autodl-tmp/IMD2020'
    mask_dir = os.path.join(data_dir, 'masks')
    tensor_dir = 'autodl-tmp/IMD2020_tensor'
    
    dataset = TamperingDataset(
        tampered_dir=os.path.join(data_dir, 'tampered'),
        mask_dir=mask_dir,
        tensor_dir=tensor_dir
    )
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = ResNetBranchWithHead(pretrained=True).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    print(f"Training ResNetBranch with {len(dataset)} samples (all used for training)")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_f1 = 0.0
        epoch_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for features, masks in pbar:
            features = features.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            bce_loss = criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce_loss + dice
            
            loss.backward()
            optimizer.step()
            
            f1, iou = compute_f1_iou(outputs, masks)
            
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_f1 += f1
            epoch_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}',
                'f1': f'{f1:.4f}',
                'iou': f'{iou:.4f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_f1 = epoch_f1 / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_loss:.4f}")
        print(f"  Dice Loss: {avg_dice:.4f}, Train F1: {avg_f1:.4f}, Train IoU: {avg_iou:.4f}")
        
        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'resnet_branch_whole_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'f1': avg_f1,
                'iou': avg_iou
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            branch_path = os.path.join(args.output_dir, f'RB_no_head_epoch_{epoch+1}.pth')
            torch.save(model.resnet_branch.state_dict(), branch_path)
            print(f"ResNetBranch weights saved: {branch_path}")
    
    print("Pretraining completed!")

if __name__ == '__main__':
    main()