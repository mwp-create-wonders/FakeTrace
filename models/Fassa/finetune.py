import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
import shutil

from forgery_uniformer import ForgeryUniformerSegmentation

class TamperingDataset(Dataset):
    def __init__(self, tampered_dir, mask_dir, tensor_dir, transform=None):
        self.tampered_dir = tampered_dir
        self.mask_dir = mask_dir
        self.tensor_dir = tensor_dir
        self.transform = transform
        
        self.image_names = []
        for fname in os.listdir(tampered_dir):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                base_name = fname[:-4]
                mask_name = f'{base_name}_mask.png' if fname.endswith('.png') else f'{base_name}_mask.jpg'
                tensor_name = f'{base_name}.npz'
                if os.path.exists(os.path.join(mask_dir, mask_name)) and os.path.exists(os.path.join(tensor_dir, tensor_name)):
                    self.image_names.append((fname, mask_name, tensor_name))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name, mask_name, tensor_name = self.image_names[idx]
        
        tampered_path = os.path.join(self.tampered_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        tensor_path = os.path.join(self.tensor_dir, tensor_name)
        
        tampered_img = Image.open(tampered_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        
        if self.transform:
            tampered_img = self.transform(tampered_img)
        
        mask_tensor = transforms.ToTensor()(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()
        
        tensor_data = np.load(tensor_path)
        feature_tensor = torch.from_numpy(tensor_data['tensor']).permute(2, 0, 1).float()
        
        return tampered_img, feature_tensor, mask_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class EdgeExtractor(nn.Module):
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
        edge = (edge > 0).float()
        return edge

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.edge_extractor = EdgeExtractor()
    
    def forward(self, pred, target):
        pred_edge = self.edge_extractor(pred)
        target_edge = self.edge_extractor(target)
        return F.mse_loss(pred_edge, target_edge)

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, edge_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.edge_loss = EdgeLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        pred_for_loss = pred[:, 0:1, :, :] if pred.shape[1] > 1 else pred
        
        dice = self.dice_loss(torch.sigmoid(pred_for_loss), target)
        bce = self.bce_loss(pred_for_loss, target)
        edge = self.edge_loss(torch.sigmoid(pred_for_loss), target)
        
        total = self.dice_weight * dice + self.bce_weight * bce + self.edge_weight * edge
        return total, dice.item(), bce.item(), edge.item()

def compute_f1_iou(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum().item()
    union = pred_bin.sum().item() + target.sum().item() - intersection
    tp = intersection
    fp = pred_bin.sum().item() - intersection
    fn = target.sum().item() - intersection
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union
    
    return f1, iou

def main():
    parser = argparse.ArgumentParser(description='Finetune ForgeryUniFormer on joint dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_22.pth', help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/finetune_checkpoint', help='Output directory for checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TamperingDataset(
        tampered_dir='/root/autodl-tmp/joint/train/tampered',
        mask_dir='/root/autodl-tmp/joint/train/masks',
        tensor_dir='/root/autodl-tmp/joint/train/train_tensor',
        transform=transform
    )
    
    test_dataset = TamperingDataset(
        tampered_dir='/root/autodl-tmp/joint/test/tampered',
        mask_dir='/root/autodl-tmp/joint/test/masks',
        tensor_dir='/root/autodl-tmp/joint/test/test_tensor',
        transform=transform
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = ForgeryUniformerSegmentation(img_size=512, num_classes=2).to(device)
    
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Freezing ResNet branch...')
    for param in model.encoder.resnet_branch.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = CombinedLoss(dice_weight=1.0, bce_weight=1.0, edge_weight=0.1)
    
    print('Training started!')
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_bce = 0.0
        train_edge = 0.0
        train_f1 = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for images, features, masks in pbar:
            images = images.to(device)
            features = features.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            
            loss, dice_val, bce_val, edge_val = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_val
            train_bce += bce_val
            train_edge += edge_val
            
            outputs_for_metric = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
            f1, iou = compute_f1_iou(torch.sigmoid(outputs_for_metric), masks)
            train_f1 += f1
            train_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_val:.4f}',
                'bce': f'{bce_val:.4f}',
                'edge': f'{edge_val:.4f}',
                'f1': f'{f1:.4f}',
                'iou': f'{iou:.4f}'
            })
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_bce /= len(train_loader)
        train_edge /= len(train_loader)
        train_f1 /= len(train_loader)
        train_iou /= len(train_loader)
        
        print(f'\nEpoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}')
        print(f'  Dice Loss: {train_dice:.4f}, BCE Loss: {train_bce:.4f}, Edge Loss: {train_edge:.4f}')
        print(f'  Train F1: {train_f1:.4f}, Train IoU: {train_iou:.4f}')
        
        model.eval()
        test_loss = 0.0
        test_dice = 0.0
        test_bce = 0.0
        test_edge = 0.0
        test_f1 = 0.0
        test_iou = 0.0
        
        with torch.no_grad():
            for images, features, masks in test_loader:
                images = images.to(device)
                features = features.to(device)
                masks = masks.to(device)
                
                outputs = model(images, features)
                loss, dice_val, bce_val, edge_val = criterion(outputs, masks)
                
                test_loss += loss.item()
                test_dice += dice_val
                test_bce += bce_val
                test_edge += edge_val
                
                outputs_for_metric = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
                f1, iou = compute_f1_iou(torch.sigmoid(outputs_for_metric), masks)
                test_f1 += f1
                test_iou += iou
        
        test_loss /= len(test_loader)
        test_dice /= len(test_loader)
        test_bce /= len(test_loader)
        test_edge /= len(test_loader)
        test_f1 /= len(test_loader)
        test_iou /= len(test_loader)
        
        print(f'Epoch {epoch+1}/{args.num_epochs} - Test Loss: {test_loss:.4f}')
        print(f'  Dice Loss: {test_dice:.4f}, BCE Loss: {test_bce:.4f}, Edge Loss: {test_edge:.4f}')
        print(f'  Test F1: {test_f1:.4f}, Test IoU: {test_iou:.4f}')
        
        checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_iou': train_iou,
            'test_iou': test_iou
        }, checkpoint_path)
        
        print(f'Checkpoint saved: {checkpoint_path}\n')
        
        scheduler.step()
    
    print('Finetuning completed!')

if __name__ == '__main__':
    main()