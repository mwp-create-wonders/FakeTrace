import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import argparse

from forgery_uniformer import ForgeryUniformerSegmentation


class CocoTestDataset(Dataset):
    def __init__(self, tampered_dir, mask_dir, tensor_dir):
        self.tampered_dir = tampered_dir
        self.mask_dir = mask_dir
        self.tensor_dir = tensor_dir
        
        self.tampered_files = sorted([f for f in os.listdir(tampered_dir) if f.endswith('.jpg')])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.tampered_files)
    
    def __getitem__(self, idx):
        tampered_name = self.tampered_files[idx]
        base_name = tampered_name.replace('.jpg', '')
        
        tampered_path = os.path.join(self.tampered_dir, tampered_name)
        mask_path = os.path.join(self.mask_dir, f'{base_name}_mask.jpg')
        tensor_path = os.path.join(self.tensor_dir, f'{base_name}.npz')
        
        tampered_img = Image.open(tampered_path).convert('RGB')
        tampered_img = self.transform(tampered_img)
        
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor = transforms.ToTensor()(mask_img)
        mask_tensor = (mask_tensor > 0.3).float()
        
        tensor_data = np.load(tensor_path)
        feature_tensor = torch.from_numpy(tensor_data['tensor']).permute(2, 0, 1).float()
        
        return {
            'image': tampered_img,
            'mask': mask_tensor,
            'feature': feature_tensor,
            'name': base_name
        }


def compute_f1_score(pred, target, threshold=0.3):
    pred_bin = (pred >= threshold).float()
    pred_flat = pred_bin.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    tp = (pred_flat * target_flat).sum()
    fp = ((1 - target_flat) * pred_flat).sum()
    fn = (target_flat * (1 - pred_flat)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1


def main():
    parser = argparse.ArgumentParser(description='Test on COCO dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_23.pth',
                        help='Path to the best checkpoint')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Whether to save predictions')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    tampered_dir = 'autodl-tmp/CoMoFod/test/tampered'
    mask_dir = 'autodl-tmp/CoMoFod/test/masks'
    tensor_dir = 'autodl-tmp/CoMoFod_tensor'
    
    dataset = CocoTestDataset(tampered_dir, mask_dir, tensor_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'Test dataset size: {len(dataset)}')
    
    model = ForgeryUniformerSegmentation(
        img_size=512,
        num_classes=2,
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        resnet_pretrained=False
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint: {args.checkpoint}')
    
    if args.save_predictions:
        os.makedirs('predictions', exist_ok=True)
    
    model.eval()
    
    all_preds = []
    all_targets = []
    f1_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            features = batch['feature'].to(device)
            names = batch['name']
            
            outputs = model(images, features)
            outputs = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
            preds = torch.sigmoid(outputs)
            
            f1 = compute_f1_score(preds, masks)
            f1_scores.append(f1)
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(masks.view(-1).cpu().numpy())
            
            if args.save_predictions:
                for j, name in enumerate(names):
                    pred_np = preds[j, 0].cpu().numpy()
                    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                    pred_img.save(f'predictions/{name}_pred.png')
            
            progress = (i + 1) / len(dataloader) * 100
            print(f'\rProgress: [{int(progress / 5) * "="}{(20 - int(progress / 5)) * " "}] {progress:.1f}%', end='')
        
        print()
    
    avg_f1 = np.mean(f1_scores)
    auc = roc_auc_score(all_targets, all_preds)
    
    print(f'\nTest Results:')
    print(f'F1 Score: {avg_f1:.4f}')
    print(f'AUC: {auc:.4f}')
    
    if args.save_predictions:
        print(f'\nPredictions saved to predictions/')


if __name__ == '__main__':
    main()