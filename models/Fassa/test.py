import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import argparse
import io

from forgery_uniformer import ForgeryUniformerSegmentation


class CocoTestDataset(Dataset):
    def __init__(self, tampered_dir, mask_dir, tensor_dir, jpeg_quality=None, gaussian_noise=None, save_check=False):
        self.tampered_dir = tampered_dir
        self.mask_dir = mask_dir
        self.tensor_dir = tensor_dir
        self.jpeg_quality = jpeg_quality
        self.gaussian_noise = gaussian_noise
        self.save_check = save_check
        self.saved = False
        
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
        
        # Save original image for checking
        if self.save_check and not self.saved:
            os.makedirs('check', exist_ok=True)
            tampered_img.save(f'check/original_{base_name}.png')
        
        # Apply JPEG compression if specified
        if self.jpeg_quality is not None:
            buffer = io.BytesIO()
            tampered_img.save(buffer, format='JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            tampered_img = Image.open(buffer).convert('RGB')
        
        # Apply Gaussian noise if specified
        if self.gaussian_noise is not None:
            img_array = np.array(tampered_img).astype(np.float32)
            noise = np.random.normal(0, self.gaussian_noise, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            tampered_img = Image.fromarray(img_array)
            
            # Save noisy image for checking
            if self.save_check and not self.saved:
                Image.fromarray(img_array).save(f'check/noisy_{base_name}.png')
                self.saved = True
        
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


def compute_iou(pred, target, threshold=0.3):
    pred_bin = (pred >= threshold).float()
    pred_flat = pred_bin.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    return iou


def main():
    parser = argparse.ArgumentParser(description='Test on COCO dataset')
    parser.add_argument('--checkpoint', type=str, default='autodl-tmp/coco_finetune_checkpoint/model_epoch_15.pth',
                        help='Path to the best checkpoint')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Whether to save predictions')
    parser.add_argument('--jpeg_quality', type=int, default=None,
                        help='JPEG compression quality (e.g., 95, 80, 50)')
    parser.add_argument('--gaussian_noise', type=float, default=None,
                        help='Gaussian noise standard deviation (e.g., 7, 15, 23)')
    parser.add_argument('--save_check', action='store_true', default=False,
                        help='Save original and processed images for checking')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.jpeg_quality is not None:
        print(f'Applying JPEG compression with quality: {args.jpeg_quality}')
    if args.gaussian_noise is not None:
        print(f'Applying Gaussian noise with std: {args.gaussian_noise}')
    
    tampered_dir = 'autodl-tmp/IMD2020/tampered'
    mask_dir = 'autodl-tmp/IMD2020/masks'
    tensor_dir = 'autodl-tmp/IMD2020_tensor'
    
    dataset = CocoTestDataset(
        tampered_dir, mask_dir, tensor_dir,
        jpeg_quality=args.jpeg_quality,
        gaussian_noise=args.gaussian_noise,
        save_check=args.save_check
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # print(f'Test dataset size: 102')
    
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
        os.makedirs('IMD2020_predictions', exist_ok=True)
    
    model.eval()
    
    all_preds = []
    all_targets = []
    f1_scores = []
    iou_scores = []
    iou_per_image = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            features = batch['feature'].to(device)
            names = batch['name']
            
            outputs, _ = model(images, features)  # 解包，忽略 anomaly_maps
            outputs = outputs[:, 0:1, :, :] if outputs.shape[1] > 1 else outputs
            preds = torch.sigmoid(outputs)
            
            f1 = compute_f1_score(preds, masks)
            f1_scores.append(f1)
            
            iou = compute_iou(preds, masks)
            iou_scores.append(iou)

            for j, name in enumerate(names):
                single_iou = compute_iou(preds[j:j+1], masks[j:j+1])
                iou_per_image.append((name, single_iou))
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(masks.view(-1).cpu().numpy())
            
            if args.save_predictions:
                for j, name in enumerate(names):
                    pred_np = preds[j, 0].cpu().numpy()
                    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
                    pred_img.save(f'IMD2020_predictions/{name}_pred.png')
            
            progress = (i + 1) / len(dataloader) * 100
            print(f'\rProgress: [{int(progress / 5) * "="}{(20 - int(progress / 5)) * " "}] {progress:.1f}%', end='')
        
        print()
    
    avg_f1 = np.mean(f1_scores)
    avg_iou = np.mean(iou_scores)
    auc = roc_auc_score(all_targets, all_preds)

    iou_per_image.sort(key=lambda x: x[1], reverse=True)
    top2_iou = iou_per_image[:2]
    
    print(f'\nTest Results:')
    print(f'F1 Score: {avg_f1:.4f}')
    print(f'IoU: {avg_iou:.4f}')
    print(f'AUC: {auc:.4f}')

    print(f'\nIoU最大的两张图片:')
    for idx, (name, iou_val) in enumerate(top2_iou, 1):
        print(f'  第{idx}名: {name}, IoU: {iou_val:.4f}')
    
    if args.save_predictions:
        print(f'\nDone!')


if __name__ == '__main__':
    main()