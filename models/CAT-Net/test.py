import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Splicing.data.dataset_user import UserDataset
from lib.models.network_CAT import CAT_Net
from lib.utils.utils import FullModel
from lib.config import config
cfg = config


def parse_args():
    parser = argparse.ArgumentParser(description='Test CAT-Net on user dataset')
    parser.add_argument('--config', type=str, default='experiments/CAT_full.yaml',
                        help='path to config file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='path to user dataset root')
    parser.add_argument('--checkpoint', type=str, default='output/splicing_dataset/CAT_full/best.pth.tar',
                        help='path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    return parser.parse_args()


def test(config, testloader, model):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            qtable = qtable.cuda()
            
            # Forward pass
            output = model(image, qtable)
            output = F.upsample(input=output, size=(size[-2], size[-1]), mode='bilinear')
            
            # Get probabilities and predictions
            prob = F.softmax(output, dim=1)[:, 1, :, :]  # Probability of tampered class
            pred = torch.argmax(output, dim=1)
            
            # Flatten and collect
            prob_flat = prob.cpu().numpy().flatten()
            pred_flat = pred.cpu().numpy().flatten()
            label_flat = label.cpu().numpy().flatten()
            
            all_preds.append(prob_flat)
            all_labels.append(label_flat)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    
    return f1, auc


def main():
    args = parse_args()
    
    # Load config
    cfg.merge_from_file(args.config)
    cfg.freeze()
    
    # Create dataset and dataloader
    test_dataset = UserDataset(
        crop_size=None,  # No cropping for testing
        grid_crop=True,  # Need grid_crop for DCT coefficients
        blocks=['RGB', 'DCTcoef', 'qtable'],  # Include RGB, DCTcoef and qtable for model input
        DCT_channels=21,  # Number of DCT channels
        data_root=args.data_root,
        mode='test',
        read_from_jpeg=False
    )
    
    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = CAT_Net(cfg)
    model = model.cuda()
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully")
    else:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    # Test
    print("Testing on user dataset...")
    f1, auc = test(cfg, testloader, model)
    
    # Output results
    print(f"Pixel-level F1 score: {f1:.4f}")
    print(f"Pixel-level AUC score: {auc:.4f}")


if __name__ == '__main__':
    main()
