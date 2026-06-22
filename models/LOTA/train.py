import os
import torch
from datetime import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import util as toolkit
from loader import get_loader as fetch_train_data, get_val_loader as fetch_val_data
from config import ConfigurationManager as Configurator
from model import model as NeuralNetwork
from util import bceLoss as compute_binary_loss


def execute_training_iteration(data_provider, network, optimizer, epoch_index):
    """Perform one training epoch"""
    network.train()
    global iteration_counter, total_batches, config

    epoch_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(data_provider, start=1):
        optimizer.zero_grad()

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()

        outputs = network(inputs).ravel()

        loss_function = compute_binary_loss()
        batch_loss = loss_function(outputs, targets)

        batch_loss.backward()
        optimizer.step()

        iteration_counter += 1
        epoch_loss += batch_loss.item()

        if batch_idx % 100 == 0 or batch_idx == 1 or batch_idx == total_batches:
            progress_percent = (batch_idx / total_batches) * 100
            print(
                f"📊 Epoch: {epoch_index:02d}/{config.epoch:02d} | "
                f"Iter: {batch_idx:04d}/{total_batches:04d} "
                f"({progress_percent:.1f}%) | "
                f"Loss: {batch_loss.item():.6f}"
            )

    avg_loss = epoch_loss / total_batches
    print(f"✅ Epoch {epoch_index:02d} finished | Avg Loss: {avg_loss:.6f}")

    return avg_loss


def perform_validation(validation_loader, network, epoch_index, storage_location):
    """Evaluate model on unified validation loader"""
    network.eval()
    global best_performing_epoch, highest_accuracy

    total_correct = 0
    total_samples = 0

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).float()

            logits = network(inputs).ravel()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct = (preds == targets).float()

            total_correct += correct.sum().item()
            total_samples += targets.numel()

            # 真实=1，生成=0
            real_mask = (targets == 1)
            fake_mask = (targets == 0)

            if real_mask.any():
                real_correct += (preds[real_mask] == targets[real_mask]).float().sum().item()
                real_total += real_mask.sum().item()

            if fake_mask.any():
                fake_correct += (preds[fake_mask] == targets[fake_mask]).float().sum().item()
                fake_total += fake_mask.sum().item()

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    real_accuracy = real_correct / real_total if real_total > 0 else 0.0
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0.0

    # 保存最佳模型
    if epoch_index == 1 or overall_accuracy > highest_accuracy:
        best_performing_epoch = epoch_index
        highest_accuracy = overall_accuracy
        best_model_path = os.path.join(storage_location, 'Network_best.pth')
        torch.save(network.state_dict(), best_model_path)
        print(f"💾 Saved best model at epoch {epoch_index}")

    print(
        f"🧪 Validation | Epoch {epoch_index:03d} | "
        f"Overall Acc: {overall_accuracy:.2%} | "
        f"Real Acc: {real_accuracy:.2%} | "
        f"Fake Acc: {fake_accuracy:.2%} | "
        f"Best Epoch: {best_performing_epoch:03d} | "
        f"Best Acc: {highest_accuracy:.2%}"
    )

    return overall_accuracy


def configure_gpu(gpu_id):
    """Set GPU configuration"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def main_execution():
    """Main training procedure"""
    global config, total_batches, iteration_counter, best_performing_epoch, highest_accuracy

    torch.set_num_threads(2)
    toolkit.set_random_seed()

    # 先解析配置
    config = Configurator().parse()

    # 再配置GPU
    configure_gpu(config.gpu_id)

    # 数据
    train_loader = fetch_train_data(config)
    total_batches = len(train_loader)

    val_loader = None
    has_validation = (
        hasattr(config, "val_real_dirs") and hasattr(config, "val_fake_dirs")
        and len(config.val_real_dirs) > 0
        and len(config.val_fake_dirs) > 0
    )
    if has_validation:
        val_loader, _ = fetch_val_data(config)

    # 模型
    model = NeuralNetwork().cuda()

    if config.load:
        model.load_state_dict(torch.load(config.load))
        print(f"Loaded model from {config.load}")

    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    output_dir = config.save_path
    os.makedirs(output_dir, exist_ok=True)

    iteration_counter = 0
    best_performing_epoch = 0
    highest_accuracy = 0.0

    print("||Training||")

    for epoch in range(1, config.epoch + 1):
        current_lr = toolkit.poly_lr(optimizer, config.lr, epoch, config.epoch)
        print(f"\n🚀 Epoch {epoch:02d}/{config.epoch:02d} | LR: {current_lr:.8f}")

        avg_loss = execute_training_iteration(
            train_loader, model, optimizer, epoch
        )

        # 每轮保存一个latest
        latest_path = os.path.join(output_dir, 'Network_latest.pth')
        torch.save(model.state_dict(), latest_path)

        # 可选：每10轮额外存档
        if epoch % 10 == 0:
            epoch_ckpt = os.path.join(output_dir, f'Network_epoch_{epoch}.pth')
            torch.save(model.state_dict(), epoch_ckpt)

        if val_loader is not None:
            perform_validation(val_loader, model, epoch, output_dir)

    print("✅ Training finished.")


if __name__ == '__main__':
    main_execution()