import os
import time
import random
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

from data.datasets import (
    build_six_folder_dataset,
    SyncPairAugmenter,
    six_folder_collate_fn,
)
from networks.trainer import Trainer
from options import TrainOptions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def build_transforms(opt):
    """
    构建普通 transform 和同步增强器
    """
    sync_transform = None
    if opt.use_sync_aug:
        crop_size = (opt.crop_size, opt.crop_size) if opt.use_random_crop else None

        sync_transform = SyncPairAugmenter(
            resize=None,  # 外面普通 transform 里统一 resize
            hflip_prob=opt.sync_hflip_prob,
            crop_size=crop_size,
            brightness=opt.sync_brightness,
            contrast=opt.sync_contrast,
            saturation=opt.sync_saturation,
            gamma_range=(opt.sync_gamma_min, opt.sync_gamma_max) if opt.use_sync_gamma else None,
            blur_prob=opt.sync_blur_prob,
            blur_radius_range=(opt.sync_blur_min, opt.sync_blur_max),
        )

    transform = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=opt.norm_mean,
            std=opt.norm_std,
        ),
    ])

    return transform, sync_transform


def create_dataloader(opt):
    transform, sync_transform = build_transforms(opt)

    dataset = build_six_folder_dataset(
        real_dir=opt.real_dir,
        real_processed_dir=opt.real_processed_dir,
        dm_dir=opt.dm_dir,
        dm_processed_dir=opt.dm_processed_dir,
        ar_dir=opt.ar_dir,
        ar_processed_dir=opt.ar_processed_dir,
        transform=transform,
        sync_transform=sync_transform,
        return_pil=False,
        strict_check=opt.strict_check,
        limit=opt.dataset_limit,
        shuffle_filenames=False,
        seed=opt.seed,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_threads,
        pin_memory=opt.pin_memory,
        drop_last=opt.drop_last,
        collate_fn=six_folder_collate_fn,
    )
    return data_loader


def print_loss_message(step, epoch, batch_idx, num_batches, losses, elapsed, lr):
    msg = (
        f"[Epoch {epoch:03d}] "
        f"[Batch {batch_idx:05d}/{num_batches:05d}] "
        f"[Step {step:07d}] "
        f"lr={lr:.6e} | "
        f"total={losses.get('loss_total', 0.0):.6f} | "
        f"bin={losses.get('loss_bin', 0.0):.6f}(w={losses.get('weight_loss_bin', 0.0):.3f}) | "
        f"src={losses.get('loss_src', 0.0):.6f}(w={losses.get('weight_loss_src', 0.0):.3f}) | "
        f"pair={losses.get('loss_pair', 0.0):.6f}(w={losses.get('weight_loss_pair', 0.0):.3f}) | "
        f"rf={losses.get('loss_rf_sep', 0.0):.6f}(w={losses.get('weight_loss_rf_sep', 0.0):.3f}) | "
        f"dmar={losses.get('loss_dm_ar', 0.0):.6f}(w={losses.get('weight_loss_dm_ar', 0.0):.3f}) | "
        f"con_bin={losses.get('loss_con_bin', 0.0):.6f}(w={losses.get('weight_loss_con_bin', 0.0):.3f}) | "
        f"con_src={losses.get('loss_con_src', 0.0):.6f}(w={losses.get('weight_loss_con_src', 0.0):.3f}) | "
        f"time/step={elapsed:.4f}s"
    )
    print(msg)


def save_checkpoint(model, save_path, epoch=None, extra_state=None):
    """
    保存完整 checkpoint，便于恢复训练
    """
    ensure_dir(os.path.dirname(save_path))

    state = {
        "model": model.model.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "scheduler": model.scheduler.state_dict() if hasattr(model, "scheduler") and model.scheduler is not None else None,
        "total_steps": getattr(model, "total_steps", 0),
    }

    if epoch is not None:
        state["epoch"] = epoch

    if extra_state is not None:
        state.update(extra_state)

    torch.save(state, save_path)


def print_train_config(opt):
    print("========== Loss Configuration ==========")
    print(f"use_loss_bin      : {opt.use_loss_bin} | lambda_bin       = {opt.lambda_bin}")
    print(f"use_loss_src      : {opt.use_loss_src} | lambda_src       = {opt.lambda_src}")
    print(f"use_loss_pair     : {opt.use_loss_pair} | lambda_pair      = {opt.lambda_pair}")
    print(f"use_loss_rf       : {opt.use_loss_rf} | lambda_rf         = {opt.lambda_rf}")
    print(f"use_loss_dmar     : {opt.use_loss_dmar} | lambda_dmar       = {opt.lambda_dmar}")
    print(f"use_loss_con_bin  : {opt.use_loss_con_bin} | lambda_con_bin   = {opt.lambda_con_bin}")
    print(f"use_loss_con_src  : {opt.use_loss_con_src} | lambda_con_src   = {opt.lambda_con_src}")
    print("----------------------------------------")
    print(f"dmar_start_epoch   : {getattr(opt, 'dmar_start_epoch', 1)}")
    print(f"dmar_warmup_epochs : {getattr(opt, 'dmar_warmup_epochs', 2)}")
    print(f"contrast_start_epoch : {getattr(opt, 'contrast_start_epoch', 0)}")
    print(f"contrast_full_epoch  : {getattr(opt, 'contrast_full_epoch', 1)}")
    print(f"con_bin_decay_epoch  : {getattr(opt, 'con_bin_decay_epoch', 2)}")
    print(f"con_bin_decay_ratio  : {getattr(opt, 'con_bin_decay_ratio', 0.3)}")
    print(f"con_src_off_epoch    : {getattr(opt, 'con_src_off_epoch', 2)}")
    print("----------------------------------------")
    print(f"margin_real_fake   : {getattr(opt, 'margin_real_fake', 0.30)}")
    print(f"margin_dm_ar       : {getattr(opt, 'margin_dm_ar', 0.15)}")
    print(f"temperature        : {getattr(opt, 'temperature', 0.10)}")
    print("========================================")


if __name__ == "__main__":
    opt = TrainOptions().parse()
    set_seed(opt.seed)

    model = Trainer(opt)
    data_loader = create_dataloader(opt)

    print(f"Length of data loader: {len(data_loader)}")
    print_train_config(opt)

    save_dir = getattr(opt, "checkpoints_dir", ".")
    ensure_dir(save_dir)

    start_epoch = 0
    best_train_loss = float("inf")

    # 可选恢复
    if opt.resume_path is not None and len(opt.resume_path) > 0:
        print(f"Resuming from checkpoint: {opt.resume_path}")
        ckpt = torch.load(opt.resume_path, map_location=model.device)

        model.model.load_state_dict(ckpt["model"], strict=opt.resume_strict)

        if opt.resume_optimizer and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            model.optimizer.load_state_dict(ckpt["optimizer"])

        if opt.resume_scheduler and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            model.scheduler.load_state_dict(ckpt["scheduler"])

        if "total_steps" in ckpt:
            model.total_steps = ckpt["total_steps"]

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        if "best_train_loss" in ckpt:
            best_train_loss = float(ckpt["best_train_loss"])

        print(f"Resume start epoch: {start_epoch}")
        print(f"Resume total steps: {model.total_steps}")
        print(f"Resume best train loss: {best_train_loss:.6f}")

    start_time = time.time()
    global_step_start = model.total_steps

    for epoch in range(start_epoch, opt.niter):
        model.set_epoch(epoch)
        model.set_model_train()

        epoch_start_time = time.time()
        epoch_loss_meter = {}
        epoch_loss_count = 0

        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"Epoch {epoch:03d}",
            ncols=140,
            leave=True,
        )

        for i, data in progress_bar:
            step_start = time.time()

            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            epoch_loss_count += 1

            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    epoch_loss_meter[k] = epoch_loss_meter.get(k, 0.0) + float(v)

            current_lr = model.optimizer.param_groups[0]["lr"]

            progress_bar.set_postfix({
                "lr": f"{current_lr:.2e}",
                "total": f"{losses.get('loss_total', 0.0):.4f}",
                "bin": f"{losses.get('loss_bin', 0.0):.4f}",
                "src": f"{losses.get('loss_src', 0.0):.4f}",
                "conb": f"{losses.get('loss_con_bin', 0.0):.4f}",
                "cons": f"{losses.get('loss_con_src', 0.0):.4f}",
            })

            if (model.total_steps % opt.print_freq == 0) or (i == len(data_loader) - 1):
                elapsed = time.time() - step_start
                print_loss_message(
                    step=model.total_steps,
                    epoch=epoch,
                    batch_idx=i + 1,
                    num_batches=len(data_loader),
                    losses=losses,
                    elapsed=elapsed,
                    lr=current_lr,
                )

            if opt.save_latest_freq > 0 and model.total_steps % opt.save_latest_freq == 0:
                latest_path = os.path.join(save_dir, f"model_iters_{model.total_steps}.pth")
                print(f"Saving latest checkpoint at step {model.total_steps} -> {latest_path}")
                save_checkpoint(
                    model,
                    latest_path,
                    epoch=epoch,
                    extra_state={"best_train_loss": best_train_loss},
                )

        # 处理最后不足 accumulation_steps 的梯度
        model.finalize_epoch()

        epoch_elapsed = time.time() - epoch_start_time

        # 计算 epoch 平均 loss
        epoch_avg_losses = {}
        if epoch_loss_count > 0:
            for k, v in epoch_loss_meter.items():
                epoch_avg_losses[k] = v / epoch_loss_count

        avg_total_loss = epoch_avg_losses.get("loss_total", float("inf"))

        print(f"End of epoch {epoch} | epoch time: {epoch_elapsed:.2f}s")
        if len(epoch_avg_losses) > 0:
            print(
                f"[Epoch {epoch:03d} Summary] "
                f"avg_total={epoch_avg_losses.get('loss_total', 0.0):.6f} | "
                f"avg_bin={epoch_avg_losses.get('loss_bin', 0.0):.6f} | "
                f"avg_src={epoch_avg_losses.get('loss_src', 0.0):.6f} | "
                f"avg_pair={epoch_avg_losses.get('loss_pair', 0.0):.6f} | "
                f"avg_rf={epoch_avg_losses.get('loss_rf_sep', 0.0):.6f} | "
                f"avg_dmar={epoch_avg_losses.get('loss_dm_ar', 0.0):.6f} | "
                f"avg_con_bin={epoch_avg_losses.get('loss_con_bin', 0.0):.6f} | "
                f"avg_con_src={epoch_avg_losses.get('loss_con_src', 0.0):.6f}"
            )

        # 保存 best train loss
        if avg_total_loss < best_train_loss:
            best_train_loss = avg_total_loss
            best_path = os.path.join(save_dir, "model_best_train_loss.pth")
            print(f"Saving best-train-loss checkpoint at epoch {epoch} -> {best_path}")
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                extra_state={
                    "best_train_loss": best_train_loss,
                    "epoch_avg_losses": epoch_avg_losses,
                },
            )

        # 按 epoch 保存
        if opt.save_epoch_freq > 0 and ((epoch + 1) % opt.save_epoch_freq == 0):
            epoch_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            print(f"Saving model at end of epoch {epoch} -> {epoch_path}")
            save_checkpoint(
                model,
                epoch_path,
                epoch=epoch,
                extra_state={
                    "best_train_loss": best_train_loss,
                    "epoch_avg_losses": epoch_avg_losses,
                },
            )

    total_elapsed = time.time() - start_time
    trained_steps = max(model.total_steps - global_step_start, 1)

    print(f"Training finished. Total time: {total_elapsed:.2f}s")
    print(f"Average time per step: {total_elapsed / trained_steps:.4f}s")
    print(f"Best training loss: {best_train_loss:.6f}")