import os
import time
import copy
import torch

from tensorboardX import SummaryWriter
from tqdm import tqdm

from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from utils import collect_efficiency_metrics


def get_val_opt(opt):
    """
    基于训练参数构造验证参数：
    val_real_dirs / val_fake_dirs
    如果未提供 val_real_dirs / val_fake_dirs，则默认复用训练集
    """
    val_opt = copy.deepcopy(opt)

    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.no_flip = True

    # 若你的 BaseOptions 里没有这两个参数，请同步加上
    if not hasattr(val_opt, "no_resize"):
        val_opt.no_resize = False
    if not hasattr(val_opt, "no_crop"):
        val_opt.no_crop = False

    val_opt.no_resize = False
    val_opt.no_crop = False

    # 验证阶段固定退化设置
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]

    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    # 优先使用独立验证集；若未提供，则复用训练集
    if hasattr(opt, "val_real_dirs") and opt.val_real_dirs:
        val_opt.real_dirs = opt.val_real_dirs
    else:
        val_opt.real_dirs = opt.real_dirs

    if hasattr(opt, "val_fake_dirs") and opt.val_fake_dirs:
        val_opt.fake_dirs = opt.val_fake_dirs
    else:
        val_opt.fake_dirs = opt.fake_dirs

    if hasattr(opt, "val_real_num"):
        val_opt.real_num = opt.val_real_num
    if hasattr(opt, "val_fake_num"):
        val_opt.fake_num = opt.val_fake_num

    return val_opt


def format_metric(value, fmt=".6f"):
    if value is None:
        return "N/A"
    return format(value, fmt)


def print_efficiency_metrics(metrics):
    print("\n===== Training Efficiency Metrics =====")
    print(f"Params (M): {format_metric(metrics.get('params_m'))}")
    print(
        f"Trainable Params (M / %): "
        f"{format_metric(metrics.get('trainable_params_m'))} / "
        f"{format_metric(metrics.get('trainable_params_pct'))}"
    )
    print(
        f"GFLOPs: {format_metric(metrics.get('gflops'))} "
        f"(method: {metrics.get('gflops_method', 'N/A')})"
    )
    print(f"Inference Time (ms/img): {format_metric(metrics.get('inference_time_ms_per_img'))}")
    print(f"Throughput (img/s): {format_metric(metrics.get('throughput_img_s'))}")
    print(f"Peak GPU Memory (GB): {format_metric(metrics.get('peak_gpu_memory_gb'))}")
    print(f"Model Size (MB): {format_metric(metrics.get('model_size_mb'))}")


def save_efficiency_metrics(exp_dir, opt, metrics):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    txt_path = os.path.join(exp_dir, "efficiency_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {opt.arch}\n")
        f.write(f"Real Dirs: {opt.real_dirs}\n")
        f.write(f"Fake Dirs: {opt.fake_dirs}\n")
        f.write(f"Batch Size: {opt.batch_size}\n")
        f.write(f"Fix Backbone: {opt.fix_backbone}\n")
        f.write(f"Params (M): {format_metric(metrics.get('params_m'))}\n")
        f.write(
            f"Trainable Params (M / %): "
            f"{format_metric(metrics.get('trainable_params_m'))} / "
            f"{format_metric(metrics.get('trainable_params_pct'))}\n"
        )
        f.write(f"GFLOPs: {format_metric(metrics.get('gflops'))}\n")
        f.write(
            f"Inference Time (ms/img): "
            f"{format_metric(metrics.get('inference_time_ms_per_img'))}\n"
        )
        f.write(f"Throughput (img/s): {format_metric(metrics.get('throughput_img_s'))}\n")
        f.write(f"Peak GPU Memory (GB): {format_metric(metrics.get('peak_gpu_memory_gb'))}\n")
        f.write(f"Model Size (MB): {format_metric(metrics.get('model_size_mb'))}\n")
        f.write(f"GFLOPs Method: {metrics.get('gflops_method', 'N/A')}\n")

    csv_path = os.path.join(exp_dir, "efficiency_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write(
            "timestamp,arch,batch_size,fix_backbone,params_m,trainable_params_m,"
            "trainable_params_pct,gflops,gflops_method,inference_time_ms_per_img,"
            "throughput_img_s,peak_gpu_memory_gb,model_size_mb\n"
        )
        f.write(
            f"{timestamp},{opt.arch},{opt.batch_size},{opt.fix_backbone},"
            f"{'' if metrics.get('params_m') is None else format_metric(metrics.get('params_m'))},"
            f"{'' if metrics.get('trainable_params_m') is None else format_metric(metrics.get('trainable_params_m'))},"
            f"{'' if metrics.get('trainable_params_pct') is None else format_metric(metrics.get('trainable_params_pct'))},"
            f"{'' if metrics.get('gflops') is None else format_metric(metrics.get('gflops'))},"
            f"{metrics.get('gflops_method', '')},"
            f"{'' if metrics.get('inference_time_ms_per_img') is None else format_metric(metrics.get('inference_time_ms_per_img'))},"
            f"{'' if metrics.get('throughput_img_s') is None else format_metric(metrics.get('throughput_img_s'))},"
            f"{'' if metrics.get('peak_gpu_memory_gb') is None else format_metric(metrics.get('peak_gpu_memory_gb'))},"
            f"{'' if metrics.get('model_size_mb') is None else format_metric(metrics.get('model_size_mb'))}\n"
        )

    print(f"Efficiency metrics saved to: {txt_path}")
    print(f"Efficiency CSV saved to: {csv_path}")


if __name__ == '__main__':
    # 训练参数
    opt = TrainOptions().parse()
    val_opt = get_val_opt(opt)

    # 模型
    model = Trainer(opt)

    # 数据
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    # 日志
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    exp_dir = os.path.join(opt.checkpoints_dir, opt.name)

    # 验证结果目录
    val_results_dir = os.path.join(exp_dir, "val_results")
    os.makedirs(val_results_dir, exist_ok=True)

    # 验证 CSV
    val_results_file = os.path.join(val_results_dir, "validation_results.csv")
    with open(val_results_file, 'w') as f:
        f.write("epoch,ap,auc,r_acc0,f_acc0,acc0,r_acc1,f_acc1,acc1,best_thres,timestamp\n")

    start_time = time.time()
    print("Length of data loader: %d" % len(data_loader))
    print("Length of val loader: %d" % len(val_loader))

    efficiency_metrics = collect_efficiency_metrics(
        model.model,
        data_loader,
        device=model.device,
        warmup_steps=opt.benchmark_warmup,
        benchmark_steps=opt.benchmark_steps,
    )
    print_efficiency_metrics(efficiency_metrics)
    save_efficiency_metrics(exp_dir, opt, efficiency_metrics)

    if efficiency_metrics.get('params_m') is not None:
        train_writer.add_scalar('efficiency/params_m', efficiency_metrics['params_m'], 0)
    if efficiency_metrics.get('trainable_params_m') is not None:
        train_writer.add_scalar('efficiency/trainable_params_m', efficiency_metrics['trainable_params_m'], 0)
    if efficiency_metrics.get('trainable_params_pct') is not None:
        train_writer.add_scalar('efficiency/trainable_params_pct', efficiency_metrics['trainable_params_pct'], 0)
    if efficiency_metrics.get('gflops') is not None:
        train_writer.add_scalar('efficiency/gflops', efficiency_metrics['gflops'], 0)
    if efficiency_metrics.get('inference_time_ms_per_img') is not None:
        train_writer.add_scalar('efficiency/inference_time_ms_per_img', efficiency_metrics['inference_time_ms_per_img'], 0)
    if efficiency_metrics.get('throughput_img_s') is not None:
        train_writer.add_scalar('efficiency/throughput_img_s', efficiency_metrics['throughput_img_s'], 0)
    if efficiency_metrics.get('peak_gpu_memory_gb') is not None:
        train_writer.add_scalar('efficiency/peak_gpu_memory_gb', efficiency_metrics['peak_gpu_memory_gb'], 0)
    if efficiency_metrics.get('model_size_mb') is not None:
        train_writer.add_scalar('efficiency/model_size_mb', efficiency_metrics['model_size_mb'], 0)

    best_acc = 0.0
    best_epoch = -1

    for epoch in range(opt.niter):
        print(f"\nEpoch {epoch + 1}/{opt.niter}")

        # ===================== 训练阶段 =====================
        model.train()
        train_bar = tqdm(
            data_loader,
            desc=f"Training Epoch {epoch}",
            leave=True,
            ncols=120,
            unit="batch"
        )

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for i, data in enumerate(train_bar):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            # 兼容 model.loss 是 tensor 或 float
            cur_loss = None
            if hasattr(model, "loss"):
                try:
                    cur_loss = float(model.loss.detach().cpu()) if hasattr(model.loss, "detach") else float(model.loss)
                except Exception:
                    cur_loss = None

            if cur_loss is not None:
                epoch_loss_sum += cur_loss
                epoch_step_count += 1
                train_bar.set_postfix(loss=f"{cur_loss:.4f}")

            # TensorBoard 训练 loss
            if cur_loss is not None and (model.total_steps % opt.loss_freq == 0):
                train_writer.add_scalar('loss', cur_loss, model.total_steps)

        train_bar.close()

        avg_train_loss = epoch_loss_sum / max(epoch_step_count, 1)
        train_writer.add_scalar('epoch_loss', avg_train_loss, epoch)
        print(f"Train loss: {avg_train_loss:.6f}")

        # 保存周期模型
        if ((epoch + 1) % opt.save_epoch_freq == 0) or (epoch == opt.niter - 1):
            print(f"Saving the model at the end of epoch {epoch}")
            model.save_networks(f'model_epoch_{epoch}.pth')

        # ===================== 验证阶段 =====================
        model.eval()
        print(f"Validating Epoch {epoch}")

        with torch.no_grad():
            ap, auc, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model.model, val_loader)

        # TensorBoard
        val_writer.add_scalar('ap', ap, epoch)
        val_writer.add_scalar('auc', auc, epoch)
        val_writer.add_scalar('r_acc0', r_acc0, epoch)
        val_writer.add_scalar('f_acc0', f_acc0, epoch)
        val_writer.add_scalar('acc0', acc0, epoch)
        val_writer.add_scalar('r_acc1', r_acc1, epoch)
        val_writer.add_scalar('f_acc1', f_acc1, epoch)
        val_writer.add_scalar('acc1', acc1, epoch)
        val_writer.add_scalar('best_thres', best_thres, epoch)

        # 保存验证结果到 CSV
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(val_results_file, 'a') as f:
            f.write(
                f"{epoch},{ap:.6f},{auc:.6f},{r_acc0:.6f},{f_acc0:.6f},"
                f"{acc0:.6f},{r_acc1:.6f},{f_acc1:.6f},{acc1:.6f},"
                f"{best_thres:.6f},{timestamp}\n"
            )

        # 每个 epoch 的详细验证结果
        epoch_val_file = os.path.join(val_results_dir, f"val_epoch_{epoch}.txt")
        with open(epoch_val_file, 'w') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"AP: {ap:.6f}\n")
            f.write(f"AUC: {auc:.6f}\n")
            f.write(f"Real Accuracy 0: {r_acc0:.6f}\n")
            f.write(f"Fake Accuracy 0: {f_acc0:.6f}\n")
            f.write(f"Overall Accuracy 0: {acc0:.6f}\n")
            f.write(f"Real Accuracy 1: {r_acc1:.6f}\n")
            f.write(f"Fake Accuracy 1: {f_acc1:.6f}\n")
            f.write(f"Overall Accuracy 1: {acc1:.6f}\n")
            f.write(f"Best Threshold: {best_thres:.6f}\n")

        print(f"(Val @ epoch {epoch}) acc: {acc0:.4f}; auc: {auc:.4f}; ap: {ap:.4f}")
        print(f"Validation results saved to: {epoch_val_file}")

        # 最佳模型
        if acc0 > best_acc:
            best_acc = acc0
            best_epoch = epoch
            model.save_networks('model_epoch_best.pth')

            best_val_file = os.path.join(val_results_dir, "best_validation_results.txt")
            with open(best_val_file, 'w') as f:
                f.write(f"Best Epoch: {epoch}\n")
                f.write(f"Best Accuracy: {best_acc:.6f}\n")
                f.write(f"AP: {ap:.6f}\n")
                f.write(f"AUC: {auc:.6f}\n")
                f.write(f"Real Accuracy 0: {r_acc0:.6f}\n")
                f.write(f"Fake Accuracy 0: {f_acc0:.6f}\n")
                f.write(f"Overall Accuracy 0: {acc0:.6f}\n")
                f.write(f"Real Accuracy 1: {r_acc1:.6f}\n")
                f.write(f"Fake Accuracy 1: {f_acc1:.6f}\n")
                f.write(f"Overall Accuracy 1: {acc1:.6f}\n")
                f.write(f"Best Threshold: {best_thres:.6f}\n")
                f.write(f"Timestamp: {timestamp}\n")

            print(f"Best model updated! Accuracy: {best_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best acc: {best_acc:.6f}")
    print(f"Total time: {total_time / 3600:.2f} hours")

    train_writer.close()
    val_writer.close()
