import os
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt(train_opt):
    """
    基于训练配置构造验证配置，避免重复 parse 命令行。
    """
    val_opt = copy.deepcopy(train_opt)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.class_bal = False
    val_opt.jpg_method = ['pil']

    if hasattr(val_opt, "blur_sig") and len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]

    if hasattr(val_opt, "jpg_qual") and len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def get_current_loss_info(model):
    """
    尝试从 Trainer 中提取当前 loss 信息。
    不同项目的 Trainer 命名可能不同，这里做一个尽量兼容的读取。
    """
    loss_info = {}

    possible_attrs = [
        'loss',
        'loss_total',
        'loss_cls',
        'loss_ce',
        'loss_bce',
    ]

    for attr in possible_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            try:
                if hasattr(value, 'item'):
                    value = value.item()
                loss_info[attr] = float(value)
            except Exception:
                pass

    # 如果 Trainer 里有 get_current_losses()，优先补充/覆盖
    if hasattr(model, 'get_current_losses'):
        try:
            losses = model.get_current_losses()
            if isinstance(losses, dict):
                for k, v in losses.items():
                    try:
                        if hasattr(v, 'item'):
                            v = v.item()
                        loss_info[k] = float(v)
                    except Exception:
                        pass
        except Exception:
            pass

    return loss_info


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt(opt)

    model = Trainer(opt)

    train_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    print("Length of train loader: %d" % len(train_loader))
    print("Length of val loader: %d" % len(val_loader))

    best_acc = 0.0

    for epoch in range(opt.niter):
        print("=" * 80)
        print("Epoch:", epoch)

        model.train()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{opt.niter}",
            ncols=120
        )

        for i, data in pbar:
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            loss_info = get_current_loss_info(model)

            # tqdm 显示当前损失
            if len(loss_info) > 0:
                show_info = {}
                for k, v in loss_info.items():
                    show_info[k] = f"{v:.4f}"
                show_info["step"] = model.total_steps
                pbar.set_postfix(show_info)
            else:
                pbar.set_postfix(step=model.total_steps)

            # 写 tensorboard
            for k, v in loss_info.items():
                train_writer.add_scalar(k, v, model.total_steps)

        # ---------------- Validation ----------------
        model.eval()
        auroc, ap, r_acc, f_acc, acc = validate(model.model, val_loader)

        val_writer.add_scalar('auroc', auroc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('real_acc', r_acc, model.total_steps)
        val_writer.add_scalar('fake_acc', f_acc, model.total_steps)

        print(
            "(Val @ epoch {}) "
            "AUROC: {:.4f}, AP: {:.4f}, Acc@0.5: {:.4f}, Real Acc: {:.4f}, Fake Acc: {:.4f}".format(
                epoch, auroc, ap, acc, r_acc, f_acc
            )
        )

        # 定期保存普通 checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print('Saving regular checkpoint at epoch %d' % epoch)
            model.save_networks('model_epoch_%s.pth' % epoch)

        # 单独保存 best
        if acc > best_acc:
            best_acc = acc
            print('Saving best model at epoch %d, best_acc=%.4f' % (epoch, best_acc))
            model.save_networks('model_epoch_best.pth')