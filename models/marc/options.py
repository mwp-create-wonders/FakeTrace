import argparse
import os
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # =========================
        # 基础
        # =========================
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="Experiment name",
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            required=True,
            help="Models are saved here",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0 or 0,1,2 ; use -1 for CPU",
        )
        parser.add_argument(
            "--num_threads",
            default=8,
            type=int,
            help="# threads for loading data",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="input batch size",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=2026,
            help="random seed",
        )

        # =========================
        # 数据路径（6 文件夹）
        # =========================
        parser.add_argument("--real_dir", type=str, required=True, help="Path to REAL images")
        parser.add_argument("--real_processed_dir", type=str, required=True, help="Path to processed REAL images")
        parser.add_argument("--dm_dir", type=str, required=True, help="Path to DM reconstructed images")
        parser.add_argument("--dm_processed_dir", type=str, required=True, help="Path to processed DM images")
        parser.add_argument("--ar_dir", type=str, required=True, help="Path to AR reconstructed images")
        parser.add_argument("--ar_processed_dir", type=str, required=True, help="Path to processed AR images")

        parser.add_argument(
            "--strict_check",
            action="store_true",
            help="Require filenames in all 6 folders to match exactly",
        )
        parser.add_argument(
            "--dataset_limit",
            type=int,
            default=None,
            help="Use only first N matched samples for debugging",
        )

        # =========================
        # 图像尺寸与预处理
        # =========================
        parser.add_argument("--image_size", type=int, default=336, help="Resize image to this size")
        parser.add_argument("--crop_size", type=int, default=336, help="Random crop size (if enabled)")

        parser.add_argument(
            "--norm_mean",
            type=float,
            nargs=3,
            default=[0.485, 0.456, 0.406],
            help="Normalization mean",
        )
        parser.add_argument(
            "--norm_std",
            type=float,
            nargs=3,
            default=[0.229, 0.224, 0.225],
            help="Normalization std",
        )

        # =========================
        # dataloader
        # =========================
        parser.add_argument("--shuffle", action="store_true", help="Shuffle dataloader")
        parser.add_argument("--drop_last", action="store_true", help="Drop last incomplete batch")
        parser.add_argument("--pin_memory", action="store_true", help="Use pin_memory in dataloader")

        # =========================
        # 同步增强器
        # =========================
        parser.add_argument("--use_sync_aug", action="store_true", help="Use synchronized augmentation")
        parser.add_argument("--use_random_crop", action="store_true", help="Use synchronized random crop")
        parser.add_argument("--sync_hflip_prob", type=float, default=0.5, help="Synchronized horizontal flip prob")

        parser.add_argument("--sync_brightness", type=float, default=0.0, help="Synchronized brightness jitter strength")
        parser.add_argument("--sync_contrast", type=float, default=0.0, help="Synchronized contrast jitter strength")
        parser.add_argument("--sync_saturation", type=float, default=0.0, help="Synchronized saturation jitter strength")

        parser.add_argument("--use_sync_gamma", action="store_true", help="Use synchronized gamma jitter")
        parser.add_argument("--sync_gamma_min", type=float, default=0.95, help="Min gamma")
        parser.add_argument("--sync_gamma_max", type=float, default=1.05, help="Max gamma")

        parser.add_argument("--sync_blur_prob", type=float, default=0.0, help="Synchronized Gaussian blur prob")
        parser.add_argument("--sync_blur_min", type=float, default=0.3, help="Min blur radius")
        parser.add_argument("--sync_blur_max", type=float, default=1.0, help="Max blur radius")

        # =========================
        # 模型 / LoRA
        # =========================
        parser.add_argument("--backbone_name", type=str, default="dinov2_vitl14", help="DINOv2 backbone name")
        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA scaling factor")

        parser.add_argument("--proj_dim", type=int, default=256, help="Projection head output dim")
        parser.add_argument("--proj_hidden_dim", type=int, default=512, help="Projection head hidden dim")
        parser.add_argument("--dropout", type=float, default=0.0, help="Task head dropout")
        parser.add_argument("--return_tokens", action="store_true", help="Return patch tokens from backbone")

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        with open(os.path.join(expr_dir, "opt.txt"), "wt") as opt_file:
            opt_file.write(message + "\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # 处理 gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = [int(x) for x in str_ids if int(x) >= 0]

        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        else:
            opt.device = torch.device("cpu")

        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.isTrain = True

        # =========================
        # 训练
        # =========================
        parser.add_argument("--niter", type=int, default=10, help="Total epochs")
        parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
        parser.add_argument("--optim", type=str, default="adam", help="[sgd, adam]")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
        parser.add_argument("--t_max", type=int, default=1000, help="T_max for cosine scheduler")

        parser.add_argument(
            "--accumulation_steps",
            type=int,
            default=1,
            help="Gradient accumulation steps",
        )

        # =========================
        # 损失权重
        # 与新版 trainer.py 对齐
        # =========================
        parser.add_argument("--lambda_bin", type=float, default=1.0, help="Weight for binary classification loss")
        parser.add_argument("--lambda_src", type=float, default=0.30, help="Weight for source classification loss")
        parser.add_argument("--lambda_pair", type=float, default=0.40, help="Weight for pair consistency loss")
        parser.add_argument("--lambda_rf", type=float, default=0.25, help="Weight for real-fake separation loss")
        parser.add_argument("--lambda_dmar", type=float, default=0.10, help="Weight for DM-AR separation loss")
        parser.add_argument("--lambda_con_bin", type=float, default=0.05, help="Weight for binary-level contrastive loss")
        parser.add_argument("--lambda_con_src", type=float, default=0.01, help="Weight for source-level fake-only contrastive loss")

        # =========================
        # 损失开关
        # =========================
        parser.add_argument("--use_loss_bin", type=str2bool, default=True,
                            help="Whether to use binary classification loss")
        parser.add_argument("--use_loss_src", type=str2bool, default=True,
                            help="Whether to use source classification loss")
        parser.add_argument("--use_loss_pair", type=str2bool, default=True,
                            help="Whether to use pair consistency loss")
        parser.add_argument("--use_loss_rf", type=str2bool, default=True,
                            help="Whether to use real-fake separation loss")
        parser.add_argument("--use_loss_dmar", type=str2bool, default=True,
                            help="Whether to use DM-AR separation loss")
        parser.add_argument("--use_loss_con_bin", type=str2bool, default=True,
                            help="Whether to use binary-level contrastive loss")
        parser.add_argument("--use_loss_con_src", type=str2bool, default=True,
                            help="Whether to use source-level fake-only contrastive loss")

        # =========================
        # DM-AR 延迟启用
        # 与新版 trainer.py 对齐
        # =========================
        parser.add_argument("--dmar_start_epoch", type=int, default=1, help="Epoch to start DM-AR loss")
        parser.add_argument("--dmar_warmup_epochs", type=int, default=2, help="Warmup epochs for DM-AR loss")

        # =========================
        # Contrastive 分阶段调度
        # 新增
        # =========================
        parser.add_argument("--contrast_start_epoch", type=int, default=0, help="Epoch to start contrastive losses")
        parser.add_argument("--contrast_full_epoch", type=int, default=1, help="Epoch when contrastive warmup reaches full weight")
        parser.add_argument("--con_bin_decay_epoch", type=int, default=2, help="Epoch to decay binary-level contrastive loss")
        parser.add_argument("--con_bin_decay_ratio", type=float, default=0.3, help="Decay ratio for binary-level contrastive loss after decay epoch")
        parser.add_argument("--con_src_off_epoch", type=int, default=2, help="Epoch to turn off source-level contrastive loss")

        # =========================
        # 损失超参数
        # 与新版 trainer.py 对齐
        # =========================
        parser.add_argument("--margin_real_fake", type=float, default=0.30, help="Margin for real-fake separation")
        parser.add_argument("--margin_dm_ar", type=float, default=0.15, help="Margin for DM-AR separation")
        parser.add_argument("--temperature", type=float, default=0.10, help="Temperature for supervised contrastive loss")

        # =========================
        # 日志与保存
        # =========================
        parser.add_argument("--print_freq", type=int, default=50, help="Print losses every N steps")
        parser.add_argument("--save_latest_freq", type=int, default=5000, help="Save latest checkpoint every N steps")
        parser.add_argument("--save_epoch_freq", type=int, default=1, help="Save checkpoint every N epochs")

        # =========================
        # 恢复训练
        # =========================
        parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint for resuming")
        parser.add_argument("--resume_strict", action="store_true", help="Strict load model weights")
        parser.add_argument("--resume_optimizer", action="store_true", help="Resume optimizer state")
        parser.add_argument("--resume_scheduler", action="store_true", help="Resume scheduler state")

        return parser