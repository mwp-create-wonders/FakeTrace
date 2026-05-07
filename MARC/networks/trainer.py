import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base_model import BaseModel
from models.dinov2_models_lora import DINOv2ModelWithLoRA


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt

        self.accumulation_steps = (
            opt.accumulation_steps if hasattr(opt, "accumulation_steps") else 1
        )
        self.current_step = 0
        self.current_epoch = 0

        self.use_tokens = getattr(opt, "return_tokens", False)

        # ========= LoRA 参数 =========
        lora_rank = getattr(opt, "lora_rank", 8)
        lora_alpha = getattr(opt, "lora_alpha", 1.0)
        lora_targets = getattr(
            opt,
            "lora_targets",
            ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )

        # ========= 模型 =========
        self.model = DINOv2ModelWithLoRA(
            name=getattr(opt, "backbone_name", "dinov2_vitl14"),
            binary_num_classes=1,
            source_num_classes=3,
            proj_dim=getattr(opt, "proj_dim", 256),
            proj_hidden_dim=getattr(opt, "proj_hidden_dim", 512),
            dropout=getattr(opt, "dropout", 0.0),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
        )

        # 初始化任务头
        nn.init.normal_(self.model.base_model.binary_head.weight.data, 0.0, 0.02)
        nn.init.constant_(self.model.base_model.binary_head.bias.data, 0.0)

        nn.init.normal_(self.model.base_model.source_head.weight.data, 0.0, 0.02)
        nn.init.constant_(self.model.base_model.source_head.bias.data, 0.0)

        if hasattr(self.model, "get_trainable_params"):
            params = self.model.get_trainable_params()
            print("Training with LoRA - only LoRA and task-head parameters will be updated")
        else:
            raise ValueError("LoRA model should have get_trainable_params method")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nTrainable parameters summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")

        # ========= 优化器 =========
        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(0.9, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=opt.lr,
                momentum=getattr(opt, "momentum", 0.9),
                weight_decay=opt.weight_decay,
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            eta_min=opt.lr * 0.001,
            T_max=getattr(opt, "t_max", 1000),
        )

        # ========= 基础损失 =========
        self.binary_loss_fn = nn.BCEWithLogitsLoss()
        self.source_loss_fn = nn.CrossEntropyLoss()

        # ========= 距离分离超参数 =========
        # 相比原版更柔和，避免后期 fake 表征被压坏
        self.margin_real_fake = getattr(opt, "margin_real_fake", 0.30)
        self.margin_dm_ar = getattr(opt, "margin_dm_ar", 0.15)
        self.temperature = getattr(opt, "temperature", 0.10)

        # ========= 损失权重 =========
        # 让 bin 任务继续主导；辅助项适当减弱
        self.lambda_bin = getattr(opt, "lambda_bin", 1.0)
        self.lambda_src = getattr(opt, "lambda_src", 0.30)
        self.lambda_pair = getattr(opt, "lambda_pair", 0.40)
        self.lambda_rf = getattr(opt, "lambda_rf", 0.25)
        self.lambda_dmar = getattr(opt, "lambda_dmar", 0.10)
        self.lambda_con_bin = getattr(opt, "lambda_con_bin", 0.05)
        self.lambda_con_src = getattr(opt, "lambda_con_src", 0.01)

        # ========= 开关控制 =========
        # 只要开关是 False，该 loss 就不会进入 total_loss
        self.use_loss_bin = getattr(opt, "use_loss_bin", True)
        self.use_loss_src = getattr(opt, "use_loss_src", True)
        self.use_loss_pair = getattr(opt, "use_loss_pair", True)
        self.use_loss_rf = getattr(opt, "use_loss_rf", True)
        self.use_loss_dmar = getattr(opt, "use_loss_dmar", True)
        self.use_loss_con_bin = getattr(opt, "use_loss_con_bin", True)
        self.use_loss_con_src = getattr(opt, "use_loss_con_src", True)

        # ========= DM-AR loss 延迟启用 =========
        # 原版有 dmar 的 warmup，这里保留，但提前到 epoch=1
        self.dmar_start_epoch = getattr(opt, "dmar_start_epoch", 1)
        self.dmar_warmup_epochs = getattr(opt, "dmar_warmup_epochs", 2)

        # ========= Contrastive loss 分阶段控制 =========
        # 目的：
        # 1) 前期轻量启用，不要一开始就强行塑形
        # 2) 中期发挥作用
        # 3) 后期衰减/关闭，避免 fake accuracy 持续下降
        self.contrast_start_epoch = getattr(opt, "contrast_start_epoch", 0)
        self.contrast_full_epoch = getattr(opt, "contrast_full_epoch", 1)

        # epoch >= con_bin_decay_epoch 后，binary contrastive 衰减
        self.con_bin_decay_epoch = getattr(opt, "con_bin_decay_epoch", 2)
        self.con_bin_decay_ratio = getattr(opt, "con_bin_decay_ratio", 0.3)

        # epoch >= con_src_off_epoch 后，source contrastive 关闭
        self.con_src_off_epoch = getattr(opt, "con_src_off_epoch", 2)

        self.model.to(self.device)
        self.optimizer.zero_grad()

    # =========================================================
    # epoch 控制
    # =========================================================
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_current_lambda_dmar(self):
        """
        延迟启用 DM-AR separation loss:
        - 若 use_loss_dmar=False，则恒为 0
        - current_epoch < dmar_start_epoch: 权重为 0
        - 到达 dmar_start_epoch 后，线性 warmup 到 lambda_dmar
        """
        if not self.use_loss_dmar:
            return 0.0

        if self.current_epoch < self.dmar_start_epoch:
            return 0.0

        if self.dmar_warmup_epochs <= 0:
            return self.lambda_dmar

        progress = (self.current_epoch - self.dmar_start_epoch + 1) / float(
            self.dmar_warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.lambda_dmar * progress

    def get_current_lambda_con_bin(self):
        """
        binary-level contrastive:
        - 前期 warmup
        - 后期衰减
        """
        if not self.use_loss_con_bin:
            return 0.0

        if self.current_epoch < self.contrast_start_epoch:
            return 0.0

        # warmup 阶段
        if self.current_epoch < self.contrast_full_epoch:
            denom = max(1, self.contrast_full_epoch - self.contrast_start_epoch + 1)
            progress = (self.current_epoch - self.contrast_start_epoch + 1) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return self.lambda_con_bin * progress

        # 后期衰减
        if self.current_epoch >= self.con_bin_decay_epoch:
            return self.lambda_con_bin * self.con_bin_decay_ratio

        return self.lambda_con_bin

    def get_current_lambda_con_src(self):
        """
        source-level contrastive(fake-only):
        - 前期 warmup
        - 后期直接关闭，避免 fake 被过度压缩
        """
        if not self.use_loss_con_src:
            return 0.0

        if self.current_epoch < self.contrast_start_epoch:
            return 0.0

        if self.current_epoch >= self.con_src_off_epoch:
            return 0.0

        if self.current_epoch < self.contrast_full_epoch:
            denom = max(1, self.contrast_full_epoch - self.contrast_start_epoch + 1)
            progress = (self.current_epoch - self.contrast_start_epoch + 1) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return self.lambda_con_src * progress

        return self.lambda_con_src

    # =========================================================
    # 数据输入
    # =========================================================
    def set_input(self, batch):
        self.batch = {}

        image_keys = [
            "real",
            "real_processed",
            "dm",
            "dm_processed",
            "ar",
            "ar_processed",
        ]

        label_keys = [
            "binary_label_real",
            "binary_label_real_processed",
            "binary_label_dm",
            "binary_label_dm_processed",
            "binary_label_ar",
            "binary_label_ar_processed",
            "source_label_real",
            "source_label_real_processed",
            "source_label_dm",
            "source_label_dm_processed",
            "source_label_ar",
            "source_label_ar_processed",
        ]

        meta_keys = [
            "filename",
            "group_id",
            "identity_real",
            "identity_dm",
            "identity_ar",
        ]

        for key in image_keys:
            self.batch[key] = batch[key].to(self.device, non_blocking=True)

        for key in label_keys:
            self.batch[key] = batch[key].to(self.device, non_blocking=True)

        for key in meta_keys:
            self.batch[key] = batch[key]

        self.batch_size = self.batch["real"].size(0)

    # =========================================================
    # 前向
    # =========================================================
    def _split_tensor_by_route(self, x, bsz):
        return {
            "real": x[0 * bsz: 1 * bsz],
            "real_processed": x[1 * bsz: 2 * bsz],
            "dm": x[2 * bsz: 3 * bsz],
            "dm_processed": x[3 * bsz: 4 * bsz],
            "ar": x[4 * bsz: 5 * bsz],
            "ar_processed": x[5 * bsz: 6 * bsz],
        }

    def forward(self):
        bsz = self.batch_size

        inputs_cat = torch.cat(
            [
                self.batch["real"],
                self.batch["real_processed"],
                self.batch["dm"],
                self.batch["dm_processed"],
                self.batch["ar"],
                self.batch["ar_processed"],
            ],
            dim=0,
        )

        outputs = self.model(
            inputs_cat,
            return_feature=True,
            return_tokens=self.use_tokens,
        )

        self.outputs = {}
        self.outputs["feature"] = self._split_tensor_by_route(outputs["feature"], bsz)
        self.outputs["binary_logits"] = self._split_tensor_by_route(outputs["binary_logits"], bsz)
        self.outputs["source_logits"] = self._split_tensor_by_route(outputs["source_logits"], bsz)
        self.outputs["proj"] = self._split_tensor_by_route(outputs["proj"], bsz)

        if "tokens" in outputs and outputs["tokens"] is not None:
            self.outputs["tokens"] = self._split_tensor_by_route(outputs["tokens"], bsz)
        else:
            self.outputs["tokens"] = None

        return self.outputs

    # =========================================================
    # 工具函数
    # =========================================================
    @staticmethod
    def cosine_distance(x, y):
        """
        cosine distance = 1 - cosine similarity
        x, y: [B, C]
        """
        return 1.0 - F.cosine_similarity(x, y, dim=1)

    def pair_consistency_loss(self, x, y):
        """
        希望 x 与 y 距离小
        """
        return self.cosine_distance(x, y).mean()

    def margin_separation_loss(self, x, y, margin):
        """
        希望 x 与 y 距离大于 margin
        若 dist < margin，则有惩罚
        """
        dist = self.cosine_distance(x, y)
        loss = F.relu(margin - dist).mean()
        return loss

    def supcon_loss(self, features, labels, temperature=0.07):
        """
        一个不依赖额外库的 supervised contrastive loss
        features: [N, D]
        labels:   [N]
        """
        device = features.device
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)

        loss = -mean_log_prob_pos
        valid_mask = mask_sum > 0
        if valid_mask.any():
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device)

        return loss

    def supcon_loss_with_mask(self, features, positive_mask, valid_mask=None, temperature=0.07):
        """
        features: [N, D]
        positive_mask: [N, N]
        valid_mask: [N]
        """
        device = features.device
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        positive_mask = positive_mask.float().to(device)

        logits_mask = torch.ones_like(positive_mask) - torch.eye(positive_mask.shape[0], device=device)
        positive_mask = positive_mask * logits_mask

        if valid_mask is not None:
            valid_mask = valid_mask.float().to(device).view(-1, 1)
            pair_valid = torch.matmul(valid_mask, valid_mask.T)
            logits_mask = logits_mask * pair_valid
            positive_mask = positive_mask * pair_valid

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        loss = -mean_log_prob_pos

        if valid_mask is not None:
            valid_vec = valid_mask.squeeze(1) > 0
            final_mask = (pos_count > 0) & valid_vec
        else:
            final_mask = pos_count > 0

        if final_mask.any():
            loss = loss[final_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device)

        return loss

    # =========================================================
    # 各项 loss
    # =========================================================
    def compute_binary_loss(self):
        logits = torch.cat(
            [
                self.outputs["binary_logits"]["real"],
                self.outputs["binary_logits"]["real_processed"],
                self.outputs["binary_logits"]["dm"],
                self.outputs["binary_logits"]["dm_processed"],
                self.outputs["binary_logits"]["ar"],
                self.outputs["binary_logits"]["ar_processed"],
            ],
            dim=0,
        ).squeeze(1)

        labels = torch.cat(
            [
                self.batch["binary_label_real"],
                self.batch["binary_label_real_processed"],
                self.batch["binary_label_dm"],
                self.batch["binary_label_dm_processed"],
                self.batch["binary_label_ar"],
                self.batch["binary_label_ar_processed"],
            ],
            dim=0,
        ).float()

        return self.binary_loss_fn(logits, labels)

    def compute_source_loss(self):
        logits = torch.cat(
            [
                self.outputs["source_logits"]["real"],
                self.outputs["source_logits"]["real_processed"],
                self.outputs["source_logits"]["dm"],
                self.outputs["source_logits"]["dm_processed"],
                self.outputs["source_logits"]["ar"],
                self.outputs["source_logits"]["ar_processed"],
            ],
            dim=0,
        )

        labels = torch.cat(
            [
                self.batch["source_label_real"],
                self.batch["source_label_real_processed"],
                self.batch["source_label_dm"],
                self.batch["source_label_dm_processed"],
                self.batch["source_label_ar"],
                self.batch["source_label_ar_processed"],
            ],
            dim=0,
        )

        return self.source_loss_fn(logits, labels)

    def compute_pair_loss(self):
        feat_real = self.outputs["feature"]["real"]
        feat_real_p = self.outputs["feature"]["real_processed"]

        feat_dm = self.outputs["feature"]["dm"]
        feat_dm_p = self.outputs["feature"]["dm_processed"]

        feat_ar = self.outputs["feature"]["ar"]
        feat_ar_p = self.outputs["feature"]["ar_processed"]

        loss_real_pair = self.pair_consistency_loss(feat_real, feat_real_p)
        loss_dm_pair = self.pair_consistency_loss(feat_dm, feat_dm_p)
        loss_ar_pair = self.pair_consistency_loss(feat_ar, feat_ar_p)

        return (loss_real_pair + loss_dm_pair + loss_ar_pair) / 3.0

    def compute_real_fake_separation_loss(self):
        feat_real = self.outputs["feature"]["real"]
        feat_real_p = self.outputs["feature"]["real_processed"]

        feat_dm = self.outputs["feature"]["dm"]
        feat_dm_p = self.outputs["feature"]["dm_processed"]

        feat_ar = self.outputs["feature"]["ar"]
        feat_ar_p = self.outputs["feature"]["ar_processed"]

        losses = [
            self.margin_separation_loss(feat_real, feat_dm, self.margin_real_fake),
            self.margin_separation_loss(feat_real, feat_dm_p, self.margin_real_fake),
            self.margin_separation_loss(feat_real, feat_ar, self.margin_real_fake),
            self.margin_separation_loss(feat_real, feat_ar_p, self.margin_real_fake),
            self.margin_separation_loss(feat_real_p, feat_dm, self.margin_real_fake),
            self.margin_separation_loss(feat_real_p, feat_dm_p, self.margin_real_fake),
            self.margin_separation_loss(feat_real_p, feat_ar, self.margin_real_fake),
            self.margin_separation_loss(feat_real_p, feat_ar_p, self.margin_real_fake),
        ]

        return sum(losses) / len(losses)

    def compute_dm_ar_separation_loss(self):
        feat_dm = self.outputs["feature"]["dm"]
        feat_dm_p = self.outputs["feature"]["dm_processed"]

        feat_ar = self.outputs["feature"]["ar"]
        feat_ar_p = self.outputs["feature"]["ar_processed"]

        losses = [
            self.margin_separation_loss(feat_dm, feat_ar, self.margin_dm_ar),
            self.margin_separation_loss(feat_dm, feat_ar_p, self.margin_dm_ar),
            self.margin_separation_loss(feat_dm_p, feat_ar, self.margin_dm_ar),
            self.margin_separation_loss(feat_dm_p, feat_ar_p, self.margin_dm_ar),
        ]

        return sum(losses) / len(losses)

    def compute_contrastive_loss(self):
        """
        同时返回:
        1) binary-level contrastive
        2) source-level contrastive(fake-only)
        """
        proj_real = self.outputs["proj"]["real"]
        proj_real_p = self.outputs["proj"]["real_processed"]
        proj_dm = self.outputs["proj"]["dm"]
        proj_dm_p = self.outputs["proj"]["dm_processed"]
        proj_ar = self.outputs["proj"]["ar"]
        proj_ar_p = self.outputs["proj"]["ar_processed"]

        proj = torch.cat(
            [proj_real, proj_real_p, proj_dm, proj_dm_p, proj_ar, proj_ar_p],
            dim=0,
        )

        bsz = proj_real.size(0)
        device = proj.device

        binary_labels = torch.cat(
            [
                torch.ones(bsz, device=device, dtype=torch.long),
                torch.ones(bsz, device=device, dtype=torch.long),
                torch.zeros(bsz, device=device, dtype=torch.long),
                torch.zeros(bsz, device=device, dtype=torch.long),
                torch.zeros(bsz, device=device, dtype=torch.long),
                torch.zeros(bsz, device=device, dtype=torch.long),
            ],
            dim=0,
        )

        loss_con_bin = self.supcon_loss(
            proj,
            binary_labels,
            temperature=self.temperature,
        )

        source_labels = torch.cat(
            [
                torch.full((bsz,), -1, device=device, dtype=torch.long),
                torch.full((bsz,), -1, device=device, dtype=torch.long),
                torch.ones(bsz, device=device, dtype=torch.long),
                torch.ones(bsz, device=device, dtype=torch.long),
                torch.full((bsz,), 2, device=device, dtype=torch.long),
                torch.full((bsz,), 2, device=device, dtype=torch.long),
            ],
            dim=0,
        )

        valid_mask = (source_labels >= 0).float()
        source_labels_clamped = source_labels.clone()
        source_labels_clamped[source_labels_clamped < 0] = 999999

        positive_mask = torch.eq(
            source_labels_clamped.view(-1, 1),
            source_labels_clamped.view(1, -1),
        ).float()

        loss_con_src = self.supcon_loss_with_mask(
            proj,
            positive_mask=positive_mask,
            valid_mask=valid_mask,
            temperature=self.temperature,
        )

        return loss_con_bin, loss_con_src

    # =========================================================
    # loss 组合与开关
    # =========================================================
    def build_loss_weights(self):
        current_lambda_dmar = self.get_current_lambda_dmar()
        current_lambda_con_bin = self.get_current_lambda_con_bin()
        current_lambda_con_src = self.get_current_lambda_con_src()

        return {
            "loss_bin": self.lambda_bin if self.use_loss_bin else 0.0,
            "loss_src": self.lambda_src if self.use_loss_src else 0.0,
            "loss_pair": self.lambda_pair if self.use_loss_pair else 0.0,
            "loss_rf_sep": self.lambda_rf if self.use_loss_rf else 0.0,
            "loss_dm_ar": current_lambda_dmar,
            "loss_con_bin": current_lambda_con_bin,
            "loss_con_src": current_lambda_con_src,
        }

    def compute_losses(self):
        """
        所有 loss 都保留计算，但是否参与 total_loss 由开关与权重共同决定。
        """
        loss_bin = self.compute_binary_loss()
        loss_src = self.compute_source_loss()
        loss_pair = self.compute_pair_loss()
        loss_rf_sep = self.compute_real_fake_separation_loss()
        loss_dm_ar = self.compute_dm_ar_separation_loss()
        loss_con_bin, loss_con_src = self.compute_contrastive_loss()

        loss_dict = {
            "loss_bin": loss_bin,
            "loss_src": loss_src,
            "loss_pair": loss_pair,
            "loss_rf_sep": loss_rf_sep,
            "loss_dm_ar": loss_dm_ar,
            "loss_con_bin": loss_con_bin,
            "loss_con_src": loss_con_src,
        }

        loss_weights = self.build_loss_weights()

        total_loss = torch.zeros((), device=loss_bin.device, dtype=loss_bin.dtype)
        for key, value in loss_dict.items():
            total_loss = total_loss + loss_weights[key] * value

        loss_dict["loss_total"] = total_loss
        loss_dict["lambda_dmar_current"] = torch.tensor(
            loss_weights["loss_dm_ar"],
            device=loss_bin.device,
            dtype=loss_bin.dtype,
        )
        loss_dict["lambda_con_bin_current"] = torch.tensor(
            loss_weights["loss_con_bin"],
            device=loss_bin.device,
            dtype=loss_bin.dtype,
        )
        loss_dict["lambda_con_src_current"] = torch.tensor(
            loss_weights["loss_con_src"],
            device=loss_bin.device,
            dtype=loss_bin.dtype,
        )

        # 方便日志里直接看哪些 loss 当前是开着的
        for key, value in loss_weights.items():
            loss_dict[f"weight_{key}"] = torch.tensor(
                value,
                device=loss_bin.device,
                dtype=loss_bin.dtype,
            )

        return loss_dict

    # =========================================================
    # 优化
    # =========================================================
    def optimize_parameters(self):
        self.current_step += 1
        self.total_steps += 1

        self.forward()
        loss_dict = self.compute_losses()

        self.loss_dict = loss_dict
        self.loss = loss_dict["loss_total"] / self.accumulation_steps
        self.loss.backward()

        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def get_current_losses(self):
        if not hasattr(self, "loss_dict"):
            return {}

        return {
            k: (v.item() if torch.is_tensor(v) else float(v))
            for k, v in self.loss_dict.items()
        }

    def set_model_train(self):
        self.model.train()

    def set_model_eval(self):
        self.model.eval()

    def finalize_epoch(self):
        if self.current_step % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()