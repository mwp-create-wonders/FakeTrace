import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        if hasattr(opt, "device"):
            self.device = opt.device
        else:
            self.device = (
                torch.device(f"cuda:{opt.gpu_ids[0]}")
                if hasattr(opt, "gpu_ids") and len(opt.gpu_ids) > 0
                else torch.device("cpu")
            )

        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _unwrap_model(self):
        if self.model is None:
            raise ValueError("self.model is None, cannot unwrap model.")
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        model_to_save = self._unwrap_model()

        state_dict = {
            "model": model_to_save.state_dict(),
            "total_steps": self.total_steps,
        }

        if self.optimizer is not None:
            state_dict["optimizer"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()

        torch.save(state_dict, save_path)
        print(f"[BaseModel] Saved checkpoint to: {save_path}")

    def load_networks(self, load_filename, load_optimizer=True, load_scheduler=True, strict=True):
        load_path = os.path.join(self.save_dir, load_filename)
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        model_to_load = self._unwrap_model()
        model_to_load.load_state_dict(checkpoint["model"], strict=strict)

        if load_optimizer and self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_scheduler and self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]

        print(f"[BaseModel] Loaded checkpoint from: {load_path}")

    def set_model_train(self):
        if self.model is not None:
            self.model.train()

    def set_model_eval(self):
        if self.model is not None:
            self.model.eval()

    def test(self):
        self.set_model_eval()
        with torch.no_grad():
            return self.forward()

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward().")


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)