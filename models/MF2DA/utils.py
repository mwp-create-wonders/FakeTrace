import itertools
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        print(output.shape)
        
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img


def _format_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _synchronize_if_needed(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _get_first_batch_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        if "img" in batch:
            return batch["img"]
    raise TypeError(f"Unsupported batch type for benchmarking: {type(batch)}")


def _build_profile_sample(images, min_batch_size=2):
    if images.shape[0] >= min_batch_size:
        return images[:min_batch_size]

    repeat_count = int(np.ceil(min_batch_size / images.shape[0]))
    expanded = images.repeat((repeat_count,) + (1,) * (images.ndim - 1))
    return expanded[:min_batch_size]


def collect_model_complexity_metrics(model, ckpt_path=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    model_size_mb = None
    if ckpt_path and os.path.isfile(ckpt_path):
        model_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    else:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

    return {
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "trainable_params_pct": trainable_pct,
        "model_size_mb": model_size_mb,
    }


def _estimate_gflops_with_hooks(model, sample_input):
    flops_holder = {"total": 0.0}
    hooks = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        batch_size = x.shape[0]
        out_tensor = output[0] if isinstance(output, (list, tuple)) else output
        output_elements = out_tensor.numel()
        kernel_ops = np.prod(module.kernel_size) * (module.in_channels / module.groups)
        bias_ops = 1 if module.bias is not None else 0
        flops_holder["total"] += output_elements * (2 * kernel_ops + bias_ops) / max(batch_size, 1)

    def linear_hook(module, inputs, output):
        x = inputs[0]
        batch_size = x.shape[0] if x.ndim > 0 else 1
        out_tensor = output[0] if isinstance(output, (list, tuple)) else output
        output_elements = out_tensor.numel()
        bias_ops = 1 if module.bias is not None else 0
        flops_holder["total"] += output_elements * (2 * module.in_features + bias_ops) / max(batch_size, 1)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for hook in hooks:
            hook.remove()

    if flops_holder["total"] <= 0:
        return None
    return flops_holder["total"] / 1e9


def estimate_gflops(model, sample_input, device):
    device = _format_device(device)
    batch_size = max(int(sample_input.shape[0]), 1)

    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=False, profile_memory=False, with_flops=True) as prof:
            with torch.no_grad():
                model(sample_input)

        total_flops = sum(getattr(evt, "flops", 0) for evt in prof.key_averages())
        if total_flops > 0:
            return total_flops / batch_size / 1e9, "torch.profiler"
    except Exception:
        pass

    hook_gflops = _estimate_gflops_with_hooks(model, sample_input)
    if hook_gflops is not None:
        return hook_gflops, "forward_hooks(approx)"

    return None, "unavailable"


def benchmark_inference(model, data_loader, device, warmup_steps=10, benchmark_steps=30):
    device = _format_device(device)
    if len(data_loader) == 0:
        return {
            "inference_time_ms_per_img": None,
            "throughput_img_s": None,
            "peak_gpu_memory_gb": None,
        }

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    was_training = model.training
    model.eval()

    total_time = 0.0
    total_images = 0

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    iterator = itertools.cycle(data_loader)
    with torch.no_grad():
        for step in range(max(warmup_steps, 0) + max(benchmark_steps, 0)):
            batch = next(iterator)
            images = _get_first_batch_images(batch).to(device, non_blocking=use_cuda)

            _synchronize_if_needed(device)
            start_time = time.perf_counter()
            _ = model(images)
            _synchronize_if_needed(device)
            elapsed = time.perf_counter() - start_time

            if step >= warmup_steps:
                total_time += elapsed
                total_images += images.shape[0]

    if was_training:
        model.train()

    inference_time_ms_per_img = None
    throughput_img_s = None
    if total_time > 0 and total_images > 0:
        inference_time_ms_per_img = total_time * 1000.0 / total_images
        throughput_img_s = total_images / total_time

    peak_gpu_memory_gb = None
    if use_cuda:
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return {
        "inference_time_ms_per_img": inference_time_ms_per_img,
        "throughput_img_s": throughput_img_s,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }


def collect_efficiency_metrics(
    model,
    data_loader,
    device,
    ckpt_path=None,
    warmup_steps=10,
    benchmark_steps=30,
):
    device = _format_device(device)
    complexity_metrics = collect_model_complexity_metrics(model, ckpt_path=ckpt_path)

    batch = next(iter(data_loader))
    sample_images = _get_first_batch_images(batch)
    sample_input = _build_profile_sample(sample_images, min_batch_size=2).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        gflops, gflops_method = estimate_gflops(model, sample_input, device)
    if was_training:
        model.train()

    runtime_metrics = benchmark_inference(
        model,
        data_loader,
        device=device,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )

    metrics = {}
    metrics.update(complexity_metrics)
    metrics.update(runtime_metrics)
    metrics["gflops"] = gflops
    metrics["gflops_method"] = gflops_method
    return metrics
