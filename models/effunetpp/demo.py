from torch import load, no_grad, argmax
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn import Sigmoid
from PIL.Image import open, fromarray
from os import listdir, makedirs
from os.path import join, splitext, exists, isdir
import numpy as np
from net import net

image_dir = "testimg"
output_dir = join(image_dir, "mask")
if not exists(output_dir):
    makedirs(output_dir)
data_transforms = Compose(
    [
        Resize((256, 256)),
        ToTensor(),
    ]
)
model = load("model.pth", weights_only=False, map_location="cpu")
model.eval()
with no_grad():
    for image_file in listdir(image_dir):
        image_path = join(image_dir, image_file)
        if isdir(image_path):
            continue
        image = open(image_path).convert("RGB")
        original_size = image.size
        input_image = data_transforms(image).unsqueeze(0).to("cpu")
        output, label = model(input_image)
        possibility = (1 - Sigmoid()(label).item()) * 100
        label = "Fake" if label.item() < 0 else "Real"
        output = argmax(output, dim=1)
        output = Compose([Resize((original_size[1], original_size[0]))])(
            output
        ).squeeze(0)
        output_np = output.detach().numpy()
        mask_np = output_np.astype(np.uint8) * 255
        mask_image = fromarray(mask_np)
        base_name, ext = splitext(image_file)
        mask_save_path = join(output_dir, f"{base_name}_mask{ext}")
        mask_image.save(mask_save_path)
        print(
            f"{label}\tFake possibility: {possibility:.2f}%\tMask saved: {mask_save_path}"
        )
