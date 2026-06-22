import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from vit_model import vit_base_patch16_224
from models import get_model
from collections import OrderedDict


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.model.visual.transformer.resblocks.img_size
        patch_size = model.model.visual.transformer.resblocks.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    # model = vit_base_patch16_224()
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    # weights_path = "./vit_base_patch16_224.pth"
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    # target_layers = [model.blocks[-1].norm1]
    
    model = get_model("CLIP:ViT-L/14")
    
    state_dict = torch.load("/home/mwp/UniversalFakeDetect-main/checkpoints_our/clip_citl14_vidLD_2r4500_9f1000/model_epoch_best.pth", map_location='cpu')
    # print(state_dict)
    # 过滤提取
    filtered_state_dict = OrderedDict()
    if 'fc.weight' in state_dict['model']:
        filtered_state_dict['weight'] = state_dict['model']['fc.weight']
    if 'fc.bias' in state_dict['model']:
        filtered_state_dict['bias'] = state_dict['model']['fc.bias']

    # print(filtered_state_dict)
    model.fc.load_state_dict(filtered_state_dict)
    print("model...")
    # *************************************************************************
    
    target_layers = [model.model.visual.transformer.resblocks[20]]
    # print(model)
    # for name, module in model.named_modules():
    #     print(f"Layer Name: {name}, Layer Type: {type(module).__name__}")

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "helicopter.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = None  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()