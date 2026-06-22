import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from models import get_model
from collections import OrderedDict

def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    model = models.resnet34(pretrained=True)
    target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]
    
    # ************************************************************************
    # model = get_model("Imagenet:resnet50")
    
    # state_dict = torch.load("/home/mwp/UniversalFakeDetect-main/checkpoints/imagenet_resnet50_vidLD_2r4500_9f1000_SFI/model_epoch_best.pth", map_location='cpu')
    # # print(state_dict)
    # # filter
    # filtered_state_dict = OrderedDict()
    # if 'fc.weight' in state_dict['model']:
    #     filtered_state_dict['weight'] = state_dict['model']['fc.weight']
    # if 'fc.bias' in state_dict['model']:
    #     filtered_state_dict['bias'] = state_dict['model']['fc.bias']

    # print(filtered_state_dict)
    # model.fc.load_state_dict(filtered_state_dict)
    # print("model...")
    # *************************************************************************
    
    target_layers = [model.layer4]
    # for name, module in model.named_modules():
    #     print(f"Layer Name: {name}, Layer Type: {type(module).__name__}")
    # print(model)
    
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "2.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # crop
    img = center_crop_img(img, 224)
    
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 404  # tabby, tabby cat
    
    # print(input_tensor.shape)
    # print(input_tensor)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()