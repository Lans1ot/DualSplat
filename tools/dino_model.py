import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x

def process_image(image, stride, transforms, device):

    image=image.cpu().numpy()
    image_array = (image * 255).astype(np.uint8)
    image_array=image_array.squeeze(0).transpose(1,2,0)
    image = Image.fromarray(image_array)
    
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed['image'])
    image_tensor = image_tensor.permute(2,0,1).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    h, w = image_tensor.shape[2:]

    height_int = ((h + stride-1) // stride)*stride
    width_int = ((w+stride-1) // stride)*stride

    image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')

    return image_resized


class Dino:
    def __init__(self, max_size=952):
        super().__init__()
        self.max_size = max_size
        self._load_model()

    def _load_model(self):
        # fine_model = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine")
        fine_model = torch.hub.load("/home/wangxu/.cache/torch/hub/ywyue_FiT3D_main", "dinov2_reg_small_fine", source="local")
        self.fine_model=fine_model

    def to(self, device):
        self.fine_model = self.fine_model.to(device)
        return self

    def eval(self):
        self.fine_model = self.fine_model.eval()
        return self