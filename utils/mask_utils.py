import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

import timm
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore", message=".*xFormers.*")

class DINOFinetune_FeatureExtractor(nn.Module):
    # not used
    def __init__(self, max_size=952):
        super(DINOFinetune_FeatureExtractor, self).__init__()
        self.max_size = max_size
        self._load_model()
    
    def _load_model(self):
        fine_model = torch.hub.load("/home/wangxu/.cache/torch/hub/ywyue_FiT3D_main", "dinov2_reg_small_fine", source="local")
        self.fine_model=fine_model

    def forward(self, image, feature_size=50):
        with torch.no_grad():
            feature_height = feature_width = feature_size
            image = F.interpolate(image.unsqueeze(0), size=(feature_height * 14, feature_width * 14), mode='bilinear', align_corners=False)

            dino_features = self.fine_model.get_intermediate_layers(image, n=[11], reshape=True, norm=True)[-1]

        return dino_features.squeeze()

class DINOFeatureExtractor(nn.Module):
    def __init__(self):
        super(DINOFeatureExtractor, self).__init__()
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # self.dinov2_model = torch.hub.load("/home/wangxu/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vits14_reg", source="local")

        self.dinov2_model = self.dinov2_model.cuda()
        self.dinov2_model.eval()

    def forward(self, gt_image, feature_size=50):
        with torch.no_grad():
            feature_height = feature_width = feature_size
            gt_img = F.interpolate(gt_image.unsqueeze(0), size=(feature_height * 14, feature_width * 14), mode='bilinear', align_corners=False)
            gt_embeddings = self.dinov2_model.forward_features(gt_img)
            dino_features = gt_embeddings["x_norm_patchtokens"].reshape(1, feature_height, feature_width, -1).permute(0, 3, 1, 2)
        return dino_features.squeeze()


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def get_regularizer(self):
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(abs(self.mlp[2].weight.data))

    def get_residual_loss(self, mask, lower_mask, upper_mask):
        return torch.mean(nn.ReLU()(mask - upper_mask) + nn.ReLU()(lower_mask - mask))

    def forward(self, features):
        x = features.reshape(features.shape[0], -1).permute(1, 0)
        x = self.mlp(x)
        x = x.reshape(features.shape[1], features.shape[2], -1).permute(2, 0, 1)
        return x

class MLPModel_2(nn.Module):
    def __init__(self):
        super(MLPModel_2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(9, 1),
            nn.Sigmoid(),
        )
    
    def get_regularizer(self):
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(abs(self.mlp[2].weight.data)) + torch.max(abs(self.mlp_2[0].weight.data))

    def get_residual_loss(self, mask, lower_mask, upper_mask):
        return torch.mean(nn.ReLU()(mask - upper_mask) + nn.ReLU()(lower_mask - mask))

    def forward(self, features, depth_residual):
        x = features.reshape(features.shape[0], -1).permute(1, 0)
        x = self.mlp(x)

        depth_residual = depth_residual.reshape(1, -1).permute(1, 0)

        x = self.mlp_2(torch.concat([x, depth_residual], dim=-1))
        x = x.reshape(features.shape[1], features.shape[2], -1).permute(2, 0, 1)
        return x

def generate_mask(residual, threshold):
    inlier_pixel = (residual < threshold).float().unsqueeze(0).unsqueeze(0)
    window = torch.ones((1, 1, 3, 3), dtype=torch.float) / (3*3)
    if residual.is_cuda:
        window = window.cuda(residual.get_device())
    inlier_neighbors = F.conv2d(inlier_pixel, window, padding=1, groups=1)
    mask = (((inlier_neighbors > 0.5).float() + inlier_pixel) > 1e-3).float()
    return mask

def calculate_residual_mask(gt, render, cum_hist, lower_bound=0.6, upper_bound=0.8):
    residual_img = torch.abs(gt - render).clone().detach()
    residual = torch.mean(residual_img, dim=0)
    error_hist = torch.histogram(
        residual.cpu(),
        bins=10000,
        range=(0.0, 1.0)
    )[0].cuda()

    cum_hist = 0.95 * cum_hist + error_hist
    cum_error = torch.cumsum(cum_hist, dim=0)
    lower_error = torch.sum(cum_hist) * lower_bound
    upper_error = torch.sum(cum_hist) * upper_bound
    lower_threshold = torch.linspace(0, 1, 10001)[torch.where(cum_error >= lower_error)[0][0]]
    upper_threshold = torch.linspace(0, 1, 10001)[torch.where(cum_error >= upper_error)[0][0]]
    lower_mask = generate_mask(residual, lower_threshold)
    upper_mask = generate_mask(residual, upper_threshold)

    return lower_mask.squeeze(0), upper_mask.squeeze(0), cum_hist

def interpolation(tensor, height, width, type='bilinear'):
    return F.interpolate(tensor.cuda().unsqueeze(0), size=(height, width), mode=type).squeeze(0)

class ResizeToMultiple(nn.Module):
    def __init__(self, multiple: int = 16,
                 interpolation: InterpolationMode = InterpolationMode.BICUBIC,
                 antialias: bool = True):
        super().__init__()
        self.multiple = multiple
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W]
        h, w = x.shape[-2], x.shape[-1]
        m = self.multiple
        new_h = ((h + m - 1) // m) * m
        new_w = ((w + m - 1) // m) * m
        if new_h == h and new_w == w:
            return x
        return TF.resize(
            x, size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,
            features_only=True,
        ).to(device).eval()

        # patch size (vit_small_patch16 -> 16)
        ps = getattr(getattr(self.model, "patch_embed", None), "patch_size", 16)
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
        self.patch_size = int(ps)

        # transform: (dtype/scale) -> resize to multiple-of-16 -> normalize
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),  # uint8 -> float in [0,1]；float则保持
            ResizeToMultiple(multiple=self.patch_size,
                             interpolation=InterpolationMode.BICUBIC,
                             antialias=True),
            transforms.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250],
            ),
        ])

    @torch.no_grad()
    def forward(self, image: torch.Tensor, feature_size = 50) -> torch.Tensor:
        """
        image: [3,H,W] or [B,3,H,W]
        return: [B, C, H//16, W//16]
        """
        if image.dim() == 3:
            image = self.transform(image).unsqueeze(0)  # [1,3,H',W']
        else:
            image = torch.stack([self.transform(im) for im in image], dim=0)  # [B,3,H',W']

        # feature_height = feature_width = feature_size
        # gt_img = F.interpolate(image, size=(feature_height * 14, feature_width * 14), mode='bilinear', align_corners=False)

        feature = self.model(image)[-1]
        return feature.squeeze()
