"""
timm 0.9.10 must be installed to use the get_intermediate_layers method.
    pip install timm==0.9.10
    pip install torch_kmeans

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import itertools
import timm
import torch
import types
import albumentations as A
from torch.nn import functional as F

from PIL import Image
from sklearn.decomposition import PCA
from torch_kmeans import KMeans, CosineSimilarity
from dinov2 import dinov2_vits14_reg
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cmap = plt.get_cmap("tab20")
MEAN = np.array([0.3609696,  0.38405442, 0.4348492])
STD = np.array([0.19669543, 0.20297967, 0.22123419])

options = ['DINOv2-reg']

timm_model_card = {
    "DINOv2": "vit_small_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "CLIP": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "MAE": "vit_base_patch16_224.mae",
    "DeiT-III": "deit3_base_patch16_224.fb_in1k"
}

our_model_card = {
    "DINOv2": "dinov2_small_fine",
    "DINOv2-reg": "dinov2_reg_small_fine",
    "CLIP": "clip_base_fine",
    "MAE": "mae_base_fine",
    "DeiT-III": "deit3_base_fine"
}



# transforms = A.Compose([
#             A.Normalize(mean=list(MEAN), std=list(STD)),
#     ])


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):

    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def viz_feat(feat):

    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()
    
    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

    return res_pred


def plot_feats(image, model_option, ori_feats, fine_feats, ori_labels=None, fine_labels=None):

    ori_feats_map = viz_feat(ori_feats)
    fine_feats_map = viz_feat(fine_feats)

    if ori_labels is not None:
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        ax[0][0].imshow(image)
        ax[0][0].set_title("Input image", fontsize=15)
        ax[0][1].imshow(ori_feats_map)
        ax[0][1].set_title("Original " + model_option, fontsize=15)
        ax[0][2].imshow(fine_feats_map)
        ax[0][2].set_title("Ours", fontsize=15)
        ax[1][1].imshow(ori_labels)
        ax[1][2].imshow(fine_labels)
        for xx in ax:
          for x in xx:
            x.xaxis.set_major_formatter(plt.NullFormatter())
            x.yaxis.set_major_formatter(plt.NullFormatter())
            x.set_xticks([])
            x.set_yticks([])
            x.axis('off')

    else:
        fig, ax = plt.subplots(1, 3, figsize=(30, 8))
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=15)
        ax[1].imshow(ori_feats_map)
        ax[1].set_title("Original " + model_option, fontsize=15)
        ax[2].imshow(fine_feats_map)
        ax[2].set_title("FiT3D", fontsize=15)

        for x in ax:
          x.xaxis.set_major_formatter(plt.NullFormatter())
          x.yaxis.set_major_formatter(plt.NullFormatter())
          x.set_xticks([])
          x.set_yticks([])
          x.axis('off')

    plt.tight_layout()
    plt.savefig("output2.png")
    # plt.close(fig)
    return fig


def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


def process_image(image, stride, transforms):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed['image'])
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    h, w = image_tensor.shape[2:]

    height_int = ((h + stride-1) // stride)*stride
    width_int = ((w+stride-1) // stride)*stride

    image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')

    return image_resized


def kmeans_clustering(feats_map, n_clusters=20):

    B, D, h, w = feats_map.shape
    feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats_map_flattened)
    labels = kmeans_engine.predict(
        feats_map_flattened
        )
    labels = labels.reshape(
        B, h, w
        ).float()
    labels = labels[0].cpu().numpy()

    label_map = cmap(labels / n_clusters)[..., :3]
    label_map = np.uint8(label_map * 255)
    label_map = Image.fromarray(label_map)

    return label_map


def run_demo(original_model, fine_model, image_path, kmeans=20):
    """
    Run the demo for a given model option and image
    model_option: ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
    image_path: path to the image
    kmeans: number of clusters for kmeans. Default is 20. -1 means no kmeans.
    """
    p = original_model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(image_path)
    transforms = A.Compose([
            A.Normalize(mean=list(MEAN), std=list(STD)),
    ])
    image_resized = process_image(image, stride, transforms)
    with torch.no_grad():
        ori_feats = original_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True,
                                    return_class_token=False, norm=True)
        fine_feats = fine_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True,
                                    return_class_token=False, norm=True)

    ori_feats = ori_feats[-1]
    fine_feats = fine_feats[-1]
    if kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, kmeans)
        fine_labels = kmeans_clustering(fine_feats, kmeans)
    else:
        ori_labels = None
        fine_labels = None
    print("image shape: ", image.size)
    print("image_resized shape: ", image_resized.shape)
    print("ori_feats shape: ", ori_feats.shape)
    print("fine_feats shape: ", fine_feats.shape)


    return plot_feats(image, "DINOv2-reg", ori_feats, fine_feats, ori_labels, fine_labels),ori_feats,fine_feats

def Dinov2RegExtractor(original_model, fine_model, image,transforms=None, kmeans=20,only_fine_feats:bool=False):
    """
    Run the demo for a given model option and image
    model_option: ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
    image_path: path to the image
    kmeans: number of clusters for kmeans. Default is 20. -1 means no kmeans.
    """
    p = original_model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image=image.cpu().numpy()
    image_array = (image * 255).astype(np.uint8)
    image_array=image_array.squeeze(0).transpose(1,2,0)
    image = Image.fromarray(image_array)
    fine_feats=None
    ori_feats=None
    if transforms is not None:
        image_resized = process_image(image, stride, transforms)
    else:
        image_resized=image
    with torch.no_grad():
        
        fine_feats = fine_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True,
                                return_class_token=False, norm=True)
        if not only_fine_feats:
            ori_feats = original_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True,
                                    return_class_token=False, norm=True)

    
    fine_feats = fine_feats[-1]
    if not only_fine_feats:
        ori_feats = ori_feats[-1]
    # For semantic segmentation
    # if kmeans != -1:
    #     ori_labels = kmeans_clustering(ori_feats, kmeans)
    #     fine_labels = kmeans_clustering(fine_feats, kmeans)
    # else:
    #     ori_labels = None
    #     fine_labels = None
    # print("image shape: ", image.size)
    # print("image_resized shape: ", image_resized.shape)
    # print("ori_feats shape: ", ori_feats.shape)
    # print("fine_feats shape: ", fine_feats.shape)


    return ori_feats,fine_feats


def LoadDinov2Model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=False,
        ).to(device)
    original_model.get_intermediate_layers = types.MethodType(
        get_intermediate_layers,
        original_model
    )
    fine_model = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine").to(device)
    fine_model.get_intermediate_layers = types.MethodType(
        get_intermediate_layers,
        fine_model
    )
    return original_model,fine_model


def GetDinov2RegFeats(original_model,fine_model,image,transforms=None,kmeans=20):
    ori_feats,fine_feats=Dinov2RegExtractor(original_model,fine_model,image,transforms)
    return fine_feats
