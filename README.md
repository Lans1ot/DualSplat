## DualSplat: Robust 3D Gaussian Splatting via Pseudo-Mask Bootstrapping from Reconstruction Failures

---

[[dataset](google pan)](https://drive.google.com/drive/folders/1_CEsQUOfd8yRpmEQshWy8ZxMrVqwc1GD?usp=drive_link) [[pointclouds(google pan)]](https://drive.google.com/drive/folders/1_CEsQUOfd8yRpmEQshWy8ZxMrVqwc1GD?usp=drive_link)

[[dataset(baidu pan)]](https://pan.baidu.com/s/1N5s6LBtEjUmh8BJxbFG6-A?pwd=sknf) [[pointclouds(baidu pan)]](https://pan.baidu.com/s/15t4rU8dGYRRMfF8Mh57z0g?pwd=tyau)

For storage limitaion, if you need full dataset, please download it from baidu pan or use colmap to regenerate it.


**we are working to release the code**

### Environment

Similar as 3DGS and RobustSplat. Please refer to `environment.yaml`

> `environment.yaml` has not been tested

### For training

The command-line parameters are roughly the same as the original 3DGS.

#### Data preparation

We follow the instruction of [DeSplat](https://github.com/AaltoML/desplat) to generate the colmap version of NeRF on-the-go and RobustNeRF. We have modified the dataset_readers to split the training and testing views. There should be a "train_list.txt" and "test_list.txt" containing the name of views at the same folder with 'sparse' folder in colmap datasets. You can refer the code to generate it or download ours prepared datasets. 

After preparing the datasets, the sam2 instance masks have to be generated. You can install the sam2 in submodules folder and run:

```bash
# for download sam2 checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P CHECKPOINTS_PATH

# for non-label mask generation
python tools/seg_all_instances.py --model_checkpoint CHECKPOINTS_PATH/sam2_hiera_large.pt --image_dir DATASET_PATH

# DATASET_PATH is the folders containg the "images, spares, train_list.txt, test_list.txt"
```

You can modified the parameters of sam2 in the code and to set input\output folders position. It is recommended input downsampled images into sam2 to speed up.

For **depths generation and  regulariztion**, you can follow the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to prepare.

```txt
DATASET_PATH/
├── images/
├── spares/
├── sam_masks/      # sam2 generated
├── pseudo_masks/   # mask filter generated, could be soft link but need to be created manually
├── depths/         # depth anything generated
├── train_list.txt  # train views
└── test_list.txt   # test views
```

For relesed datasets which are mention above, `pseudo_masks` is named as `masks_npy`

#### Stage-1

run 
```bash
python ./prepare.py -s DATASET_PATH -m MODEL_PATH -r X --densification_interval 100 --densify_from_iter 500 --densify_until_iter 15000 --iteration 10000 --origin_masks SAM2_MASKS_FOLDER --eval
# the densification setting should be the same as origin 3DGS

# SAM2_MASKS_FOLDER should be the folder name containing the masks under DATASET_PATH folder, default setting is 'sam_masks'

python ./render.py -m MODEL_PATH

```

It is recommended input **suitable** images. In paper, we compare 8x downsample with other model but in Stage-1, we use 4x downsample. Although we haven't yet explored the effects of different resolutions

#### Mask Filter

After training and rendering in Stage-1, run:

```bash
python ./tools/dino.py --root MODEL_PATH/train/ours_10000
# generating feature similarity map

# MODEL_PATH = ROOT/MODEL_NAME
python ./tools/extract_mask.py --model_name MODEL_NAME --root ROOT --masks DATASET_PATH/MASK_PATH --output_folder OUTPUT_FOLDER_NAME
```

After generating the pseudo masks, it will generate two folder. One contain the visualized masks and the other contain npy masks. Copy or create soft link of npy folder to the DATASET_PATH.

The the pseudo masks can be put into use anywhere.

#### Stage-2

```bash
python ./train.py -s DATASET_PATH -m MODEL_PATH --filtered_masks PSEUDO_FOLDER_NAME --eval -d DEPTHS_FOLDER_NAME -r X

python ./render.py -m MODEL_PATH

python ./metrics.py -m MODEL_PATH
```

---

Bou can try ours advanced work with voxel regularization--a simple way to remove the floaters behind the masks. The `voxel_size` should be manually set, it is recommend to set as big as the parts of static objects or background.

```bash 
python ./train_v3_w_voxel.py --voxel_size X ...(other parameters)
```

Besides, we found these settting below may imporve the quality

```txt
    densification_interval = 1000
    densify_from_iter = 3_000
    densify_until_iter = 25_000
    densify_grad_threshold = 0.00012
```

### Additional 3-times results

#### PSNR

| Corner    | Fountain    | Mountain    | Patio  | Patio_high  | Spot    | avg.    |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 26.336      | 21.143      | 21.819      | 21.568      | 23.273      | 25.679      | 23.303      |
| 25.948      | 21.225      | 21.834      | 21.471      | 23.281      | 25.530      | 23.215      |
| 26.672      | 21.254      | 21.964      | 21.857      | 23.310      | 25.722      | 23.463      |

#### SSIM

| Corner    | Fountain    | Mountain    | Patio  | Patio_high  | Spot    | avg.    |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 0.894       | 0.704       | 0.747       | 0.828       | 0.840       | 0.905       | 0.820       |
| 0.888       | 0.705       | 0.751       | 0.828       | 0.841       | 0.904       | 0.820       |
| 0.896       | 0.705       | 0.757       | 0.829       | 0.840       | 0.907       | 0.822       |
#### LPIPS

| Corner    | Fountain    | Mountain    | Patio  | Patio_high  | Spot    | avg.    |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 0.053       | 0.132       | 0.128       | 0.079       | 0.091       | 0.047       | 0.088       |
| 0.059       | 0.132       | 0.126       | 0.081       | 0.091       | 0.048       | 0.089       |
| 0.051       | 0.132       | 0.123       | 0.080       | 0.092       | 0.046       | 0.087       |

---

This repo mainly followed [RobustSplat](https://github.com/fcyycf/RobustSplat) and [DroneSplat](https://github.com/BITyia/DroneSplat), many thanks to their awesome work!