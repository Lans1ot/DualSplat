## DualSplat: Robust 3D Gaussian Splatting via Pseudo-Mask Bootstrapping from Reconstruction Failures

---

[[dataset]]() [[pointclouds]]() [[pseudo masks]]()

**we are working to release the code**

### Environment

Similar as 3DGS and RobustSplat. Please refer to `environment.yaml`

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

For depths regulariztion, you can follow the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to prepare.

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
| 26.33550453 | 21.14323044 | 21.81937599 | 21.85120773 | 23.27262878 | 25.6794548  | 23.35023371 |
| 25.94782066 | 21.22486687 | 21.83425331 | 21.9464016  | 23.28129005 | 25.52963638 | 23.29404481 |
| 26.67227745 | 21.25419617 | 21.96355629 | 21.78841019 | 23.30951309 | 25.72206497 | 23.45166969 |

#### SSIM

| Corner    | Fountain    | Mountain    | Patio  | Patio_high  | Spot    | avg.    |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 0.893862069 | 0.703564107 | 0.746993601 | 0.861568987 | 0.840292931 | 0.905113041 | 0.825232456 |
| 0.887990594 | 0.705296397 | 0.750601709 | 0.863253355 | 0.840882361 | 0.904368103 | 0.825398753 |
| 0.896334827 | 0.704749584 | 0.75665921  | 0.862533212 | 0.839657009 | 0.906562209 | 0.827749342 |

#### LPIPS

| Corner    | Fountain    | Mountain    | Patio  | Patio_high  | Spot    | avg.    |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 0.052855324 | 0.131507501 | 0.127776876 | 0.047327757 | 0.090814091 | 0.046667572 | 0.082824854 |
| 0.059263788 | 0.132290617 | 0.125829563 | 0.04729116  | 0.091198772 | 0.047763299 | 0.083939533 |
| 0.050970484 | 0.132039934 | 0.123378254 | 0.046817847 | 0.091623649 | 0.046172939 | 0.081833851 |

---

This repo mainly followed [RobustSplat](https://github.com/fcyycf/RobustSplat) and [DroneSplat](https://github.com/BITyia/DroneSplat), many thanks to their awesome work!