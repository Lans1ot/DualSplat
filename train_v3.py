#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.mask_utils import MLPModel, calculate_residual_mask, interpolation
from utils.mask_utils import MLPModel_2
#from utils.mask_utils import DINOFinetune_FeatureExtractor as DINOFeatureExtractor
from utils.mask_utils import DINOFeatureExtractor
# from utils.mask_utils import DINOv3FeatureExtractor as DINOFeatureExtractor
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    
    gaussiansN = 1
    GsDict = {}
    RenderDict = {}

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    for i in range(gaussiansN):
        GsDict[f"gaussians{i}"] = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        GsDict[f"scene{i}"] = Scene(dataset, GsDict[f"gaussians{i}"], resolution_scales=[1., 4.])
        GsDict[f"gaussians{i}"].training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            GsDict[f"gaussians{i}"].restore(model_params, opt)

        GsDict[f"viewpoint_stack{i}"] = GsDict[f"scene{i}"].getTrainCameras().copy()
        GsDict[f"coarse_stack{i}"] = GsDict[f"scene{i}"].getTrainCameras(4).copy()
        GsDict[f"viewpoint_indices{i}"] = list(range(len(GsDict[f"viewpoint_stack{i}"])))
        GsDict[f"historical_hist{i}"] = torch.zeros((10000)).cuda()
        GsDict[f"mlp_model{i}"] = MLPModel_2().to(device="cuda")
        GsDict[f"mlp_optimizer{i}"] = optim.Adam(GsDict[f"mlp_model{i}"].parameters(), lr=1e-3)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Prepare for mask estimation
    if not opt.disable_mask:
        feature_extractor = DINOFeatureExtractor().cuda()
        features_fine, features_coarse = {}, {}
        for cam in tqdm(GsDict[f"scene{0}"].getTrainCameras(), desc=f"DINOv2 GT Feature Extraction"):
            features_fine[cam.image_name] = feature_extractor(cam.original_image.cuda(), opt.upper_feat_res).cpu().detach()
            features_coarse[cam.image_name] = feature_extractor(cam.original_image.cuda(), opt.lower_feat_res).cpu().detach()

        #mlp_model = MLPModel().to(device="cuda")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        for i in range(gaussiansN):
            GsDict[f"gaussians{i}"].update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                GsDict[f"gaussians{i}"].oneupSHdegree()

            # Pick a random Camera
            if not GsDict[f"viewpoint_stack{i}"]:
                GsDict[f"viewpoint_stack{i}"] = GsDict[f"scene{i}"].getTrainCameras().copy()
                GsDict[f"coarse_stack{i}"] = GsDict[f"scene{i}"].getTrainCameras(4).copy()
                GsDict[f"viewpoint_indices{i}"] = list(range(len(GsDict[f"viewpoint_stack{i}"])))
            rand_idx = randint(0, len(GsDict[f"viewpoint_indices{i}"]) - 1)
            RenderDict[f"viewpoint_cam{i}"] = GsDict[f"viewpoint_stack{i}"].pop(rand_idx)
            RenderDict[f"coarse_cam{i}"] = GsDict[f"coarse_stack{i}"].pop(rand_idx)
            GsDict[f"viewpoint_indices{i}"].pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        for i in range(gaussiansN):
            RenderDict[f"render_pkg{i}"] = render(RenderDict[f"viewpoint_cam{i}"], GsDict[f"gaussians{i}"], pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            RenderDict[f"image{i}"], RenderDict[f"viewspace_point_tensor{i}"], RenderDict[f"visibility_filter{i}"], RenderDict[f"radii{i}"] = RenderDict[f"render_pkg{i}"]["render"], RenderDict[f"render_pkg{i}"]["viewspace_points"], RenderDict[f"render_pkg{i}"]["visibility_filter"], RenderDict[f"render_pkg{i}"]["radii"]
            # coarse scale rendering
            if not opt.disable_mask and iteration < opt.bootstrap_iter:
                RenderDict[f"coarse_image{i}"] = render(RenderDict[f"coarse_cam{i}"], GsDict[f"gaussians{i}"], pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                RenderDict[f"coarse_gt{i}"] = RenderDict[f"coarse_cam{i}"].original_image.cuda()

            if RenderDict[f"viewpoint_cam{i}"].alpha_mask is not None:
                alpha_mask = RenderDict[f"viewpoint_cam{i}"].alpha_mask.cuda()
                RenderDict[f"image{i}"] *= alpha_mask

            RenderDict[f"gt_image{i}"] = RenderDict[f"viewpoint_cam{i}"].original_image.cuda()

            image_name = RenderDict[f"viewpoint_cam{i}"].image_name
            image_name = image_name if image_name.find(".") != -1 else image_name + ".png"
            
            RenderDict[f"filtered_mask{i}"] = RenderDict[f"viewpoint_cam{i}"].get_filtered_mask(os.path.join(args.source_path, args.filtered_masks, image_name)).cuda().float()
            RenderDict[f"filtered_mask{i}"] = interpolation(RenderDict[f"filtered_mask{i}"].unsqueeze(0), RenderDict[f"image{i}"].shape[1], RenderDict[f"image{i}"].shape[2]).float()
            RenderDict[f"loss_mask{i}"] = None

        for i in range(gaussiansN):
            # MLP eval for masked loss calculation
            if opt.disable_mask or iteration < opt.mask_beginning:
                RenderDict[f"Ll1{i}"] = l1_loss(RenderDict[f"image{i}"] * RenderDict[f"filtered_mask{i}"], RenderDict[f"gt_image{i}"] * RenderDict[f"filtered_mask{i}"])
                # if FUSED_SSIM_AVAILABLE:
                #     ssim_value = fused_ssim((RenderDict[f"image{i}"] * RenderDict[f"filtered_mask{i}"]).unsqueeze(0), (RenderDict[f"gt_image{i}"] * RenderDict[f"filtered_mask{i}"]).unsqueeze(0))
                # else:
                    # ssim_value = ssim(RenderDict[f"image{i}"] * RenderDict[f"filtered_mask{i}"], RenderDict[f"gt_image{i}"] * RenderDict[f"filtered_mask{i}"])
                RenderDict[f"ssim_value{i}"] = ssim(RenderDict[f"image{i}"] * RenderDict[f"filtered_mask{i}"], RenderDict[f"gt_image{i}"] * RenderDict[f"filtered_mask{i}"])
                RenderDict[f"loss{i}"] = (1.0 - opt.lambda_dssim) * RenderDict[f"Ll1{i}"] + opt.lambda_dssim * (1.0 - RenderDict[f"ssim_value{i}"])

            else:
                mono_invdepth = RenderDict[f"viewpoint_cam{i}"].invdepthmap.cuda()
                invDepth = RenderDict[f"render_pkg{i}"]["depth"]
                depth_residual = mono_invdepth.detach() - invDepth.detach()

                GsDict[f"mlp_model{i}"].eval()
                upsample_feature = interpolation(features_fine[RenderDict[f"viewpoint_cam{i}"].image_name], RenderDict[f"image{i}"].shape[1], RenderDict[f"image{i}"].shape[2])
                RenderDict[f"mask{i}"] = GsDict[f"mlp_model{i}"](upsample_feature, depth_residual)

                RenderDict[f"loss_mask{i}"] = RenderDict[f"mask{i}"].clone().detach() > 0.2
                # loss_mask = mask.clone().detach()
                RenderDict[f"loss_mask{i}"] = -F.max_pool2d(-(RenderDict[f"loss_mask{i}"].float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)
                
                image_ = RenderDict[f"image{i}"] * RenderDict[f"loss_mask{i}"] + RenderDict[f"image{i}"].detach() * (1 - RenderDict[f"loss_mask{i}"])

                # Ll1 = (loss_mask * torch.abs((image - gt_image))).mean()
                # Lssim = (1.0 - ssim((loss_mask * image), (loss_mask * gt_image), size_average=False)).mean()
                RenderDict[f"Ll1{i}"] = torch.abs((image_ - RenderDict[f"gt_image{i}"])).mean()
                RenderDict[f"Lssim{i}"] = (1.0 - ssim(image_, RenderDict[f"gt_image{i}"], size_average=False)).mean()
                RenderDict[f"loss{i}"] = (1.0 - opt.lambda_dssim) * RenderDict[f"Ll1{i}"] + opt.lambda_dssim * RenderDict[f"Lssim{i}"]

                # import torchvision.utils as tutils
                # output_path = os.path.join("./try", args.model_path.split('/')[-1])
                # os.makedirs(output_path, exist_ok=True)
                # tutils.save_image(RenderDict[f"loss_mask{i}"].float(), os.path.join(output_path, f"{image_name}"))

        # Depth regularization
        for i in range(gaussiansN):
            if depth_l1_weight(iteration) > 0 and RenderDict[f"viewpoint_cam{i}"].depth_reliable:
                invDepth = RenderDict[f"render_pkg{i}"]["depth"]
                mono_invdepth = RenderDict[f"viewpoint_cam{i}"].invdepthmap.cuda()
                depth_mask = RenderDict[f"viewpoint_cam{i}"].depth_mask.cuda()
                
                mask_ = RenderDict[f"loss_mask{i}"].detach() if RenderDict[f"loss_mask{i}"] is not None else RenderDict[f"filtered_mask{i}"]

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask * mask_).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                RenderDict[f"loss{i}"] += Ll1depth
                RenderDict[f"Ll1depth{i}"] = Ll1depth.item()
            else:
                RenderDict[f"Ll1depth{i}"] = 0

        for i in range(gaussiansN):
            RenderDict[f"loss{i}"].backward()

            # MLP training
        reset_start = iteration // opt.opacity_reset_interval * opt.opacity_reset_interval
        reset_end = reset_start + 300
        if not opt.disable_mask and iteration >= opt.mask_beginning and (not((iteration>reset_start) and (iteration<reset_end) and (iteration>=opt.reset_iter))):
            for i in range(gaussiansN):
                GsDict[f"mlp_model{i}"].train()            
                if iteration < opt.bootstrap_iter:
                    gt_feature = features_coarse[RenderDict[f"viewpoint_cam{i}"].image_name].cuda()
                    render_feature = feature_extractor(RenderDict[f"image{i}"].detach(),opt.lower_feat_res) 
                    lower_mask, upper_mask, GsDict[f"historical_hist{i}"] = calculate_residual_mask(RenderDict[f"coarse_gt{i}"], RenderDict[f"coarse_image{i}"], GsDict[f"historical_hist{i}"])
                else:
                    gt_feature = features_fine[RenderDict[f"viewpoint_cam{i}"].image_name].cuda()
                    render_feature = feature_extractor(RenderDict[f"image{i}"].detach(),opt.upper_feat_res) 
                    lower_mask, upper_mask, RenderDict[f"historical_hist{i}"] = calculate_residual_mask(RenderDict[f"gt_image{i}"], RenderDict[f"image{i}"], GsDict[f"historical_hist{i}"])
                lower_mask = interpolation(lower_mask, RenderDict[f"image{i}"].shape[1], RenderDict[f"image{i}"].shape[2])
                upper_mask = interpolation(upper_mask, RenderDict[f"image{i}"].shape[1], RenderDict[f"image{i}"].shape[2])

                cosine = (1.-F.cosine_similarity(gt_feature, render_feature, dim=0).unsqueeze(0).sub(0.5).div(0.5)).clip(0.,1.)
                cosine = 1. - interpolation(cosine, RenderDict[f"image{i}"].shape[1], RenderDict[f"image{i}"].shape[2])

                reg_loss = 0.5 * GsDict[f"mlp_model{i}"].get_regularizer()
                #reg_loss += 2.0 * ((1-mask) * math.exp(-iteration / opt.beta_reg)).mean()
                
                prior_loss = (torch.abs(RenderDict[f"filtered_mask{i}"] - RenderDict[f"mask{i}"])).mean() * math.exp(-iteration / 10000)

                residual_loss = GsDict[f"mlp_model{i}"].get_residual_loss(RenderDict[f"mask{i}"].flatten(), lower_mask.flatten(), upper_mask.flatten())
                cosine_loss = torch.abs(RenderDict[f"mask{i}"] - cosine).mean()

                robustness_loss = 0.5 * cosine_loss + 0.5 * residual_loss
                
                if(iteration <= opt.densify_from_iter):
                    robustness_loss = robustness_loss * math.exp((iteration - opt.densify_from_iter) / 10000)

                mask_loss = robustness_loss + reg_loss + prior_loss
                mask_loss.backward()


                # import torchvision.utils as tutils
                # output_path = os.path.join("./try", args.model_path.split('/')[-1])
                # os.makedirs(output_path, exist_ok=True)
                # tutils.save_image(loss_mask.float(), os.path.join(output_path, f"{image_name}.png"))
                GsDict[f"mlp_optimizer{i}"].step()
                GsDict[f"mlp_optimizer{i}"].zero_grad(set_to_none=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * RenderDict[f"loss{0}"].item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * RenderDict[f"Ll1depth{0}"] + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, RenderDict[f"Ll1{0}"], RenderDict[f"loss{0}"], l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, GsDict[f"scene{0}"], render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                for i in range(gaussiansN):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    GsDict[f"scene{i}"].save(iteration, i)

            for i in range(gaussiansN):
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    GsDict[f"gaussians{i}"].max_radii2D[RenderDict[f"visibility_filter{i}"]] = torch.max(GsDict[f"gaussians{i}"].max_radii2D[RenderDict[f"visibility_filter{i}"]], RenderDict[f"radii{i}"][RenderDict[f"visibility_filter{i}"]])
                    GsDict[f"gaussians{i}"].add_densification_stats(RenderDict[f"viewspace_point_tensor{i}"], RenderDict[f"visibility_filter{i}"])

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        GsDict[f"gaussians{i}"].densify_and_prune(opt.densify_grad_threshold, 0.005, GsDict[f"scene{i}"].cameras_extent, None, RenderDict[f"radii{i}"])
                    
                    if iteration >= opt.reset_iter and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        GsDict[f"gaussians{i}"].reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    GsDict[f"gaussians{i}"].exposure_optimizer.step()
                    GsDict[f"gaussians{i}"].exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = RenderDict[f"radii{i}"] > 0
                        GsDict[f"gaussians{i}"].optimizer.step(visible, RenderDict[f"radii{i}"].shape[0])
                        GsDict[f"gaussians{i}"].optimizer.zero_grad(set_to_none = True)
                    else:
                        GsDict[f"gaussians{i}"].optimizer.step()
                        GsDict[f"gaussians{i}"].optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((GsDict[f"gaussians{i}"].capture(), iteration), GsDict[f"scene{i}"].model_path + f"/chkpnt_{i}_" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    print("train mlp v2 with train.py v3")
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")