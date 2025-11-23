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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
import open3d as o3d
import numpy as np
from utils.mask_utils import MLPModel_2 as MLPModel, calculate_residual_mask, interpolation
from utils.mask_utils import DINOFeatureExtractor
import torch.optim as optim
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

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=[1., 4.])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    GsDict = {}
    gaussiansN = args.gaussiansN

    for i in range(gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        elif i > 0:
            GsDict[f"gs{i}"] = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            GsDict[f"gs{i}"].create_from_pcd(scene.scene_info.point_cloud, scene.scene_info.train_cameras, scene.cameras_extent)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    for i in range(gaussiansN):
        GsDict[f"viewpoint_stack{i}"] = scene.getTrainCameras().copy()
        GsDict[f"coarse_stack{i}"] = scene.getTrainCameras(4.).copy()
        GsDict[f"viewpoint_indices{i}"] = list(range(len(GsDict[f"viewpoint_stack{i}"])))
        historical_hist = torch.zeros((10000)).cuda()
        GsDict[f"historical_hist{i}"] = historical_hist

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    feature_extractor = DINOFeatureExtractor().cuda()
    features_fine, features_coarse = {}, {}
    for cam in tqdm(scene.getTrainCameras(), desc=f"DINOv2 GT Feature Extraction"):
        features_fine[cam.image_name] = feature_extractor(cam.original_image.cuda(), opt.feat_res).cpu()
        features_coarse[cam.image_name] = feature_extractor(cam.original_image.cuda(), opt.lower_feat_res).cpu()

    mlp_model = MLPModel().to(device="cuda")
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        for i in range(gaussiansN):
            GsDict[f"gs{i}"].update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                GsDict[f"gs{i}"].oneupSHdegree()

        pseudo_stack_co = None
        RenderDict = {}
        LossDict = {}
        # Pick a random Camera
        for i in range(gaussiansN):
            if not GsDict[f"viewpoint_stack{i}"]:
                GsDict[f"viewpoint_stack{i}"] = scene.getTrainCameras().copy()
                GsDict[f"coarse_stack{i}"] = scene.getTrainCameras(4.).copy()
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
            RenderDict[f"render_pkg_gs{i}"] = render(RenderDict[f"viewpoint_cam{i}"], GsDict[f'gs{i}'], pipe, bg)
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]
            RenderDict[f"depth_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["depth"]
            gt_image = RenderDict[f"viewpoint_cam{i}"].original_image.cuda()
            image_name = RenderDict[f"viewpoint_cam{i}"].image_name
            image_name = image_name if image_name.find(".") != -1 else image_name + ".png"
            
            filtered_mask = RenderDict[f"viewpoint_cam{i}"].get_filtered_mask(os.path.join(args.source_path, args.filtered_masks, image_name)).cuda().float()
            filtered_mask = interpolation(filtered_mask.unsqueeze(0), gt_image.shape[1], gt_image.shape[2]).float()
            RenderDict[f"filtered_mask{i}"] = filtered_mask

            if iteration < opt.bootstrap_iter:
                RenderDict[f"coarse_image{i}"] = render(RenderDict[f"coarse_cam{i}"], gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]

        # Loss
        mlp_model.eval()
        for i in reversed(range(args.gaussiansN)): # make Ll1 is the l1 loss of gs_0
            mono_invdepth = RenderDict[f"viewpoint_cam{i}"].invdepthmap.cuda()
            gt_image = RenderDict[f"viewpoint_cam{i}"].original_image.cuda()

            upsample_feature = interpolation(features_fine[RenderDict[f"viewpoint_cam{i}"].image_name], gt_image.shape[1], gt_image.shape[2])
            image = RenderDict[f"image_gs{i}"]
            invDepth = RenderDict[f"depth_gs{i}"]
            depth_residual = mono_invdepth.detach() - invDepth.detach()
            mask = mlp_model(upsample_feature, depth_residual)

            loss_mask = mask.clone().detach() > 0.2
            loss_mask = -F.max_pool2d(-(loss_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)
            RenderDict[f"mask{i}"] = mask
            RenderDict[f"loss_mask{i}"] = loss_mask

            Ll1 = (loss_mask * torch.abs((image - gt_image))).mean()
            ssim_value = (1.0 - ssim(loss_mask * image, loss_mask * gt_image, size_average=False)).mean()
            LossDict[f"loss_gs{i}"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_value

        # Depth regularization
        if depth_l1_weight(iteration) > 0:
            for i in range(args.gaussiansN):
                if RenderDict[f"viewpoint_cam{i}"].depth_reliable:
                    invDepth = RenderDict[f"depth_gs{i}"]
                    # mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    mono_invdepth = RenderDict[f"viewpoint_cam{i}"].invdepthmap.cuda()
                    depth_mask = RenderDict[f"viewpoint_cam{i}"].depth_mask.cuda()

                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask * RenderDict[f"loss_mask{i}"]).mean()
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                    LossDict[f"depth_loss_gs{i}"] = Ll1depth.item()
                    LossDict[f"loss_gs{i}"] += Ll1depth

            Ll1depth = LossDict[f"depth_loss_gs{0}"]
        else:
            Ll1depth = 0
       
        if args.not_only_rgb and args.coreg:
            if iteration % args.sample_pseudo_interval == 0 and iteration <= args.end_sample_pseudo and iteration >= args.start_sample_pseudo:
                if not pseudo_stack_co:
                    pseudo_stack_co = scene.getTrainCameras().copy()
                pseudo_cam_co = pseudo_stack_co.pop(randint(0, len(pseudo_stack_co) - 1))

                for i in range(args.gaussiansN):
                    RenderDict[f"render_pkg_pseudo_co_gs{i}"] = render(pseudo_cam_co, GsDict[f'gs{i}'], pipe, bg)
                    RenderDict[f"image_pseudo_co_gs{i}"] = RenderDict[f"render_pkg_pseudo_co_gs{i}"]["render"]
                    # import torchvision.utils as tutils
                    # tutils.save_image(RenderDict[f"image_pseudo_co_gs{i}"], f"./temp/{iteration}_{i}.png")
                    # RenderDict[f"depth_pseudo_co_gs{i}"] = RenderDict[f"render_pkg_pseudo_co_gs{i}"]["depth"]
                # co-reg, co photometric
                for i in range(args.gaussiansN):
                    for j in range(args.gaussiansN):
                        if i != j:
                            pseudp_gt_image = RenderDict[f"image_pseudo_co_gs{j}"].clone().detach()
                            pseudo_Ll1 = l1_loss(RenderDict[f"image_pseudo_co_gs{i}"], pseudp_gt_image)
                            # pseudo_ssim_value = ssim(RenderDict[f"image_pseudo_co_gs{i}"], pseudp_gt_image)
                            # LossDict[f"loss_gs{i}"] += (1.0 - opt.lambda_dssim) * pseudo_Ll1 + opt.lambda_dssim * (1.0 - pseudo_ssim_value) / (args.gaussiansN - 1)
                            LossDict[f"loss_gs{i}"] += pseudo_Ll1

        loss = LossDict["loss_gs0"]
        for i in range(args.gaussiansN):
            LossDict[f"loss_gs{i}"].backward()
        
        # MLP Training
        reset_start = iteration // opt.opacity_reset_interval * opt.opacity_reset_interval
        reset_end = reset_start + 300
        mask_losses = 0
        if iteration >= opt.mask_beginning and (not((iteration>reset_start) and (iteration<reset_end) and (iteration>=opt.reset_iter))):
            mlp_model.train()            

            for i in range(gaussiansN):
                mask = RenderDict[f"mask{i}"]
                image = RenderDict[f"image_gs{i}"]
                gt_image = RenderDict[f"viewpoint_cam{i}"].original_image.cuda()
                filtered_mask = RenderDict[f"filtered_mask{i}"]

                if iteration < opt.bootstrap_iter:
                    coarse_gt = RenderDict[f"coarse_cam{i}"].original_image.cuda()
                    coarse_image = RenderDict[f"coarse_image{i}"]
                    gt_feature = features_coarse[RenderDict[f"viewpoint_cam{i}"].image_name].cuda()
                    render_feature = feature_extractor(image.detach(),opt.lower_feat_res) 
                    lower_mask, upper_mask, GsDict[f"historical_hist{i}"] = calculate_residual_mask(coarse_gt, coarse_image, GsDict[f"historical_hist{i}"])
                else:
                    gt_feature = features_fine[RenderDict[f"viewpoint_cam{i}"].image_name].cuda()
                    render_feature = feature_extractor(image.detach(), opt.feat_res) 
                    lower_mask, upper_mask, GsDict[f"historical_hist{i}"] = calculate_residual_mask(gt_image, image, GsDict[f"historical_hist{i}"])
                lower_mask = interpolation(lower_mask, image.shape[1], image.shape[2])
                upper_mask = interpolation(upper_mask, image.shape[1], image.shape[2])

                cosine = (1.-F.cosine_similarity(gt_feature, render_feature, dim=0).unsqueeze(0).sub(0.5).div(0.5)).clip(0.,1.)
                cosine = 1. - interpolation(cosine, image.shape[1], image.shape[2])

                #reg_loss += 2.0 * ((1-mask) * math.exp(-iteration / opt.beta_reg)).mean()
                
                prior_loss = (torch.abs(filtered_mask - mask)).mean() * math.exp(-iteration / 10000)

                residual_loss = mlp_model.get_residual_loss(mask.flatten(), lower_mask.flatten(), upper_mask.flatten())
                cosine_loss = torch.abs(mask - cosine).mean()

                robustness_loss = 0.5 * cosine_loss + 0.5 * residual_loss
                
                if(iteration <= opt.densify_from_iter):
                    robustness_loss = robustness_loss * math.exp((iteration - opt.densify_from_iter) / 10000)

                mask_loss = robustness_loss + prior_loss
                mask_losses += (mask_loss / gaussiansN)


            # import torchvision.utils as tutils
            # output_path = os.path.join("./try", args.model_path.split('/')[-1])
            # os.makedirs(output_path, exist_ok=True)
            # tutils.save_image(loss_mask.float(), os.path.join(output_path, f"{image_name}.png"))
            
            reg_loss = 0.5 * mlp_model.get_regularizer()
            mask_losses += reg_loss
            mask_losses.backward()
            mlp_optimizer.step()
            mlp_optimizer.zero_grad(set_to_none=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                for i in range(1, gaussiansN):
                    scene.save_other_gaussian(iteration, GsDict[f"gs{i}"], i)

            # Densification
            if iteration < opt.densify_until_iter:
                for i in range(gaussiansN):
                # Keep track of max radii in image-space for pruning
                    gaussians = GsDict[f"gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]

                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None, radii)
                    
                    if iteration >= opt.reset_iter and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for i in range(gaussiansN):
                    gaussians = GsDict[f"gs{i}"]
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((GsDict[f"gs{0}"].capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                for i in range(1, gaussiansN):
                    torch.save((GsDict[f"gs{i}"].capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + str(i) + ".pth")

            if args.not_only_rgb and args.coprune:
                if iteration > opt.densify_from_iter and iteration % opt.coprune_interval == 0:
                    for i in range(gaussiansN):
                        for j in range(gaussiansN):
                            if i == j:
                                continue

                            source_cloud = o3d.geometry.PointCloud()
                            source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_xyz.clone().cpu().numpy())
                            target_cloud = o3d.geometry.PointCloud()
                            target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{j}"].get_xyz.clone().cpu().numpy())
                            trans_matrix = np.identity(4)
                            threshold = args.coprune_threshold
                            evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_matrix)
                            correspondence = np.array(evaluation.correspondence_set)
                            mask_consistent = torch.zeros((GsDict[f"gs{i}"].get_xyz.shape[0], 1)).cuda()
                            mask_consistent[correspondence[:, 0], :] = 1
                            GsDict[f"indice_consistent_gs{i}to{j}"] = correspondence
                            GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())
                    for i in range(gaussiansN):
                        GsDict[f"gs{i}"].prune_points(GsDict[f"mask_inconsistent_gs{i}"].squeeze())


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
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
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