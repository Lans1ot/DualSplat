import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss_prepare as l1_loss, ssim_prepare as ssim
from gaussian_renderer import render
import sys
import uuid
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
from utils.image_utils import resize_mask_nearest_hw
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def compute_instance_loss_sum(combined_loss, label_map):

    instance_ids = torch.unique(label_map)
    instance_loss_sum = {}

    for instance_id in instance_ids:
        if instance_id.item() == 0:
            continue
        mask = (label_map == instance_id).float()
        instance_area = mask.sum()
        instance_loss = (combined_loss * mask).sum().item() / instance_area.item()
        instance_loss_sum[instance_id.item()] = instance_loss

    return instance_loss_sum

def compute_combined_loss(l1_loss_val, ssim_val):
    Ll1_norm = normalize_to_01(l1_loss_val.mean(dim=0)).detach()
    ssim_value_norm = normalize_to_01(ssim_val).detach()

    combined_loss = (4 * Ll1_norm + ssim_value_norm) / 5.0
    return combined_loss, Ll1_norm, ssim_value_norm

def compute_instance_losses(combined_loss, label_map, iteration, base_threshold=1., threshold_local=1.5):
    instance_loss_sum = compute_instance_loss_sum(combined_loss, label_map)
    heatmap = torch.zeros_like(label_map, dtype=torch.float32)

    for instance_id, loss_sum in instance_loss_sum.items():
        if instance_id == 0:
            continue
        mask = (label_map == instance_id).float()
        heatmap += mask * loss_sum

    heatmap_norm = normalize_to_01(heatmap)

    mean_value = heatmap_norm.mean()
    std_value = heatmap_norm.std()
    instance_threshold = mean_value + std_value * (base_threshold + threshold_local * (args.iterations - iteration) / args.iterations)

    # heatmap_norm = torch.tensor(heatmap_norm, dtype=torch.float32).cuda()
    heatmap_norm = heatmap_norm.clone().detach().cuda()
    heatmap_binary = (heatmap_norm < instance_threshold).float()

    return heatmap_norm, heatmap_binary

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def normalize_to_01(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    mask_dict = {}

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        image_name = viewpoint_cam.image_name
        image_name = image_name if image_name.find(".") != -1 else image_name + ".png"
        if image_name not in mask_dict.keys():
            origin_mask = viewpoint_cam.get_origin_mask(os.path.join(args.source_path, args.origin_masks, image_name)).cuda()
            label_map_for_image = resize_mask_nearest_hw(origin_mask, image)
            mask_dict[image_name] = label_map_for_image
        else:
            label_map_for_image = mask_dict[image_name]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = 1.0 - ssim(image, gt_image)

        combined_loss, Ll1_norm, ssim_value_norm = compute_combined_loss(Ll1, ssim_value)

        label_map_for_image = label_map_for_image.clone().detach().cuda()
        image_name = viewpoint_cam.image_name
        heatmap_norm, heatmap_binary = compute_instance_losses(combined_loss, label_map_for_image, iteration)

        if iteration > args.mask_start_iter:
            loss = (1.0 - opt.lambda_dssim) * (Ll1 * heatmap_binary).mean() + opt.lambda_dssim * (ssim_value * heatmap_binary).mean()
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1.mean() + opt.lambda_dssim * ssim_value.mean()


        loss.backward()

        iter_end.record()

        with torch.no_grad():

            densify_grad_threshold = opt.densify_grad_threshold
            # if args.schedule_densify_grad_threshold:
            #     densify_grad_threshold = densify_grad_threshold + (0.001 - densify_grad_threshold) * (iteration / opt.iterations)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1.mean(), loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, render_pkg["radii"])

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
    #parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 2000, 3000,4000, 5000, 6000, 7_000, 10000, 15000, 20000, 30000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--use_mask", default=True)
    parser.add_argument("--mask_start_iter", type=int, default=500)
    # parser.add_argument("--threshold_local", type=float, default=1.5)
    # parser.add_argument("--use_masks", action="store_true")
    # parser.add_argument("--mask_start_iter", type=int, default=500)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)

    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    print("\nTraining complete.")