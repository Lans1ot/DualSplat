#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instance segmentation of images using SAM2 model and saving the results as label maps.
"""

# Standard library imports
import os
import json
import argparse
from typing import Dict, List, Any, Tuple

# Third-party imports
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Local module imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def setup_gpu_acceleration() -> None:
    """
    Configure GPU acceleration settings, including automatic mixed precision and TensorFloat32 support.
    """
    # Enable automatic mixed precision
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # For Ampere and above GPU architectures, enable TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Enable TensorFloat32 to improve matrix multiplication performance
        # Reference: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Namespace containing all parameters
    """
    parser = argparse.ArgumentParser(description='Segment images using SAM2 model and save label maps.')
    parser.add_argument('--model_checkpoint', 
                        type=str, 
                        default="checkpoints/sam2_hiera_large.pt", 
                        help='Path to SAM2 model checkpoint')
    parser.add_argument('--model_cfg', 
                        type=str, 
                        default="sam2_hiera_l.yaml", 
                        help='Path to SAM2 model configuration')
    parser.add_argument('--image_dir', 
                        type=str, 
                        help='Directory containing images to segment')
    return parser.parse_args()


def show_annotations(annotations: List[Dict[str, Any]], borders: bool = False) -> None:
    """
    Display masks with optional borders.
    
    Args:
        annotations: List of annotations containing segmentation masks
        borders: Whether to display borders, default is True
    """
    if not annotations:
        return

    sorted_annotations = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create transparent background image
    mask_shape = sorted_annotations[0]['segmentation'].shape
    img = np.ones((mask_shape[0], mask_shape[1], 4))
    img[:, :, 3] = 0
    
    for annotation in sorted_annotations:
        mask = annotation['segmentation']
        # Generate random color for each mask
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[mask] = color_mask 
        
        # Draw borders if needed
        if borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                       for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
            
    ax.imshow(img)


def is_fully_contained(mask_a: Dict[str, Any], mask_b: Dict[str, Any]) -> bool:
    """
    Check if mask_a is fully contained within mask_b.
    
    Args:
        mask_a: First mask
        mask_b: Second mask
        
    Returns:
        bool: True if mask_a is fully contained in mask_b
    """
    # Check if bounding box is contained
    bbox_a, bbox_b = mask_a['bbox'], mask_b['bbox']
    if not (bbox_a[0] >= bbox_b[0] and bbox_a[1] >= bbox_b[1] and
            bbox_a[0] + bbox_a[2] <= bbox_b[0] + bbox_b[2] and
            bbox_a[1] + bbox_a[3] <= bbox_b[1] + bbox_b[3]):
        return False
    
    # Check if segmentation mask is contained
    mask_a_pixels = mask_a['segmentation']
    mask_b_pixels = mask_b['segmentation']
    if np.all(mask_a_pixels[mask_a_pixels] == mask_b_pixels[mask_a_pixels]):
        return True
    
    return False


def merge_masks(masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge masks by removing those fully contained within others.
    
    Args:
        masks: List of masks to merge
        
    Returns:
        List[Dict[str, Any]]: List of merged masks
    """
    merged_masks = []
    for i, mask_a in enumerate(masks):
        merged = False
        for j, mask_b in enumerate(masks):
            if i != j and is_fully_contained(mask_a, mask_b):
                merged = True
                break
        if not merged:
            merged_masks.append(mask_a)
    return merged_masks

def merge_masks_(masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replace 'remove contained masks' with 'set difference' behavior:
    If mask A contains mask B, keep B, and mutate A as A - B.
    Masks whose area becomes zero after subtractions are dropped.
    """
    # 深拷贝，避免原列表被外部复用时出意外
    out = [dict(m) for m in masks]
    print("mask count : " + str(len(masks)))

    def _recompute_area_bbox(seg: np.ndarray) -> Tuple[int, List[int]]:
        """根据二值mask重算面积和bbox（xywh）。若为空，返回0和[0,0,0,0]。"""
        seg_bool = seg.astype(bool)
        area = int(seg_bool.sum())
        if area == 0:
            return 0, [0, 0, 0, 0]
        ys, xs = np.where(seg_bool)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # xywh：宽高需要 +1（像素索引是闭区间）
        bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
        return area, bbox

    n = len(out)
    # 两两检查：如果 mask_i 包含 mask_j（即 is_fully_contained(mask_j, mask_i)），则 i ← i - j
    for i in range(n):
        #print(out[i]['segmentation'])
        #seg_i = out[i]['segmentation'].astype(bool)
        seg_i = out[i]['segmentation']
        changed = False
        for j in range(n):
            if i == j:
                continue
            # a 包含 b <=> is_fully_contained(b, a)
            if is_fully_contained(out[j], out[i]):
                seg_j = out[j]['segmentation'].astype(bool)
                # 集合减法：A = A & (~B)
                new_seg_i = np.logical_and(seg_i, np.logical_not(seg_j))
                if new_seg_i is not seg_i:
                    # 确保引用更新
                    seg_i = new_seg_i
                else:
                    # 即便返回同一对象，也在值层面发生了变化
                    pass
                changed = True

        if changed:
            area_i, bbox_i = _recompute_area_bbox(seg_i)
            #out[i]['segmentation'] = seg_i.astype(np.uint8)
            out[i]['segmentation'] = seg_i
            out[i]['area'] = area_i
            out[i]['bbox'] = bbox_i

    # 丢弃被“挖空”成空集合的mask
    out = [m for m in out if int(np.asarray(m['segmentation']).astype(bool).sum()) > 0]

    return out


def create_label_map(image_name: str, masks: List[Dict[str, Any]], 
                    image_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Create a label map for a given image based on its masks.
    
    Args:
        image_name: Image name
        masks: List of masks for the image
        image_shape: Image shape (height, width)
        
    Returns:
        Dict[str, Any]: Dictionary containing image name and label map
    """
    # Sort masks by area in descending order
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    label_map = np.zeros(image_shape[:2], dtype=np.int32)
    
    # Assign unique label to each mask
    for idx, mask in enumerate(sorted_masks, start=1):
        segmentation = mask['segmentation'].astype(np.int32)
        label_map[segmentation == 1] = idx
    
    data = {
        'image_name': image_name,
        'label_map': label_map.tolist()
    }
    
    return data


def save_label_maps_as_json(label_maps: Dict[str, Any], output_file_path: str) -> None:
    """
    Save all label maps to a single JSON file.
    
    Args:
        label_maps: Dictionary of label maps
        output_file_path: Output JSON file path
    """
    try:
        with open(output_file_path, 'w') as json_file:
            json.dump(label_maps, json_file)
        print(f"Successfully saved label maps to: {output_file_path}")
    except Exception as e:
        print(f"Error saving label maps: {e}")


def load_sam2_model(model_cfg: str, model_checkpoint: str) -> SAM2AutomaticMaskGenerator:
    """
    Load SAM2 model and create automatic mask generator.
    
    Args:
        model_cfg: Model configuration file path
        model_checkpoint: Model checkpoint path
        
    Returns:
        SAM2AutomaticMaskGenerator: Configured mask generator
    """
    try:
        # Load SAM2 model
        sam2_model = build_sam2(model_cfg, model_checkpoint, device='cuda', apply_postprocessing=False)
        
        # Create automatic mask generator
        # mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, pred_iou_thresh=0.7, stability_score_thresh=0.9, crop_n_layers=1) # fast
        # mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, points_per_side=32, points_per_batch=128, pred_iou_thresh=0.7, stability_score_thresh=0.92, stability_score_offset=0.7, crop_n_layers=1, box_nms_thresh=0.7, crop_n_points_downscale_factor=2, min_mask_region_area=25.0, use_m2m=True,)
        # mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, points_per_side=32, points_per_batch=128, pred_iou_thresh=0.8, stability_score_thresh=0.9, box_nms_thresh=0.8, crop_n_layers=2, crop_nms_thresh=0.8, crop_overlap_ratio=0.6, crop_n_points_downscale_factor=1, min_mask_region_area=0, use_m2m=False, multimask_output=True, mask_threshold=0.0, output_mode="binary_mask") # slow, origin


        return mask_generator
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        raise


def process_image(image_path: str, mask_generator: SAM2AutomaticMaskGenerator, 
                 output_dir: str) -> Tuple[str, np.ndarray]:
    """
    Process a single image and generate masks and label map.
    
    Args:
        image_path: Image file path
        mask_generator: SAM2 mask generator
        output_dir: Output directory
        
    Returns:
        Tuple[str, np.ndarray]: Image name and label map data
    """
    try:
        # print(f"Processing image: {image_path}")
        # Open image and get dimensions
        image = Image.open(image_path)
        width, height = image.size
        # Convert image to NumPy array
        image_array = np.array(image.convert("RGB"))
        
        # Generate and merge masks
        masks = mask_generator.generate(image_array)
        merged_masks = merge_masks(masks)
        
        # Create visualization image
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111)
        plt.imshow(np.zeros((height, width, 4)))  # Create fully transparent background
        show_annotations(merged_masks)
        
        # Save mask visualization
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '.jpg')
        plt.axis('off')
        ax.set_position([0, 0, 1, 1])  # Ensure content fills entire figure
        plt.savefig(output_file_path, bbox_inches=None, pad_inches=0)
        plt.close()
        
        # Create label map
        img_name = os.path.basename(image_path)
        label_map_data = create_label_map(img_name, merged_masks, (height, width))
        
        return img_name, label_map_data['label_map']
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise e


def main() -> None:
    """
    Main function to process all images and save results.
    """
    # Parse arguments and setup GPU acceleration
    args = parse_arguments()
    setup_gpu_acceleration()
    
    try:
        # Load model
        mask_generator = load_sam2_model(args.model_cfg, args.model_checkpoint)
        
        # Prepare image paths
        images_dir = os.path.join(args.image_dir, "images")
        print(f"Processing" + images_dir)
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
            
        image_extensions = ('.JPG', '.jpg', '.png', '.jpeg')
        image_paths = [
            os.path.join(images_dir, fname) 
            for fname in os.listdir(images_dir) 
            if fname.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print(f"Warning: No image files found in {images_dir}")
            return
            
        # Output directory
        output_dir = os.path.join(args.image_dir, 'sam_masks')
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths.sort()
        # Process all images
        all_label_maps = {}
        for image_path in tqdm(image_paths, desc="Processing images"):
            img_name, label_map = process_image(image_path, mask_generator, output_dir)
            if img_name and label_map is not None:
                all_label_maps[img_name] = label_map
            np.save(os.path.join(output_dir, img_name.split(".")[0] + ".npy"), label_map)
        
        # Save all label maps
        if all_label_maps:
            output_json_path = os.path.join(output_dir, 'masks.json')
            #save_label_maps_as_json(all_label_maps, output_json_path)
        else:
            #print("Warning: No label maps were generated")
            pass
            
    except Exception as e:
        print(f"Error: {e}")
        raise e


if __name__ == "__main__":
    main()



