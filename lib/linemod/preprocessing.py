"""
lib/linemod/preprocessing.py — Frame loading and input tensor preparation.

Covers everything between raw files on disk and model-ready CUDA tensors:
  - load_frame()    : reads RGB, depth, segnet label from disk
  - prepare_input() : masks, back-projects depth to point cloud, crops image
  - load_gt()       : reads ground truth R and t from gt.yml
  - load_model_pts(): reads 3D model vertices from .ply
"""

import os
import numpy as np
import numpy.ma as ma
import yaml
import torch
from PIL import Image

from lib.linemod.config import (
    CAM_CX, CAM_CY, CAM_FX, CAM_FY,
    XMAP, YMAP, NORM, NUM_POINTS
)
from datasets.linemod.dataset import get_bbox, mask_to_bbox, ply_vtx


def _load_mask(path):
    """Load a binary object mask from disk and normalise it to [H, W] uint8
    with 255 marking object pixels.

    LineMOD ships two mask formats:
      - SegNet predictions (segnet_results/<obj>_label/<fname>_label.png) —
        single-channel grayscale, 255 = object.
      - Ground-truth masks (data/<obj>/mask/<fname>.png) — sometimes
        grayscale (255 = object), sometimes RGB ((255,255,255) = object,
        which is how the original training dataset code reads them).
    This helper accepts either and always returns the grayscale convention
    so the downstream code in prepare_input() only has to handle one case.
    """
    img = np.array(Image.open(path))
    if img.ndim == 3:
        # RGB ground-truth mask — fold to grayscale on white pixels.
        white = ((img[:, :, 0] == 255) &
                 (img[:, :, 1] == 255) &
                 (img[:, :, 2] == 255))
        return (white.astype(np.uint8) * 255)
    return img.astype(np.uint8)


def load_frame(dataset_root, obj, frame_name, mask_source='segnet'):
    """Load RGB, depth and object mask for one frame.

    Parameters
    ----------
    mask_source : 'segnet' (default) or 'gt'
        - 'segnet' reads the SegNet prediction at
              segnet_results/<obj>_label/<fname>_label.png
          (only present for the test split — what eval_linemod_bench uses).
        - 'gt' reads the ground-truth mask at
              data/<obj>/mask/<fname>.png
          (present for every frame, train and test). Use this when running
          on the union of train+test so the mask quality stays uniform
          across the sequence.

    Returns
    -------
    img        : PIL.Image  (H x W x 3)
    depth      : np.ndarray (H x W), uint16, depth in mm
    label      : np.ndarray (H x W), uint8, 255 = object pixel
    img_arr    : np.ndarray (H x W x 3), uint8 copy of RGB
    """
    rgb_path   = '{0}/data/{1}/rgb/{2}.png'.format(
        dataset_root, '%02d' % obj, frame_name)
    depth_path = '{0}/data/{1}/depth/{2}.png'.format(
        dataset_root, '%02d' % obj, frame_name)

    if mask_source == 'segnet':
        label_path = '{0}/segnet_results/{1}_label/{2}_label.png'.format(
            dataset_root, '%02d' % obj, frame_name)
    elif mask_source == 'gt':
        label_path = '{0}/data/{1}/mask/{2}.png'.format(
            dataset_root, '%02d' % obj, frame_name)
    else:
        raise ValueError("mask_source must be 'segnet' or 'gt', got {!r}"
                         .format(mask_source))

    img     = Image.open(rgb_path)
    depth   = np.array(Image.open(depth_path))
    label   = _load_mask(label_path)
    img_arr = np.array(img)[:, :, :3]

    return img, depth, label, img_arr


def prepare_input(img, depth, label):
    """Preprocess one frame into model-ready CUDA tensors.

    Replicates dataset.py __getitem__ preprocessing:
      1. Combine depth validity mask and segnet label mask
      2. Snap bounding box to border grid
      3. Sample exactly NUM_POINTS pixels from the masked crop
      4. Back-project sampled depth pixels into 3D point cloud
      5. Normalise and pack image crop

    Returns None if the segmentation mask is empty.

    Returns
    -------
    img_t      : torch.Tensor [1, 3, H', W'] on CUDA
    cloud_t    : torch.Tensor [1, NUM_POINTS, 3] on CUDA
    choose_t   : torch.LongTensor [1, NUM_POINTS] on CUDA
    bbox       : tuple (rmin, rmax, cmin, cmax)
    mask_label : np.ndarray bool [H, W], True = object pixel
    """
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    mask = mask_label * mask_depth

    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

    img_arr  = np.array(img)[:, :, :3]
    img_crop = np.transpose(img_arr, (2, 0, 1))[:, rmin:rmax, cmin:cmax]

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
        return None

    if len(choose) > NUM_POINTS:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:NUM_POINTS] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, NUM_POINTS - len(choose)), 'wrap')

    depth_m = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_m  = XMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_m  = YMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2   = depth_m / 1000.0
    pt0   = (ymap_m - CAM_CX) * pt2 / CAM_FX
    pt1   = (xmap_m - CAM_CY) * pt2 / CAM_FY
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    img_t    = NORM(torch.from_numpy(img_crop.astype(np.float32)))
    cloud_t  = torch.from_numpy(cloud.astype(np.float32))
    choose_t = torch.LongTensor(np.array([choose]).astype(np.int32))

    return (img_t.cuda().unsqueeze(0),
            cloud_t.cuda().unsqueeze(0),
            choose_t.cuda(),
            (rmin, rmax, cmin, cmax),
            mask_label)


def load_gt(dataset_root, obj, frame_name, gt_cache=None):
    """Load ground truth R and t for one frame from gt.yml.

    Parameters
    ----------
    gt_cache : dict or None
        Pass a pre-loaded gt.yml dict to avoid re-reading the file on every frame
        (use when iterating over a full sequence).

    Returns
    -------
    R_gt : np.ndarray (3, 3)
    t_gt : np.ndarray (3,), in metres
    """
    if gt_cache is None:
        with open('{0}/data/{1}/gt.yml'.format(
                dataset_root, '%02d' % obj), 'r') as f:
            gt_cache = yaml.load(f, Loader=yaml.SafeLoader)

    rank = int(frame_name)
    if obj == 2:
        meta = next(m for m in gt_cache[rank] if m['obj_id'] == 2)
    else:
        meta = gt_cache[rank][0]

    R_gt = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    t_gt = np.array(meta['cam_t_m2c']) / 1000.0  # mm → m
    return R_gt, t_gt


def load_model_pts(dataset_root, obj):
    """Load 3D model vertices from the .ply file (in mm)."""
    return ply_vtx('{0}/models/obj_{1}.ply'.format(
        dataset_root, '%02d' % obj))


def load_frame_names(dataset_root, obj, split='test'):
    """Return an ordered list of frame name strings for the requested split.

    Parameters
    ----------
    split : 'test' | 'train' | 'all'
        - 'test'  : just test.txt, in file order.
        - 'train' : just train.txt, in file order.
        - 'all'   : union of train.txt and test.txt, sorted numerically by
                    LineMOD frame id (deduped). This produces the natural
                    capture order the camera recorded the sequence in,
                    with no gaps — useful for fluid replay over the whole
                    sequence rather than the sparse test-only timeline.
    """
    def _read(path):
        with open(path, 'r') as f:
            return [l.strip() for l in f if l.strip()]

    if split in ('test', 'train'):
        path = '{0}/data/{1}/{2}.txt'.format(dataset_root, '%02d' % obj, split)
        return _read(path)

    if split == 'all':
        train_path = '{0}/data/{1}/train.txt'.format(dataset_root, '%02d' % obj)
        test_path  = '{0}/data/{1}/test.txt'.format( dataset_root, '%02d' % obj)
        # Use set() for dedup (rare overlap is harmless), then sort by the
        # integer id so the resulting timeline matches the capture order.
        merged = set(_read(train_path)) | set(_read(test_path))
        return sorted(merged, key=lambda s: int(s))

    raise ValueError("split must be 'test', 'train' or 'all', got {!r}"
                     .format(split))


def load_frame_names_with_origin(dataset_root, obj, split='all'):
    """Same as load_frame_names() but also returns which split each frame
    came from.

    Returns
    -------
    names   : list[str]  — ordered frame name strings
    origins : list[str]  — parallel list, each entry is 'train' or 'test'

    Useful when downstream code (e.g. HUD overlays, ADD plots) wants to
    distinguish train frames from test frames visually.
    """
    def _read(path):
        with open(path, 'r') as f:
            return [l.strip() for l in f if l.strip()]

    train_set = set(_read('{0}/data/{1}/train.txt'.format(
        dataset_root, '%02d' % obj)))
    test_set  = set(_read('{0}/data/{1}/test.txt'.format(
        dataset_root, '%02d' % obj)))
    names     = load_frame_names(dataset_root, obj, split=split)
    # If a frame appears in both files (extremely unlikely on LineMOD but
    # possible after a re-split), prefer the 'test' label as the more
    # informative one for evaluation.
    origins   = ['test' if n in test_set else 'train' for n in names]
    return names, origins


def load_diameter(obj, dataset_config_dir='datasets/linemod/dataset_config'):
    """Return object diameter in mm from models_info.yml.

    Parameters
    ----------
    dataset_config_dir : str
        Path to the dataset_config directory, relative to the repo root.
        Pass an absolute path if the working directory may vary.
    """
    path = os.path.join(dataset_config_dir, 'models_info.yml')
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.SafeLoader)
    return info[obj]['diameter']
