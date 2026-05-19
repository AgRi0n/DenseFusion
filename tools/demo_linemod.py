"""
demo_linemod.py — Single-frame inference visualizer for LineMOD.

Usage:
    python3 tools/demo_linemod.py \
        --dataset_root ./datasets/linemod/Linemod_preprocessed \
        --model        trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
        --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
        --obj 1 --idx 0 --output_dir demo_out
"""

import _init_paths
import argparse
import os
import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib.linemod.config      import OBJLIST, NUM_POINTS
from lib.linemod.inference   import load_models, run_inference
from lib.linemod.preprocessing import (load_frame, prepare_input, load_gt,
                                        load_model_pts, load_frame_names,
                                        load_diameter)
from lib.linemod.metrics     import compute_pose_metrics
from lib.linemod.visualization import (project_points, pose_panel,
                                       bbox_3d_corners, draw_bbox_3d)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, required=True)
parser.add_argument('--model',        type=str, required=True)
parser.add_argument('--refine_model', type=str, required=True)
parser.add_argument('--obj',          type=int, default=1,
                    help='Object ID (1,2,4,5,6,8,9,10,11,12,13,14,15)')
parser.add_argument('--idx',          type=int, default=0,
                    help='Frame index within the object test set')
parser.add_argument('--output_dir',   type=str, default='demo_out')
opt = parser.parse_args()

assert opt.obj in OBJLIST, 'Invalid --obj. Choose from: {}'.format(OBJLIST)
obj_idx  = OBJLIST.index(opt.obj)
# Anchor output path from repo root (parent of tools/) regardless of cwd
out_dir = os.path.join(opt.output_dir, 'demo', 'obj{:02d}'.format(opt.obj))
os.makedirs(out_dir, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────
estimator, refiner = load_models(opt.model, opt.refine_model)
model_pts = load_model_pts(opt.dataset_root, opt.obj)
frame_names = load_frame_names(opt.dataset_root, opt.obj)
frame_name  = frame_names[opt.idx]
print('Loaded: obj={}, frame={}'.format(opt.obj, frame_name))

# ── Frame ─────────────────────────────────────────────────────────────────────
img, depth, label, img_arr = load_frame(opt.dataset_root, opt.obj, frame_name)
result = prepare_input(img, depth, label)
if result is None:
    print('ERROR: empty mask for this frame, try a different --idx')
    exit(1)
img_t, cloud_t, choose_t, bbox, mask_label = result
idx_t = torch.LongTensor([obj_idx]).cuda()

# ── Inference ─────────────────────────────────────────────────────────────────
R, t, _, _ = run_inference(estimator, refiner, img_t, cloud_t, choose_t, idx_t)

# ── Ground truth and metrics ──────────────────────────────────────────────────
R_gt, t_gt   = load_gt(opt.dataset_root, opt.obj, frame_name)
diameter_mm  = load_diameter(opt.obj)
metrics      = compute_pose_metrics(R, t, R_gt, t_gt, model_pts, obj_idx, diameter_mm)

print('\n── Pose Metrics ────────────────────────────────────────────────')
print('{}: {:.2f} mm  (thr {:.2f} mm)  →  {}'.format(
    metrics['add_metric'], metrics['add_value_mm'], metrics['add_threshold_mm'],
    'PASS ✓' if metrics['add_success'] else 'FAIL ✗'))
print('Geodesic  : {:.4f}°'.format(metrics['geodesic_deg']))
print('Euclidean : {:.2f} mm'.format(metrics['euclidean_mm']))
print('────────────────────────────────────────────────────────────────\n')

# ── Visualize ─────────────────────────────────────────────────────────────────
sample_idx = np.random.choice(len(model_pts), min(500, len(model_pts)), replace=False)
pts_s      = model_pts[sample_idx] / 1000.0
proj_pred  = project_points(pts_s, R,    t)
proj_gt    = project_points(pts_s, R_gt, t_gt)
bbox_3d_m  = bbox_3d_corners(model_pts) / 1000.0  # mm → m (model frame)

rmin, rmax, cmin, cmax = bbox
status      = '✓ PASS' if metrics['add_success'] else '✗ FAIL'
title_color = 'green' if metrics['add_success'] else 'red'
sup = ('DenseFusion — Object {:02d}, Frame {}    {}  {} = {:.2f} mm  |  '
       'Geodesic = {:.2f}°  |  Euclidean = {:.2f} mm').format(
    opt.obj, frame_name, status,
    metrics['add_metric'], metrics['add_value_mm'],
    metrics['geodesic_deg'], metrics['euclidean_mm'])

# Figure 1 — overview
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
fig1.suptitle(sup, fontsize=11, color=title_color)
axes1[0].imshow(img_arr)
axes1[0].add_patch(patches.Rectangle(
    (cmin, rmin), cmax-cmin, rmax-rmin, linewidth=2, edgecolor='lime',
    facecolor='none', linestyle='--'))
# Ground-truth 3D bbox only; the predicted bbox is shown in panel 3.
draw_bbox_3d(axes1[0], bbox_3d_m, R_gt, t_gt, color='lime', lw=1.5)
axes1[0].set_title('RGB input + 2D detection (dashed) & GT 3D bbox (green)')
axes1[0].set_xlim(0, img_arr.shape[1])
axes1[0].set_ylim(img_arr.shape[0], 0)
axes1[0].axis('off')
axes1[1].imshow(mask_label, cmap='gray')
axes1[1].set_title('Segmentation mask (SegNet output)')
axes1[1].axis('off')
# 3D bbox + pose axes only, no model-point overlay (cleaner read of the
# pose against the raw image).
pose_panel(axes1[2], proj_pred, proj_gt, R, t, R_gt, t_gt,
           background=img_arr,
           title='3D bbox (red=pred, green=GT) + pose axes — RGB',
           bbox_3d=bbox_3d_m, show_points=False)
fig1.tight_layout()
p1 = os.path.join(out_dir, 'frame{:04d}_overview.png'.format(opt.idx))
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print('Saved: {}'.format(p1))

# Figure 2 — masked object panels
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig2.suptitle(sup, fontsize=11, color=title_color)
pose_panel(axes2[0], proj_pred, proj_gt, R, t, R_gt, t_gt,
           background=img_arr, title='Full frame — masked object',
           mask_label=mask_label, bbox_3d=bbox_3d_m)
pose_panel(axes2[1], proj_pred, proj_gt, R, t, R_gt, t_gt,
           background=img_arr, title='Object crop',
           mask_label=mask_label, axes_scale=0.3,
           bbox_3d=bbox_3d_m, bbox_lw=2.0)
axes2[1].set_xlim(cmin, cmax)
axes2[1].set_ylim(rmax, rmin)
fig2.tight_layout()
p2 = os.path.join(out_dir, 'frame{:04d}_masked.png'.format(opt.idx))
fig2.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('Saved: {}'.format(p2))
