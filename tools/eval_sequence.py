"""
eval_sequence.py — Full sequence evaluation for one object.

Usage:
    python3 tools/eval_sequence.py \
        --dataset_root ./datasets/linemod/Linemod_preprocessed \
        --model        trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
        --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
        --obj 1 --output_dir demo_out
"""

import _init_paths
import argparse
import os
import numpy as np
import yaml
import torch
import cv2

from lib.linemod.config        import OBJLIST, NUM_POINTS
from lib.linemod.inference     import load_models, warmup_gpu, run_inference
from lib.linemod.preprocessing import (load_frame, prepare_input, load_gt,
                                        load_model_pts, load_frame_names,
                                        load_diameter)
from lib.linemod.metrics       import compute_add
from lib.linemod.visualization import timing_plots, draw_frame_cv2

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, required=True)
parser.add_argument('--model',        type=str, required=True)
parser.add_argument('--refine_model', type=str, required=True)
parser.add_argument('--obj',          type=int, required=True,
                    help='Object ID (1,2,4,5,6,8,9,10,11,12,13,14,15)')
parser.add_argument('--output_dir',   type=str, default='demo_out')
parser.add_argument('--verbose',      action='store_true',
                    help='Print per-frame logs during evaluation loop')
opt = parser.parse_args()

def log(msg):
    if opt.verbose:
        print(msg)

assert opt.obj in OBJLIST, 'Invalid --obj. Choose from: {}'.format(OBJLIST)
obj_idx  = OBJLIST.index(opt.obj)
out_dir = os.path.join(opt.output_dir, 'sequence', 'obj{:02d}'.format(opt.obj))

# ── Setup ─────────────────────────────────────────────────────────────────────
estimator, refiner = load_models(opt.model, opt.refine_model)
warmup_gpu(estimator)

model_pts    = load_model_pts(opt.dataset_root, opt.obj)
diameter_mm  = load_diameter(opt.obj)
threshold    = diameter_mm / 1000.0 * 0.1
frame_names  = load_frame_names(opt.dataset_root, opt.obj)
idx_t        = torch.LongTensor([obj_idx]).cuda()

# Pre-load full gt.yml once for the sequence
with open('{0}/data/{1}/gt.yml'.format(
        opt.dataset_root, '%02d' % opt.obj), 'r') as f:
    gt_cache = yaml.load(f, Loader=yaml.SafeLoader)

print('Object {:02d} — {} test frames'.format(opt.obj, len(frame_names)))

# ── Sequence loop ─────────────────────────────────────────────────────────────
t_est_seq    = []
t_ref_seq    = []
add_seq      = []
success_seq  = []
video_frames = []

for frame_i, fname in enumerate(frame_names):
    img, depth, label, img_arr = load_frame(opt.dataset_root, opt.obj, fname)
    result = prepare_input(img, depth, label)
    if result is None:
        print('Frame {} — empty mask, skipped'.format(fname))
        continue

    img_t, cloud_t, choose_t, bbox, mask_label = result
    R_pred, t_pred, t_est, t_ref = run_inference(
        estimator, refiner, img_t, cloud_t, choose_t, idx_t, measure_time=True)

    R_gt, t_gt = load_gt(opt.dataset_root, opt.obj, fname, gt_cache)
    add_val, _ = compute_add(R_pred, t_pred, R_gt, t_gt, model_pts, obj_idx)
    success     = add_val < threshold
    t_total     = t_est + t_ref

    t_est_seq.append(t_est)
    t_ref_seq.append(t_ref)
    add_seq.append(add_val * 1000.0)
    success_seq.append(int(success))

    frame_bgr = draw_frame_cv2(img_arr, R_pred, t_pred, R_gt, t_gt,
                                model_pts, bbox, add_val, threshold,
                                frame_i, t_total)
    video_frames.append(frame_bgr)

    log('Frame {:04d}/{:04d}  t={:.1f}ms  ADD={:.2f}mm  {}'.format(
        frame_i + 1, len(frame_names), t_total,
        add_val * 1000, 'PASS' if success else 'FAIL'))

# ── Save video ────────────────────────────────────────────────────────────────
video_path = os.path.join(out_dir, 'livefeed.mp4')
h, w = video_frames[0].shape[:2]
writer = cv2.VideoWriter(video_path,
    cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
for f in video_frames:
    writer.write(f)
writer.release()
print('Video saved: {}'.format(video_path))

# ── Summary ───────────────────────────────────────────────────────────────────
t_tot        = np.array(t_est_seq) + np.array(t_ref_seq)
success_rate = np.mean(success_seq) * 100.0

print('\n── Sequence summary ─────────────────────────────────────────────')
print('Object {:02d}  |  {} frames  |  success {:.1f}%'.format(
    opt.obj, len(t_tot), success_rate))
print('{:<18} {:>8} {:>8} {:>8} {:>8}'.format('', 'median', 'mean', 'p95', 'max'))
for label, arr in [('Estimator (ms)', np.array(t_est_seq)),
                   ('Refiner (ms)',   np.array(t_ref_seq)),
                   ('Total (ms)',     t_tot)]:
    print('{:<18} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}'.format(
        label, np.median(arr), np.mean(arr),
        np.percentile(arr, 95), np.max(arr)))
print('Achievable fps (median / p95): {:.1f} / {:.1f}'.format(
    1000/np.median(t_tot), 1000/np.percentile(t_tot, 95)))

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_title = ('DenseFusion — Object {:02d} sequence  |  {} frames  |  '
              'success {:.1f}%  |  median {:.1f} ms ({:.1f} fps)').format(
    opt.obj, len(t_tot), success_rate,
    np.median(t_tot), 1000/np.median(t_tot))

plot_path = os.path.join(out_dir, 'performance.png')
timing_plots(t_est_seq, t_ref_seq, add_seq, success_seq,
             diameter_mm * 0.1,
             plot_title,
             plot_path)
