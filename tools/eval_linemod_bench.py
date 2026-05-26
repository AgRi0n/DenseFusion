"""
eval_linemod_bench.py — Full LineMOD evaluation with per-frame timing.

Usage:
    python3 tools/eval_linemod_bench.py \
        --dataset_root ./datasets/linemod/Linemod_preprocessed \
        --model        trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
        --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
        [--verbose] [--output_dir demo_out]
"""

import _init_paths
import argparse
import os
import copy
import numpy as np
import yaml
import torch
import torch.utils.data

from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.linemod.config      import OBJLIST, NUM_POINTS, ITERATION
from lib.linemod.inference   import load_models, warmup_gpu, run_inference
from lib.linemod.metrics     import compute_add
from lib.linemod.visualization import timing_plots

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='')
parser.add_argument('--model',        type=str, default='')
parser.add_argument('--refine_model', type=str, default='')
parser.add_argument('--verbose',      action='store_true',
                    help='Print per-frame logs during evaluation loop')
parser.add_argument('--output_dir',   type=str, default='demo_out')
opt = parser.parse_args()

def log(msg):
    if opt.verbose:
        print(msg)

out_dir = os.path.join(opt.output_dir, 'bench')

# ── Setup ─────────────────────────────────────────────────────────────────────
estimator, refiner = load_models(opt.model, opt.refine_model)

testdataset    = PoseDataset_linemod('eval', NUM_POINTS, False,
                                     opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(
    testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list         = testdataset.get_sym_list()
num_points_mesh  = testdataset.get_num_points_mesh()
criterion        = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
with open('datasets/linemod/dataset_config/models_info.yml', 'r') as f:
    meta = yaml.load(f, Loader=yaml.SafeLoader)
for obj in OBJLIST:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)

success_count = [0] * len(OBJLIST)
num_count     = [0] * len(OBJLIST)

fw = open(os.path.join(out_dir, 'eval_logs.txt'), 'w')

t_est_all = []
t_ref_all = []
add_all   = []
suc_all   = []

warmup_gpu(estimator)

# ── Eval loop ─────────────────────────────────────────────────────────────────
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data

    if len(points.size()) == 2:
        log('No.{} NOT Pass! Lost detection!'.format(i))
        if opt.verbose:
            fw.write('No.{} NOT Pass! Lost detection!\n'.format(i))
        continue

    points, choose, img, target, model_points, idx = (
        points.cuda(), choose.cuda(), img.cuda(),
        target.cuda(), model_points.cuda(), idx.cuda())

    R, t, t_est, t_ref = run_inference(
        estimator, refiner, img, points, choose, idx, measure_time=True)

    # ADD score
    mp_np = model_points[0].cpu().detach().numpy()
    tg_np = target[0].cpu().detach().numpy()
    add, _ = compute_add(R,
                         t,
                         np.zeros((3, 3)),   # placeholder — using dataset target
                         np.zeros(3),
                         mp_np * 1000.0,
                         idx[0].item())

    # Use dataset-style distance (target already transformed by GT)
    from lib.transformations import quaternion_matrix
    r_mat = R
    pred  = np.dot(mp_np, r_mat.T) + t
    if idx[0].item() in sym_list:
        pred_t_   = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target_t_ = torch.from_numpy(tg_np.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = torch.cdist(pred_t_.transpose(0,1).unsqueeze(0),
                           target_t_.transpose(0,1).unsqueeze(0)).argmin(dim=2)
        target_t_ = torch.index_select(target_t_, 1, inds.view(-1))
        dis = torch.mean(torch.norm(
            pred_t_.transpose(1,0) - target_t_.transpose(1,0), dim=1)).item()
    else:
        dis = float(np.mean(np.linalg.norm(pred - tg_np, axis=1)))

    success = dis < diameter[idx[0].item()]
    if success:
        success_count[idx[0].item()] += 1
    num_count[idx[0].item()] += 1

    t_est_all.append(t_est)
    t_ref_all.append(t_ref)
    add_all.append(dis * 1000.0)
    suc_all.append(int(success))

    msg = 'No.{} {}! dis={:.4f}  t_est={:.1f}ms  t_ref={:.1f}ms'.format(
        i, 'Pass' if success else 'NOT Pass', dis, t_est, t_ref)
    log(msg)
    fw.write(msg + '\n')

# ── Results ───────────────────────────────────────────────────────────────────
for i in range(len(OBJLIST)):
    msg = 'Object {} success rate: {:.4f}'.format(
        OBJLIST[i], float(success_count[i]) / num_count[i])
    log(msg)
    fw.write(msg + '\n')

final = 'ALL success rate: {:.4f}'.format(
    float(sum(success_count)) / sum(num_count))
print(final)
fw.write(final + '\n')
fw.close()

# ── Timing summary ────────────────────────────────────────────────────────────
t_tot = np.array(t_est_all) + np.array(t_ref_all)
print('\n── Timing summary (ms) ──────────────────────────────────────────')
print('{:<20} {:>8} {:>8} {:>8} {:>8}'.format('', 'median', 'mean', 'p95', 'max'))
for label, arr in [('Estimator', np.array(t_est_all)),
                   ('Refiner (x{})'.format(ITERATION), np.array(t_ref_all)),
                   ('Total', t_tot)]:
    print('{:<20} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}'.format(
        label, np.median(arr), np.mean(arr),
        np.percentile(arr, 95), np.max(arr)))
print('Achievable fps (median / p95): {:.1f} / {:.1f}'.format(
    1000/np.median(t_tot), 1000/np.percentile(t_tot, 95)))

# ── Plot ──────────────────────────────────────────────────────────────────────
threshold_mm = float(np.mean([d * 1000 for d in diameter]))
plot_title = ('DenseFusion LineMOD — full eval  |  '
              'success {:.1f}%  |  median {:.1f} ms ({:.1f} fps)').format(
    np.mean(suc_all) * 100, np.median(t_tot), 1000/np.median(t_tot))

plot_path = os.path.join(out_dir, 'realtime.png')
plot_path = os.path.join(out_dir, 'realtime.png')
timing_plots(t_est_all, t_ref_all, add_all, suc_all,
             threshold_mm, plot_title, plot_path)
