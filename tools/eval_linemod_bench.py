import _init_paths
import argparse
import os
import time
import numpy as np
import yaml
import copy
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import quaternion_matrix, quaternion_from_matrix

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',  type=str, default='', help='dataset root dir')
parser.add_argument('--model',         type=str, default='', help='resume PoseNet model')
parser.add_argument('--refine_model',  type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--verbose',       action='store_true',  help='print per-frame logs and save plot to disk')
opt = parser.parse_args()

def log(msg):
    if opt.verbose:
        print(msg)

# ── Model setup ───────────────────────────────────────────────────────────────
num_objects = 13
objlist     = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points  = 500
iteration   = 4
bs          = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir  = 'experiments/eval_result/linemod'

estimator = PoseNet(num_points=num_points, num_obj=num_objects).cuda()
refiner   = PoseRefineNet(num_points=num_points, num_obj=num_objects).cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

testdataset    = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list        = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion        = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file, Loader=yaml.SafeLoader)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)

success_count = [0 for _ in range(num_objects)]
num_count     = [0 for _ in range(num_objects)]

if opt.verbose:
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

# ── Per-frame timing ──────────────────────────────────────────────────────────
# Each frame records 3 durations (ms):
#   t_estimator : estimator forward pass only
#   t_refiner   : all refiner passes combined
#   t_total     : estimator + all refiner passes
t_estimator_all = []
t_refiner_all   = []
t_total_all     = []

# ── Eval loop ─────────────────────────────────────────────────────────────────
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data

    if len(points.size()) == 2:
        log('No.{0} NOT Pass! Lost detection!'.format(i))
        if opt.verbose:
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue

    points, choose, img, target, model_points, idx = (
        points.cuda(), choose.cuda(), img.cuda(),
        target.cuda(), model_points.cuda(), idx.cuda()
    )

    # ── Estimator ─────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

    torch.cuda.synchronize()
    t_est = (time.perf_counter() - t0) * 1000.0  # ms

    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

    # ── Refiner ───────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for ite in range(0, iteration):
        T = torch.from_numpy(my_t.astype(np.float32)).cuda() \
              .view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
        my_r = my_r_final
        my_t = my_t_final

    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t1) * 1000.0  # ms

    t_estimator_all.append(t_est)
    t_refiner_all.append(t_ref)
    t_total_all.append(t_est + t_ref)

    # ── ADD score ─────────────────────────────────────────────────────────────
    mp_np = model_points[0].cpu().detach().numpy()
    tg_np = target[0].cpu().detach().numpy()
    my_r_mat = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(mp_np, my_r_mat.T) + my_t

    if idx[0].item() in sym_list:
        pred_t_   = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target_t_ = torch.from_numpy(tg_np.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = torch.cdist(pred_t_.transpose(0,1).unsqueeze(0),
                           target_t_.transpose(0,1).unsqueeze(0)).argmin(dim=2)
        target_t_ = torch.index_select(target_t_, 1, inds.view(-1))
        dis = torch.mean(torch.norm(
            (pred_t_.transpose(1,0) - target_t_.transpose(1,0)), dim=1), dim=0).item()
    else:
        dis = float(np.mean(np.linalg.norm(pred - tg_np, axis=1)))

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        log('No.{0} Pass! Distance: {1}  |  t_est: {2:.1f}ms  t_ref: {3:.1f}ms  t_total: {4:.1f}ms'.format(
            i, dis, t_est, t_ref, t_est + t_ref))
        if opt.verbose:
            fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        log('No.{0} NOT Pass! Distance: {1}  |  t_est: {2:.1f}ms  t_ref: {3:.1f}ms  t_total: {4:.1f}ms'.format(
            i, dis, t_est, t_ref, t_est + t_ref))
        if opt.verbose:
            fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

# ── Results ───────────────────────────────────────────────────────────────────
for i in range(num_objects):
    msg = 'Object {0} success rate: {1}'.format(
        objlist[i], float(success_count[i]) / num_count[i])
    log(msg)
    if opt.verbose:
        fw.write(msg + '\n')

final_msg = 'ALL success rate: {0}'.format(
    float(sum(success_count)) / sum(num_count))
print(final_msg)
if opt.verbose:
    fw.write(final_msg + '\n')
    fw.close()

# ── Timing summary ────────────────────────────────────────────────────────────
t_est_arr = np.array(t_estimator_all)
t_ref_arr = np.array(t_refiner_all)
t_tot_arr = np.array(t_total_all)

print('\n── Timing summary (ms) ──────────────────────────────────────────')
print('{:<20} {:>8} {:>8} {:>8} {:>8}'.format('', 'median', 'mean', 'p95', 'max'))
for label, arr in [('Estimator', t_est_arr), ('Refiner (x{})'.format(iteration), t_ref_arr), ('Total', t_tot_arr)]:
    print('{:<20} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}'.format(
        label,
        np.median(arr), np.mean(arr),
        np.percentile(arr, 95), np.max(arr)))
print('─────────────────────────────────────────────────────────────────')
print('  30 fps target : {:.0f} ms/frame'.format(1000/30))
print('  60 fps target : {:.0f} ms/frame'.format(1000/60))
fps_median = 1000.0 / np.median(t_tot_arr)
fps_p95    = 1000.0 / np.percentile(t_tot_arr, 95)
print('  Achievable fps (median) : {:.1f} fps'.format(fps_median))
print('  Achievable fps (p95)    : {:.1f} fps'.format(fps_p95))

# ── Performance plot ──────────────────────────────────────────────────────────
# Panel 1 — per-frame total time (timeline)
#   Latency for every frame in evaluation order.
#   Reference lines at 33ms (30fps) and 16ms (60fps).
#
# Panel 2 — estimator vs refiner breakdown (stacked bar, binned)
#   Mean time split between estimator and refiner across frame buckets.
#   Shows which component dominates the budget.
#
# Panel 3 — total time distribution (histogram + CDF)
#   Full distribution of per-frame latency with fps reference lines.
#   CDF lets you read off: "X% of frames process within Y ms".

frames = np.arange(len(t_tot_arr))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('DenseFusion LineMOD — Real-time feasibility', fontsize=13)

# Panel 1 — per-frame latency timeline
axes[0].plot(frames, t_tot_arr, color='steelblue', linewidth=0.8, alpha=0.7, label='Total')
axes[0].axhline(1000/30, color='orange', linestyle='--', linewidth=1.5, label='30 fps (33 ms)')
axes[0].axhline(1000/60, color='red',    linestyle='--', linewidth=1.5, label='60 fps (16 ms)')
axes[0].set_xlabel('Frame index')
axes[0].set_ylabel('Time (ms)')
axes[0].set_title('Per-frame latency')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel 2 — stacked bar: estimator vs refiner (every 50 frames binned)
bin_size = max(1, len(t_tot_arr) // 20)
n_bins   = len(t_tot_arr) // bin_size
bin_est  = [np.mean(t_est_arr[k*bin_size:(k+1)*bin_size]) for k in range(n_bins)]
bin_ref  = [np.mean(t_ref_arr[k*bin_size:(k+1)*bin_size]) for k in range(n_bins)]
bin_x    = [k * bin_size for k in range(n_bins)]
axes[1].bar(bin_x, bin_est, width=bin_size*0.8, label='Estimator', color='steelblue', alpha=0.85)
axes[1].bar(bin_x, bin_ref, width=bin_size*0.8, bottom=bin_est, label='Refiner (x{})'.format(iteration),
            color='darkorange', alpha=0.85)
axes[1].axhline(1000/30, color='orange', linestyle='--', linewidth=1.5, label='30 fps')
axes[1].axhline(1000/60, color='red',    linestyle='--', linewidth=1.5, label='60 fps')
axes[1].set_xlabel('Frame index (binned)')
axes[1].set_ylabel('Mean time (ms)')
axes[1].set_title('Estimator vs Refiner breakdown')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3 — histogram + CDF
axes[3-1].hist(t_tot_arr, bins=40, color='steelblue', alpha=0.6,
               density=True, label='Histogram')
ax_cdf = axes[2].twinx()
sorted_t = np.sort(t_tot_arr)
cdf = np.arange(1, len(sorted_t)+1) / len(sorted_t)
ax_cdf.plot(sorted_t, cdf * 100, color='navy', linewidth=2, label='CDF')
ax_cdf.set_ylabel('Cumulative % of frames')
ax_cdf.set_ylim(0, 105)
axes[2].axvline(1000/30, color='orange', linestyle='--', linewidth=1.5, label='30 fps (33 ms)')
axes[2].axvline(1000/60, color='red',    linestyle='--', linewidth=1.5, label='60 fps (16 ms)')
axes[2].set_xlabel('Total time per frame (ms)')
axes[2].set_ylabel('Density')
axes[2].set_title('Latency distribution + CDF')
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax_cdf.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plot_path = '{0}/eval_realtime.png'.format(output_result_dir)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print('Plot saved: {}'.format(plot_path))

plt.close()