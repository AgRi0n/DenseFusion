"""
lib/linemod/inference.py — Model loading, GPU warmup, and inference pipeline.

Covers:
  - load_models()     : load PoseNet + PoseRefineNet from checkpoint paths
  - warmup_gpu()      : silent forward pass to initialise CUDA before timing
  - run_inference()   : estimator + iterative refiner, returns R, t and timing
"""

import time
import copy
import numpy as np
import torch

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from lib.linemod.config import NUM_POINTS, NUM_OBJS, ITERATION


def load_models(model_path, refine_model_path):
    """Load PoseNet and PoseRefineNet from checkpoint files onto CUDA.

    Returns
    -------
    estimator : PoseNet       (eval mode, on GPU)
    refiner   : PoseRefineNet (eval mode, on GPU)
    """
    estimator = PoseNet(num_points=NUM_POINTS, num_obj=NUM_OBJS).cuda()
    refiner   = PoseRefineNet(num_points=NUM_POINTS, num_obj=NUM_OBJS).cuda()
    estimator.load_state_dict(torch.load(model_path))
    refiner.load_state_dict(torch.load(refine_model_path))
    estimator.eval()
    refiner.eval()
    return estimator, refiner


def warmup_gpu(estimator):
    """Run a silent dummy forward pass to initialise CUDA kernels.

    CUDA allocates memory and JIT-compiles kernels on the first call.
    Without warmup, the first real frame appears 5-10x slower than steady state.
    Call this once after load_models() and before starting any timing measurement.
    """
    print('Warming up GPU...')
    with torch.no_grad():
        _img    = torch.zeros(1, 3, 80, 80).cuda()
        _cloud  = torch.zeros(1, NUM_POINTS, 3).cuda()
        _choose = torch.zeros(1, NUM_POINTS).long().cuda()
        _idx    = torch.zeros(1).long().cuda()
        estimator(_img, _cloud, _choose, _idx)
    torch.cuda.synchronize()
    print('Warmup done.')


def run_inference(estimator, refiner, img_t, cloud_t, choose_t, idx_t,
                  measure_time=False):
    """Run estimator + iterative refiner for one frame.

    Parameters
    ----------
    measure_time : bool
        If True, synchronise CUDA and record wall-clock time for estimator
        and refiner separately.

    Returns
    -------
    R_final : np.ndarray (3, 3)  — predicted rotation matrix
    t_final : np.ndarray (3,)    — predicted translation vector (metres)
    t_est   : float or None      — estimator time in ms (if measure_time)
    t_ref   : float or None      — refiner time in ms   (if measure_time)
    """
    # ── Estimator ─────────────────────────────────────────────────────────────
    if measure_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    with torch.no_grad():
        pred_r, pred_t, pred_c, emb = estimator(img_t, cloud_t, choose_t, idx_t)

    if measure_time:
        torch.cuda.synchronize()
        t_est = (time.perf_counter() - t0) * 1000.0
    else:
        t_est = None

    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, NUM_POINTS, 1)
    pred_c = pred_c.view(1, NUM_POINTS)
    _, which_max = torch.max(pred_c, 1)
    pred_t_flat  = pred_t.view(NUM_POINTS, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().numpy()
    my_t = (cloud_t.view(NUM_POINTS, 1, 3) + pred_t_flat)[which_max[0]].view(-1).cpu().numpy()

    # ── Refiner ───────────────────────────────────────────────────────────────
    if measure_time:
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    for _ in range(ITERATION):
        T = (torch.from_numpy(my_t.astype(np.float32)).cuda()
             .view(1, 3).repeat(NUM_POINTS, 1).contiguous().view(1, NUM_POINTS, 3))
        my_mat = quaternion_matrix(my_r)
        R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        with torch.no_grad():
            new_points    = torch.bmm((cloud_t - T), R).contiguous()
            pred_r2, pred_t2 = refiner(new_points, emb, idx_t)

        pred_r2 = pred_r2.view(1, 1, -1)
        pred_r2 = pred_r2 / torch.norm(pred_r2, dim=2).view(1, 1, 1)
        my_r2   = pred_r2.view(-1).cpu().numpy()
        my_t2   = pred_t2.view(-1).cpu().numpy()
        my_mat2 = quaternion_matrix(my_r2)
        my_mat2[0:3, 3] = my_t2

        my_mat_f = np.dot(my_mat, my_mat2)
        my_r_f   = copy.deepcopy(my_mat_f)
        my_r_f[0:3, 3] = 0
        my_r = quaternion_from_matrix(my_r_f, True)
        my_t = np.array([my_mat_f[0][3], my_mat_f[1][3], my_mat_f[2][3]])

    if measure_time:
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t1) * 1000.0
    else:
        t_ref = None

    R_final = quaternion_matrix(my_r)[:3, :3]
    return R_final, my_t, t_est, t_ref
