"""
lib/linemod/metrics.py — Pose accuracy metrics for LineMOD evaluation.

Implements:
  - compute_add()          : ADD or ADD-S distance (standard LineMOD metric)
  - compute_pose_metrics() : ADD/ADD-S + geodesic rotation error + Euclidean translation error
"""

import numpy as np
from scipy.spatial.distance import cdist

from lib.linemod.config import SYM_LIST


def compute_add(R_pred, t_pred, R_gt, t_gt, model_pts, obj_idx):
    """Compute ADD (or ADD-S for symmetric objects) in metres.

    ADD  = mean_p || (R_pred @ p + t_pred) - (R_gt @ p + t_gt) ||
    ADD-S = mean_p  min_q || (R_pred @ p + t_pred) - (R_gt @ q + t_gt) ||

    Parameters
    ----------
    model_pts : np.ndarray [N, 3] in mm — converted to metres internally.
    obj_idx   : int — index in OBJLIST (not the object ID).

    Returns
    -------
    add   : float, distance in metres
    name  : str, 'ADD' or 'ADD-S'
    """
    pts = model_pts / 1000.0
    pc  = (R_pred @ pts.T).T + t_pred
    gc  = (R_gt   @ pts.T).T + t_gt

    if obj_idx in SYM_LIST:
        add  = cdist(pc, gc).min(axis=1).mean()
        name = 'ADD-S'
    else:
        add  = np.linalg.norm(pc - gc, axis=1).mean()
        name = 'ADD'
    return add, name


def compute_pose_metrics(R_pred, t_pred, R_gt, t_gt, model_pts, obj_idx, diameter_mm):
    """Compute three complementary pose error metrics.

    1. ADD / ADD-S — standard LineMOD accuracy metric (threshold = 10% diameter)
    2. Geodesic    — angular distance on SO(3) between the two rotations (degrees)
    3. Euclidean   — L2 distance between translation vectors (mm)

    For symmetric objects, the geodesic is minimised over rotations around the
    symmetry axis (Z in model space), sampled at 1° resolution.

    Returns
    -------
    dict with keys:
        add_metric, add_value_mm, add_threshold_mm, diameter_mm, add_success,
        geodesic_deg, euclidean_mm
    """
    add, metric_name = compute_add(R_pred, t_pred, R_gt, t_gt, model_pts, obj_idx)

    diameter_m  = diameter_mm / 1000.0
    threshold   = diameter_m * 0.1
    add_success = add < threshold

    # ── Geodesic rotation error ───────────────────────────────────────────────
    if obj_idx in SYM_LIST:
        angles  = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        min_geo = np.inf
        for a in angles:
            ca, sa = np.cos(a), np.sin(a)
            Rz     = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
            R_rel  = R_pred.T @ (R_gt @ Rz)
            trace  = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            angle  = np.degrees(np.arccos(trace))
            if angle < min_geo:
                min_geo = angle
        geodesic_deg = min_geo
    else:
        R_rel        = R_pred.T @ R_gt
        trace        = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        geodesic_deg = np.degrees(np.arccos(trace))

    # ── Euclidean translation error ───────────────────────────────────────────
    euclidean_mm = np.linalg.norm(t_pred - t_gt) * 1000.0

    return {
        'add_metric':       metric_name,
        'add_value_mm':     add * 1000.0,
        'add_threshold_mm': threshold * 1000.0,
        'diameter_mm':      diameter_mm,
        'add_success':      add_success,
        'geodesic_deg':     geodesic_deg,
        'euclidean_mm':     euclidean_mm,
    }
