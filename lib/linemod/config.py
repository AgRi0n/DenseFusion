"""
lib/linemod/config.py — Shared constants for LineMOD dataset and inference.

Single source of truth for camera intrinsics, pixel coordinate maps,
object list, symmetry list, and model hyperparameters.
All tools import from here; nothing is duplicated across scripts.
"""

import numpy as np
import torchvision.transforms as transforms

# ── Camera intrinsics (Primesense, calibrated with OpenCV) ────────────────────
CAM_CX = 325.26110
CAM_CY = 242.04899
CAM_FX = 572.41140
CAM_FY = 573.57043

IMG_W = 640
IMG_H = 480

# ── Pixel coordinate grids — built once, reused everywhere ───────────────────
# XMAP[v, u] = v  (row index)
# YMAP[v, u] = u  (col index)
# Named following dataset.py convention (XMAP=rows, YMAP=cols).
XMAP = np.array([[j for i in range(IMG_W)] for j in range(IMG_H)])
YMAP = np.array([[i for i in range(IMG_W)] for j in range(IMG_H)])

# ── ImageNet normalisation used by the ResNet backbone ────────────────────────
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

# ── Object and symmetry lists ─────────────────────────────────────────────────
OBJLIST  = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJS = len(OBJLIST)

# Indices into OBJLIST (not object IDs) for rotationally symmetric objects.
# Index 7 → obj_id 10, Index 8 → obj_id 11.
SYM_LIST = [7, 8]

# ── Model hyperparameters ─────────────────────────────────────────────────────
NUM_POINTS      = 500   # input point cloud size
NUM_POINTS_MESH = 500   # model mesh points sampled for loss / metrics
ITERATION       = 4     # number of iterative refinement passes
