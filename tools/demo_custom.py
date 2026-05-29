"""
demo_custom.py — DenseFusion inference on a single custom RGB-D frame.

Required:
    --rgb   <path>   RGB image (.png)
    --npz   <path>   Aligned depth file (.npz)
    --mask  <path>   Binary object mask (.png, 255 = object)
    --ply   <path>   3D model (.ply, vertices in mm)

Usage:
    python3 tools/demo_custom.py \
        --rgb   datasets/custom/.../RGB_img-42.png \
        --npz   datasets/custom/.../DEPTH_aligned_to_RGB_ahat-42.npz \
        --mask  datasets/custom/.../Masks/DEPTH_aligned_to_RGB_ahat-42-MASK.png \
        --ply   datasets/custom/models/obj_01.ply
"""

import _init_paths, argparse, os, re
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from PIL import Image
import torchvision.transforms as transforms

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from lib.linemod.inference import run_inference
from lib.linemod.visualization import bbox_3d_corners
from lib.custom import results as pose_results
from datasets.linemod.dataset import get_bbox, mask_to_bbox, ply_vtx

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--rgb',          required=True)
p.add_argument('--npz',          required=True)
p.add_argument('--mask',         required=True)
p.add_argument('--ply',          required=True)
p.add_argument('--model',        required=True)
p.add_argument('--refine_model', required=True)
p.add_argument('--obj_id',       type=int,   default=0)
p.add_argument('--num_obj',      type=int,   default=2)
p.add_argument('--profile',      default='hololens2_ahat')
p.add_argument('--camera_profiles', default='tools/config/camera_profiles.yaml')
p.add_argument('--output_dir',   default='demo_out/custom')
opt = p.parse_args()

NUM_POINTS = 500
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── Camera intrinsics ─────────────────────────────────────────────────────────
with open(opt.camera_profiles) as f:
    prof = yaml.safe_load(f)['profiles'][opt.profile]

npz = np.load(opt.npz, allow_pickle=True)

def _intrinsics(prof, npz):
    if prof.get('cam_fx') is not None:
        return float(prof['cam_fx']), float(prof['cam_fy']), float(prof['cam_cx']), float(prof['cam_cy'])
    key = prof.get('npz_intrinsics_key')
    if key and key in npz:
        K = npz[key][:3, :3]
        return float(K[0,0]), float(K[1,1]), float(K[2,0]), float(K[2,1])
    raise RuntimeError('Cannot determine camera intrinsics from profile or NPZ.')

cam_fx, cam_fy, cam_cx, cam_cy = _intrinsics(prof, npz)

# ── Load inputs ───────────────────────────────────────────────────────────────
img_arr = np.array(Image.open(opt.rgb))[:, :, :3]
img_h, img_w = img_arr.shape[:2]

depth = npz[prof.get('npz_depth_key', 'depth')].astype(np.float32)
if not prof.get('depth_in_metres', True):
    depth = depth / 1000.0

raw_mask = np.array(Image.open(opt.mask))
if raw_mask.ndim == 3:
    label = ((raw_mask[:,:,0]==255)&(raw_mask[:,:,1]==255)&(raw_mask[:,:,2]==255)).astype(np.uint8)*255
else:
    label = raw_mask.astype(np.uint8)

# ── Prepare tensors ───────────────────────────────────────────────────────────
mask_label = (label == 255)
mask       = mask_label & (depth > 0)
rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
if len(choose) == 0:
    print('ERROR: empty mask after combining with valid depth.'); exit(1)
choose = choose[:NUM_POINTS] if len(choose) > NUM_POINTS else np.pad(choose, (0, NUM_POINTS-len(choose)), 'wrap')
if len(choose) > NUM_POINTS:
    idx = np.zeros(len(choose), dtype=int); idx[:NUM_POINTS] = 1
    np.random.shuffle(idx); choose = choose[idx.nonzero()]

xmap = np.tile(np.arange(img_h)[:,None], (1, img_w)).astype(np.float32)
ymap = np.tile(np.arange(img_w)[None,:], (img_h, 1)).astype(np.float32)

d  = depth[rmin:rmax, cmin:cmax].flatten()[choose][:,None]
xm = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:,None]
ym = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:,None]
cloud = np.concatenate([(ym-cam_cx)*d/cam_fx, (xm-cam_cy)*d/cam_fy, d], axis=1).astype(np.float32)

img_crop = np.transpose(img_arr, (2,0,1))[:, rmin:rmax, cmin:cmax]
img_t    = NORM(torch.from_numpy(img_crop.astype(np.float32))).cuda().unsqueeze(0)
cloud_t  = torch.from_numpy(cloud).cuda().unsqueeze(0)
choose_t = torch.LongTensor(np.array([choose]).astype(np.int32)).cuda()
idx_t    = torch.LongTensor([opt.obj_id]).cuda()

# ── Inference ─────────────────────────────────────────────────────────────────
estimator = PoseNet(num_points=NUM_POINTS, num_obj=opt.num_obj).cuda()
refiner   = PoseRefineNet(num_points=NUM_POINTS, num_obj=opt.num_obj).cuda()
estimator.load_state_dict(torch.load(opt.model)); estimator.eval()
refiner.load_state_dict(torch.load(opt.refine_model)); refiner.eval()

R, t, _, _ = run_inference(estimator, refiner, img_t, cloud_t, choose_t, idx_t)
print(f'\nR:\n{np.round(R,4)}\nt: {np.round(t,4)} m\n')

# ── 3D model ──────────────────────────────────────────────────────────────────
model_pts  = ply_vtx(opt.ply)
sample_idx = np.random.choice(len(model_pts), min(500, len(model_pts)), replace=False)
pts_s      = model_pts[sample_idx] / 1000.0
bbox_3d_m  = bbox_3d_corners(model_pts) / 1000.0
_EDGES     = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def _project(pts, R_mat, t_vec):
    p = (R_mat @ pts.T).T + t_vec
    return np.stack([(p[:,0]/p[:,2])*cam_fx+cam_cx, (p[:,1]/p[:,2])*cam_fy+cam_cy], axis=1)

def _bbox3d(ax, corners, R_mat, t_vec, color, lw=2.0):
    proj = _project(corners, R_mat, t_vec)
    for i,j in _EDGES:
        ax.plot([proj[i,0],proj[j,0]], [proj[i,1],proj[j,1]], color=color, lw=lw, alpha=0.9)

proj_pred = _project(pts_s, R, t)
in_frame  = (proj_pred[:,0]>=0)&(proj_pred[:,0]<img_w)&(proj_pred[:,1]>=0)&(proj_pred[:,1]<img_h)

# ── Parse IDs ─────────────────────────────────────────────────────────────────
frame_id = int(m.group(1)) if (m := re.search(r'[_\-](\d+)\.npz$', os.path.basename(opt.npz))) else 0
mask_id  = int(m.group(1)) if (m := re.search(r'[_\-](\d+)[^0-9]*\.png$', os.path.basename(opt.mask))) else 0
os.makedirs(opt.output_dir, exist_ok=True)
title = f'Frame {frame_id:04d}  |  mask_id={mask_id}  |  obj_id={opt.obj_id}'

# ── Fig 1: Overview — full image + 2D bbox + predicted 3D bbox ───────────────
fig, ax = plt.subplots(figsize=(10,6), facecolor='white')
ax.imshow(img_arr)
ax.add_patch(patches.Rectangle((cmin,rmin), cmax-cmin, rmax-rmin,
             lw=2, edgecolor='lime', facecolor='none', linestyle='--'))
_bbox3d(ax, bbox_3d_m, R, t, color='red')
ax.set(xlim=(0,img_w), ylim=(img_h,0), title=f'{title}\n2D bbox (lime) + predicted 3D bbox (red)')
ax.axis('off'); fig.tight_layout()
p1 = os.path.join(opt.output_dir, f'{frame_id:04d}_overview.png')
fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {p1}')

# ── Fig 2: Crop — masked object + point cloud + 3D bbox ──────────────────────
seg = mask_label.astype(np.float32)
isolated = (img_arr.astype(np.float32)*seg[:,:,None] + 255*(1-seg[:,:,None])).astype(np.uint8)
fig, ax = plt.subplots(figsize=(6,6), facecolor='white')
ax.imshow(isolated, extent=[0,img_w,img_h,0])
ax.scatter(proj_pred[in_frame,0], proj_pred[in_frame,1], s=3, c='red', alpha=0.75, linewidths=0, label='Predicted')
_bbox3d(ax, bbox_3d_m, R, t, color='red')
ax.set(xlim=(cmin,cmax), ylim=(rmax,rmin), title=f'{title}\nCrop — point cloud + 3D bbox')
ax.legend(loc='upper right', fontsize=8, markerscale=4, facecolor='white', edgecolor='gray')
ax.axis('off'); fig.tight_layout()
p2 = os.path.join(opt.output_dir, f'{frame_id:04d}_crop.png')
fig.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f'Saved: {p2}')

# ── JSON results ──────────────────────────────────────────────────────────────
json_path = os.path.join(opt.output_dir, 'results.json')
entry = pose_results.build_entry(
    frame_id=frame_id, mask_id=mask_id, obj_id=opt.obj_id,
    R=R, t=t,
    rgb_path=opt.rgb, npz_path=opt.npz, mask_path=opt.mask, ply_path=opt.ply,
    camera_profile=opt.profile,
    cam_fx=cam_fx, cam_fy=cam_fy, cam_cx=cam_cx, cam_cy=cam_cy,
)
pose_results.upsert(json_path, entry)
print(f'Pose saved: {json_path}')
