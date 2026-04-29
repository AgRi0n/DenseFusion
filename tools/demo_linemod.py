"""
demo_linemod.py — DenseFusion inference visualizer for LineMOD

Runs the model on a single image from the dataset and saves a visualization
showing the RGB input, the segmentation mask, and the predicted 6D pose
(3D model reprojected onto the image).

Usage (from the repo root, inside the container):
    python3 tools/demo_linemod.py \
        --dataset_root ./datasets/linemod/Linemod_preprocessed \
        --model trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
        --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
        --obj 1 \
        --idx 0 \
        --output demo_output.png

Arguments:
    --obj    Object ID to test (1,2,4,5,6,8,9,10,11,12,13,14,15)
    --idx    Index within that object's test set (0, 1, 2 ...)
    --output Output image path (saved to disk, viewable outside the container)
"""

import _init_paths
import argparse
import os
import copy
import random
import numpy as np
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy.ma as ma
import cv2

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from datasets.linemod.dataset import ply_vtx, get_bbox, mask_to_bbox

# ── Camera intrinsics (LineMOD) ───────────────────────────────────────────────
CAM_CX = 325.26110
CAM_CY = 242.04899
CAM_FX = 572.41140
CAM_FY = 573.57043

BORDER_LIST = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
XMAP = np.array([[j for i in range(640)] for j in range(480)])
YMAP = np.array([[i for i in range(640)] for j in range(480)])
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

NUM_POINTS = 500
NUM_POINTS_MESH = 500
ITERATION = 4
SYM_LIST = [7, 8]  # object indices (not IDs) that are symmetric


def project_points(pts_3d, R, t, fx, fy, cx, cy):
    """Project 3D points onto the image plane."""
    pts = (R @ pts_3d.T).T + t  # [N, 3]
    x = (pts[:, 0] / pts[:, 2]) * fx + cx
    y = (pts[:, 1] / pts[:, 2]) * fy + cy
    return np.stack([x, y], axis=1)


def load_image(dataset_root, obj, frame_idx):
    """Load RGB, depth, and segnet mask for a given object and frame index."""
    test_file = open('{0}/data/{1}/test.txt'.format(dataset_root, '%02d' % obj))
    lines = [l.strip() for l in test_file.readlines() if l.strip()]
    test_file.close()

    frame_name = lines[frame_idx]

    rgb_path   = '{0}/data/{1}/rgb/{2}.png'.format(dataset_root, '%02d' % obj, frame_name)
    depth_path = '{0}/data/{1}/depth/{2}.png'.format(dataset_root, '%02d' % obj, frame_name)
    label_path = '{0}/segnet_results/{1}_label/{2}_label.png'.format(dataset_root, '%02d' % obj, frame_name)

    img   = Image.open(rgb_path)
    depth = np.array(Image.open(depth_path))
    label = np.array(Image.open(label_path))

    return img, depth, label, frame_name


def prepare_input(img, depth, label):
    """Replicate dataset preprocessing to produce model input tensors."""
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    mask = mask_label * mask_depth

    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

    img_arr = np.array(img)[:, :, :3]
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_crop = img_arr[:, rmin:rmax, cmin:cmax]

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

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked  = XMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked  = YMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / 1000.0
    pt0 = (ymap_masked - CAM_CX) * pt2 / CAM_FX
    pt1 = (xmap_masked - CAM_CY) * pt2 / CAM_FY
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    img_tensor   = NORM(torch.from_numpy(img_crop.astype(np.float32)))
    cloud_tensor = torch.from_numpy(cloud.astype(np.float32))
    choose_tensor = torch.LongTensor(np.array([choose]).astype(np.int32))

    return (img_tensor.cuda().unsqueeze(0),
            cloud_tensor.cuda().unsqueeze(0),
            choose_tensor.cuda(),
            (rmin, rmax, cmin, cmax),
            mask_label)


def run_inference(estimator, refiner, img_t, cloud_t, choose_t, idx_t):
    with torch.no_grad():
        pred_r, pred_t, pred_c, emb = estimator(img_t, cloud_t, choose_t, idx_t)

    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, NUM_POINTS, 1)
    pred_c = pred_c.view(1, NUM_POINTS)
    _, which_max = torch.max(pred_c, 1)
    pred_t_flat = pred_t.view(NUM_POINTS, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().numpy()
    my_t = (cloud_t.view(NUM_POINTS, 1, 3) + pred_t_flat)[which_max[0]].view(-1).cpu().numpy()

    for _ in range(ITERATION):
        T = torch.from_numpy(my_t.astype(np.float32)).cuda().view(1, 3).repeat(NUM_POINTS, 1).contiguous().view(1, NUM_POINTS, 3)
        my_mat = quaternion_matrix(my_r)
        R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        with torch.no_grad():
            new_points = torch.bmm((cloud_t - T), R).contiguous()
            pred_r2, pred_t2 = refiner(new_points, emb, idx_t)

        pred_r2 = pred_r2.view(1, 1, -1)
        pred_r2 = pred_r2 / torch.norm(pred_r2, dim=2).view(1, 1, 1)
        my_r2 = pred_r2.view(-1).cpu().numpy()
        my_t2 = pred_t2.view(-1).cpu().numpy()
        my_mat2 = quaternion_matrix(my_r2)
        my_mat2[0:3, 3] = my_t2

        my_mat_final = np.dot(my_mat, my_mat2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r = quaternion_from_matrix(my_r_final, True)
        my_t = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

    R_final = quaternion_matrix(my_r)[:3, :3]
    return R_final, my_t


def load_ground_truth(dataset_root, obj, frame_name):
    """Load the ground truth rotation and translation from gt.yml.

    gt.yml stores one entry per frame, keyed by frame index (int).
    Each entry contains:
      cam_R_m2c — flattened 3x3 rotation matrix (model-to-camera)
      cam_t_m2c — translation vector in millimetres

    Source: dataset.py lines 130-131, same loading logic.
    """
    gt_path = '{0}/data/{1}/gt.yml'.format(dataset_root, '%02d' % obj)
    with open(gt_path, 'r') as f:
        gt = yaml.load(f, Loader=yaml.SafeLoader)

    rank = int(frame_name)
    # Object 2 has multiple annotated instances per frame; take the one with obj_id == 2
    if obj == 2:
        meta = next(m for m in gt[rank] if m['obj_id'] == 2)
    else:
        meta = gt[rank][0]

    R_gt = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    t_gt = np.array(meta['cam_t_m2c']) / 1000.0  # mm → m
    return R_gt, t_gt


def compute_add(R_pred, t_pred, R_gt, t_gt, model_pts, obj_idx, diameter_mm):
    """Compute the ADD (or ADD-S for symmetric objects) metric.

    ADD — Average Distance of Model Points:
        For each point p in the 3D model, compute the distance between
        its position under the predicted pose and its position under the
        ground truth pose, then average across all points.

            ADD = mean_p || (R_pred @ p + t_pred) - (R_gt @ p + t_gt) ||

    ADD-S (symmetric variant):
        For symmetric objects the correspondence is ambiguous, so instead
        of comparing point-to-point we compare each predicted point to its
        nearest neighbour in the ground truth point cloud.

            ADD-S = mean_p  min_q || (R_pred @ p + t_pred) - (R_gt @ q + t_gt) ||

    Threshold: a pose is considered correct if ADD (or ADD-S) < 10% of
    the object's diameter. This is the standard LineMOD evaluation protocol
    used in eval_linemod.py line 132.

    Source: eval_linemod.py lines 119-132 + models_info.yml for diameter.
    """
    pts = model_pts / 1000.0  # mm → m

    pred_cloud = (R_pred @ pts.T).T + t_pred  # [N, 3]
    gt_cloud   = (R_gt   @ pts.T).T + t_gt    # [N, 3]

    if obj_idx in SYM_LIST:
        # ADD-S: nearest-neighbour matching via cdist
        from scipy.spatial.distance import cdist
        dists = cdist(pred_cloud, gt_cloud)
        add = dists.min(axis=1).mean()
        metric_name = 'ADD-S'
    else:
        # ADD: point-to-point
        add = np.linalg.norm(pred_cloud - gt_cloud, axis=1).mean()
        metric_name = 'ADD'

    diameter_m    = diameter_mm / 1000.0
    threshold     = diameter_m * 0.1
    success       = add < threshold

    return {
        'metric':       metric_name,
        'value_mm':     add * 1000.0,
        'threshold_mm': threshold * 1000.0,
        'diameter_mm':  diameter_mm,
        'success':      success,
    }


def draw_axes(ax, R, t, length_m=0.05):
    """Draw a 3D pose as a projected XYZ tricolour frame on a matplotlib axis.

    The three axes of the object coordinate frame are projected onto the image
    plane using the same pinhole model as project_points(). Each axis is drawn
    as an arrow from the object centre (t) to (t + R[:,i] * length_m):
        X → red   Y → green   Z → blue

    length_m controls the visual size of the frame in metres; tune it to the
    scale of the object if axes appear too large or too small.
    """
    origin = project_points(np.zeros((1, 3)), R, t, CAM_FX, CAM_FY, CAM_CX, CAM_CY)[0]
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    for i, (color, label) in enumerate(zip(colors, labels)):
        tip_3d = (R[:, i] * length_m).reshape(1, 3)
        tip_2d = project_points(tip_3d, R, t, CAM_FX, CAM_FY, CAM_CX, CAM_CY)[0]
        # Arrow from origin to tip
        ax.annotate('', xy=tip_2d, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(tip_2d[0], tip_2d[1], label, color=color, fontsize=9, fontweight='bold')


def _pose_panel(ax, proj_pred, proj_gt, R_pred, t_pred, R_gt, t_gt,
                background, title, mask_label=None):
    """Shared helper: draw one predicted-vs-GT panel with axes frames.

    background  — RGB image array, or None for a black background (point-cloud only mode)
    mask_label  — if provided, the RGB image is masked so only the object region is shown
    """
    def in_frame(proj):
        return ((proj[:, 0] >= 0) & (proj[:, 0] < 640) &
                (proj[:, 1] >= 0) & (proj[:, 1] < 480))

    if background is None:
        # Black background — point cloud only
        ax.set_facecolor('black')
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # inverted Y to match image coords
    else:
        img_display = background.copy()
        if mask_label is not None:
            # Apply segmentation mask: zero out pixels where mask == 0
            seg = (mask_label == 255).astype(np.uint8)
            img_display = img_display * seg[:, :, np.newaxis]
        ax.imshow(img_display)

    valid_gt   = in_frame(proj_gt)
    valid_pred = in_frame(proj_pred)
    ax.scatter(proj_gt[valid_gt, 0],     proj_gt[valid_gt, 1],
               s=1.5, c='lime', alpha=0.5, linewidths=0, label='Ground truth')
    ax.scatter(proj_pred[valid_pred, 0], proj_pred[valid_pred, 1],
               s=1.5, c='red',  alpha=0.5, linewidths=0, label='Predicted')

    # 6D pose frames — predicted (solid) and ground truth (dashed via double draw)
    draw_axes(ax, R_pred, t_pred)
    draw_axes(ax, R_gt,   t_gt)   # GT axes drawn identically; they overlap if pose is perfect

    ax.legend(loc='upper right', fontsize=8, markerscale=6,
              facecolor='#111111', labelcolor='white')
    ax.set_title(title, color='white' if background is None else 'black')
    ax.axis('off')


def visualize(img, depth, mask_label, bbox, model_pts, R, t, R_gt, t_gt,
              obj, frame_name, add_result, output_path):
    """Produce two output images:

    1. <output_path>          — overview: RGB input | segmentation mask | masked RGB overlay
    2. <stem>_pointcloud.png  — point-cloud only view on black background

    Each predicted-vs-GT panel contains:
      - Ground truth point cloud (green)
      - Predicted point cloud (red)
      - Predicted 6D pose frame (RGB tricolour arrows)
      - Ground truth 6D pose frame (RGB tricolour arrows, overlaps predicted if pose is good)
    """
    rmin, rmax, cmin, cmax = bbox
    img_arr = np.array(img)

    # Subsample model points for clean projection
    sample_idx = np.random.choice(len(model_pts), min(500, len(model_pts)), replace=False)
    pts_sample = model_pts[sample_idx] / 1000.0  # mm → m

    proj_pred = project_points(pts_sample, R,    t,    CAM_FX, CAM_FY, CAM_CX, CAM_CY)
    proj_gt   = project_points(pts_sample, R_gt, t_gt, CAM_FX, CAM_FY, CAM_CX, CAM_CY)

    # Shared title info
    status = '✓ PASS' if add_result['success'] else '✗ FAIL'
    add_str = '{}  {} = {:.2f} mm  (threshold {:.2f} mm)'.format(
        status, add_result['metric'], add_result['value_mm'], add_result['threshold_mm'])
    title_color = 'green' if add_result['success'] else 'red'
    sup = 'DenseFusion — Object {:02d}, Frame {}    {}'.format(obj, frame_name, add_str)

    # ── Figure 1: overview (RGB | mask | masked overlay) ──────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig1.suptitle(sup, fontsize=11, color=title_color)

    # Panel 1 — RGB + bounding box
    axes1[0].imshow(img_arr)
    rect = patches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                              linewidth=2, edgecolor='lime', facecolor='none')
    axes1[0].add_patch(rect)
    axes1[0].set_title('RGB input + detection bbox')
    axes1[0].axis('off')

    # Panel 2 — segmentation mask
    axes1[1].imshow(mask_label, cmap='gray')
    axes1[1].set_title('Segmentation mask (SegNet output)')
    axes1[1].axis('off')

    # Panel 3 — masked RGB + point clouds + pose axes
    _pose_panel(axes1[2], proj_pred, proj_gt, R, t, R_gt, t_gt,
                background=img_arr,
                title='Predicted (red) vs GT (green) — RGB',
                mask_label=None)

    fig1.tight_layout()
    fig1.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('Saved: {}'.format(output_path))

    # ── Figure 2: point-cloud only on black background ─────────────────────────
    stem, ext = os.path.splitext(output_path)
    pc_path = '{}_pointcloud{}'.format(stem, ext)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig2.suptitle(sup, fontsize=11, color=title_color)

    _pose_panel(axes2[0], proj_pred, proj_gt, R, t, R_gt, t_gt,
                background=None,
                title='Point cloud — full frame')

    # Zoom into the object bounding box for the second panel
    _pose_panel(axes2[1], proj_pred, proj_gt, R, t, R_gt, t_gt,
                background=None,
                title='Point cloud — object crop')
    axes2[1].set_xlim(cmin, cmax)
    axes2[1].set_ylim(rmax, rmin)  # inverted Y

    fig2.tight_layout()
    fig2.savefig(pc_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig2)
    print('Saved: {}'.format(pc_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',  type=str, required=True)
    parser.add_argument('--model',         type=str, required=True)
    parser.add_argument('--refine_model',  type=str, required=True)
    parser.add_argument('--obj',           type=int, default=1,
                        help='Object ID (1,2,4,5,6,8,9,10,11,12,13,14,15)')
    parser.add_argument('--idx',           type=int, default=0,
                        help='Frame index within the object test set')
    parser.add_argument('--output',        type=str, default='demo_output.png')
    opt = parser.parse_args()

    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    assert opt.obj in objlist, 'Invalid object ID. Choose from: {}'.format(objlist)

    # Load model
    num_objects = 13
    estimator = PoseNet(num_points=NUM_POINTS, num_obj=num_objects).cuda()
    refiner   = PoseRefineNet(num_points=NUM_POINTS, num_obj=num_objects).cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    # Load 3D model points
    model_pts = ply_vtx('{0}/models/obj_{1}.ply'.format(opt.dataset_root, '%02d' % opt.obj))

    # Load image
    img, depth, label, frame_name = load_image(opt.dataset_root, opt.obj, opt.idx)
    print('Loaded: obj={}, frame={}'.format(opt.obj, frame_name))

    # Prepare input tensors
    result = prepare_input(img, depth, label)
    if result is None:
        print('ERROR: empty mask for this frame, try a different --idx')
        return
    img_t, cloud_t, choose_t, bbox, mask_label = result

    obj_idx = objlist.index(opt.obj)
    idx_t = torch.LongTensor([obj_idx]).cuda()

    # Run inference
    R, t = run_inference(estimator, refiner, img_t, cloud_t, choose_t, idx_t)
    print('R:\n{}'.format(np.round(R, 4)))
    print('t: {}'.format(np.round(t, 4)))

    # Load ground truth pose
    R_gt, t_gt = load_ground_truth(opt.dataset_root, opt.obj, frame_name)
    print('R_gt:\n{}'.format(np.round(R_gt, 4)))
    print('t_gt: {}'.format(np.round(t_gt, 4)))

    # Load object diameter from models_info.yml (used for ADD threshold)
    models_info_path = '{}/dataset_config/models_info.yml'.format('datasets/linemod')
    with open(models_info_path, 'r') as f:
        models_info = yaml.load(f, Loader=yaml.SafeLoader)
    diameter_mm = models_info[opt.obj]['diameter']

    # Compute ADD / ADD-S
    add_result = compute_add(R, t, R_gt, t_gt, model_pts, obj_idx, diameter_mm)
    print('\n── ADD Result ──────────────────────────────')
    print('Metric:    {}'.format(add_result['metric']))
    print('Value:     {:.2f} mm'.format(add_result['value_mm']))
    print('Threshold: {:.2f} mm  (10% of {:.1f} mm diameter)'.format(
        add_result['threshold_mm'], add_result['diameter_mm']))
    print('Result:    {}'.format('PASS ✓' if add_result['success'] else 'FAIL ✗'))
    print('────────────────────────────────────────────\n')

    # Visualize
    visualize(img, depth, mask_label, bbox, model_pts, R, t, R_gt, t_gt,
              opt.obj, frame_name, add_result, opt.output)


if __name__ == '__main__':
    main()