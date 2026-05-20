"""
lib/linemod/visualization.py — Visualisation helpers for pose estimation results.

Covers:
  - project_points()  : 3D → 2D projection using LineMOD camera convention
  - bbox_3d_corners() : 8-corner axis-aligned bbox in model frame
  - draw_axes()       : matplotlib tricolour XYZ pose frame
  - draw_bbox_3d()    : matplotlib 12-edge 3D bbox overlay
  - pose_panel()      : single predicted-vs-GT matplotlib panel
  - timing_plots()    : 4-panel performance figure (latency/ADD/fps/breakdown)
  - draw_frame_cv2()  : OpenCV overlay for video frames (live feed emulation)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2

from lib.linemod.config import CAM_CX, CAM_CY, CAM_FX, CAM_FY, ITERATION


# ── Projection ────────────────────────────────────────────────────────────────

def project_points(pts_3d, R, t):
    """Project 3D model points onto the image plane (LineMOD camera convention).

    LineMOD's back-projection uses XMAP for rows and YMAP for columns,
    with cx/fx applied to columns and cy/fy applied to rows. We mirror
    that convention here so projected points align with the dataset images.

        u (col) = (pts[:,0] / Z) * FX + CX
        v (row) = (pts[:,1] / Z) * FY + CY

    Returns
    -------
    np.ndarray [N, 2] — (u, v) pixel coordinates
    """
    pts = (R @ pts_3d.T).T + t
    u   = (pts[:, 0] / pts[:, 2]) * CAM_FX + CAM_CX
    v   = (pts[:, 1] / pts[:, 2]) * CAM_FY + CAM_CY
    return np.stack([u, v], axis=1)


def _in_frame(proj, w=640, h=480):
    return ((proj[:, 0] >= 0) & (proj[:, 0] < w) &
            (proj[:, 1] >= 0) & (proj[:, 1] < h))


# ── 3D bounding box ──────────────────────────────────────────────────────────

# 12 edges of an axis-aligned cuboid given the 8 corners produced by
# bbox_3d_corners(): indices 0–3 form the bottom face, 4–7 the top face,
# matched in lockstep so 0↔4, 1↔5, 2↔6, 3↔7 are the vertical struts.
_BBOX_3D_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical struts
]


def bbox_3d_corners(model_pts):
    """Eight axis-aligned bounding-box corners derived from a model point cloud.

    Parameters
    ----------
    model_pts : np.ndarray (N, 3)
        Model vertices in any consistent unit; corners are returned in the same.

    Returns
    -------
    np.ndarray (8, 3)
    """
    mn = model_pts.min(axis=0)
    mx = model_pts.max(axis=0)
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ])


def draw_bbox_3d(ax, corners_3d, R, t,
                 color='lime', lw=1.5, alpha=0.9, zorder=4):
    """Project an axis-aligned 3D bbox under pose (R, t) and draw its 12 edges.

    corners_3d : np.ndarray (8, 3) in metres, expressed in the model frame.
    """
    proj = project_points(corners_3d, R, t)
    for i, j in _BBOX_3D_EDGES:
        ax.plot([proj[i, 0], proj[j, 0]],
                [proj[i, 1], proj[j, 1]],
                color=color, lw=lw, alpha=alpha, zorder=zorder)


# ── Matplotlib helpers ────────────────────────────────────────────────────────

def draw_axes(ax, R, t, length_m=0.05, scale=1.0, frame='camera'):
    """Draw a projected tricolour pose frame on a matplotlib axis.

    Two reference frames are supported:

      frame='camera' (default)
        The arrows follow the camera *optical* frame: +x to image right,
        +y down, +z into the scene (depth, standard photography convention).
        The arrows are anchored at the object centre t but their orientation
        does not depend on R, so they keep a constant attitude across frames
        while the anchor (O) moves with the object.

      frame='object'
        The arrows follow the object's own frame, rotated by R and
        translated by t — i.e. the classical pose-estimation visualisation
        of the object's local coordinate system.

    Colour and naming conventions (both modes):
        x → red    y → green    z → blue
        The origin is marked with a white-filled dot labelled "O".

    Parameters
    ----------
    ax       : matplotlib axis
    R, t     : pose of the object in the camera frame, R (3,3), t (3,)
    length_m : arrow length in metres (5 cm by default)
    scale    : screen-space scale on the arrow length. Use ~0.3 for
               zoomed crop panels to avoid arrows overflowing the panel.
    frame    : 'camera' or 'object' — see above.
    """
    if frame == 'camera':
        R_axes = np.eye(3)
    elif frame == 'object':
        R_axes = R
    else:
        raise ValueError("frame must be 'camera' or 'object', got {!r}"
                         .format(frame))

    # Build the four 3D points to project: origin and three axis tips, each
    # expressed in the frame we want to visualise. project_points then maps
    # them to camera coordinates via R_axes @ p + t before perspective divide.
    pts        = np.zeros((4, 3))
    pts[1, 0]  = length_m   # +x tip
    pts[2, 1]  = length_m   # +y tip
    pts[3, 2]  = length_m   # +z tip
    proj       = project_points(pts, R_axes, t)
    origin     = proj[0]

    for i, (color, label) in enumerate(zip(['red',  'green', 'blue'],
                                           ['x',    'y',     'z'])):
        tip_2d = origin + (proj[i + 1] - origin) * scale
        ax.annotate('', xy=tip_2d, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(tip_2d[0], tip_2d[1], label,
                color=color, fontsize=9, fontweight='bold')

    # Origin marker "O" — small white-filled dot + bold black label so it
    # reads on both light and dark backgrounds.
    ax.scatter([origin[0]], [origin[1]], s=24, c='white',
               edgecolors='black', linewidths=1.1, zorder=6)
    ax.text(origin[0] + 6, origin[1] - 6, 'O',
            color='black', fontsize=9, fontweight='bold', zorder=6)


def pose_panel(ax, proj_pred, proj_gt, R_pred, t_pred, R_gt, t_gt,
               background, title, mask_label=None, axes_scale=1.0,
               bbox_3d=None, bbox_lw=1.5, axes_frame='camera',
               show_points=True):
    """Draw one predicted-vs-GT overlay panel.

    Parameters
    ----------
    background  : np.ndarray [H, W, 3] or None
        RGB image to display behind the point clouds. None = black background.
    mask_label  : np.ndarray bool [H, W] or None
        If provided, pixels outside the mask are set to white (object isolation).
    axes_scale  : float
        Screen-space scale for pose frame arrows (use 0.3 for zoomed panels).
    bbox_3d     : np.ndarray (8, 3) or None
        Eight model-frame bbox corners in metres. When given, the cuboid is
        projected under both poses (lime = GT, red = predicted) and overlaid.
    bbox_lw     : float
        Line width for the 3D bbox edges.
    axes_frame  : 'camera' (default) or 'object'
        Reference frame used by the XYZ tricolour arrows. See draw_axes().
        In 'camera' mode the arrows are drawn once at the predicted origin
        (orientation is constant, so a second copy at the GT origin would
        be redundant; the translation error is already visible from the
        lime/red point clouds). In 'object' mode both predicted and GT
        local frames are drawn to make the rotation error legible.
    show_points : bool (default True)
        When False the predicted/ground-truth model-point scatter overlays
        are skipped, leaving only the background image, the 3D bbox and the
        pose axes. Useful for an uncluttered "pose + bbox on image" view.
    """
    dark_bg = (background is None)

    if dark_bg:
        ax.set_facecolor('black')
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
    else:
        ax.set_facecolor('white')
        img_display = background.copy().astype(np.float32)
        if mask_label is not None:
            seg = mask_label.astype(np.float32)
            img_display = (img_display * seg[:, :, np.newaxis] +
                           255.0 * (1.0 - seg[:, :, np.newaxis]))
        ax.imshow(img_display.astype(np.uint8), extent=[0, 640, 480, 0])

    if show_points:
        vgt   = _in_frame(proj_gt)
        vpred = _in_frame(proj_pred)
        ax.scatter(proj_gt[vgt, 0],     proj_gt[vgt, 1],
                   s=2.5, c='lime', alpha=0.7, linewidths=0,
                   label='Ground truth', zorder=3)
        ax.scatter(proj_pred[vpred, 0], proj_pred[vpred, 1],
                   s=2.5, c='red',  alpha=0.7, linewidths=0,
                   label='Predicted',     zorder=3)

    if bbox_3d is not None:
        draw_bbox_3d(ax, bbox_3d, R_gt,   t_gt,   color='lime', lw=bbox_lw)
        draw_bbox_3d(ax, bbox_3d, R_pred, t_pred, color='red',  lw=bbox_lw)

    if axes_frame == 'camera':
        # Constant orientation across frames — drawing the GT copy on top
        # would just overlay an identical trident slightly offset.
        draw_axes(ax, R_pred, t_pred, scale=axes_scale, frame='camera')
    else:
        draw_axes(ax, R_pred, t_pred, scale=axes_scale, frame='object')
        draw_axes(ax, R_gt,   t_gt,   scale=axes_scale, frame='object')

    lfc = '#111111' if dark_bg else 'white'
    llc = 'white'   if dark_bg else 'black'
    # Only draw the legend when there is something to label (point clouds).
    # Without point overlays the legend would be empty and matplotlib warns.
    if show_points:
        ax.legend(loc='upper right', fontsize=8, markerscale=6,
                  facecolor=lfc, labelcolor=llc, edgecolor='gray')
    ax.set_title(title, color='white' if dark_bg else 'black')
    ax.axis('off')


# ── Multi-panel timing figure ─────────────────────────────────────────────────

def timing_plots(t_est_seq, t_ref_seq, add_seq, success_seq, threshold_mm,
                 title, output_path=None):
    """Produce a 4-panel performance figure and save or show it.

    Panel 1 — per-frame latency timeline with 30/60 fps reference lines
    Panel 2 — ADD distance per frame (green=PASS, red=FAIL)
    Panel 3 — fps histogram + CDF
    Panel 4 — estimator vs refiner stacked area

    Parameters
    ----------
    output_path : str or None
        If provided, save to disk. Otherwise call plt.show().
    """
    t_est_arr = np.array(t_est_seq)
    t_ref_arr = np.array(t_ref_seq)
    t_tot_arr = t_est_arr + t_ref_arr
    add_arr   = np.array(add_seq)
    frames    = np.arange(len(t_tot_arr))
    window    = max(1, len(t_tot_arr) // 20)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    fig.suptitle(title, fontsize=12)

    # Panel 1 — latency timeline
    ax1     = fig.add_subplot(gs[0, 0])
    rolling = np.convolve(t_tot_arr, np.ones(window)/window, mode='same')
    ax1.plot(frames, t_tot_arr, color='steelblue', linewidth=0.6, alpha=0.4, label='Raw')
    ax1.plot(frames, rolling,   color='steelblue', linewidth=1.8, label='Rolling mean')
    ax1.axhline(1000/30, color='orange', linestyle='--', linewidth=1.5, label='30 fps (33 ms)')
    ax1.axhline(1000/60, color='red',    linestyle='--', linewidth=1.5, label='60 fps (16 ms)')
    ax1.fill_between(frames, t_tot_arr, 1000/30,
                     where=t_tot_arr > 1000/30, alpha=0.15, color='orange',
                     label='Above 30 fps budget')
    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Per-frame inference latency')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — ADD timeline
    ax2        = fig.add_subplot(gs[0, 1])
    colors_add = ['green' if s else 'red' for s in success_seq]
    ax2.bar(frames, add_arr, color=colors_add, alpha=0.6, width=1.0)
    ax2.axhline(threshold_mm, color='black', linestyle='--', linewidth=1.5,
                label='Threshold ({:.1f} mm)'.format(threshold_mm))
    ax2.plot(frames,
             np.convolve(add_arr, np.ones(window)/window, mode='same'),
             color='navy', linewidth=1.5, label='Rolling mean ADD')
    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('ADD distance (mm)')
    ax2.set_title('ADD accuracy  (green=PASS  red=FAIL)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3 — fps histogram + CDF
    ax3     = fig.add_subplot(gs[1, 0])
    ax3c    = ax3.twinx()
    fps_arr = 1000.0 / t_tot_arr
    ax3.hist(fps_arr, bins=40, color='steelblue', alpha=0.55, density=True)
    sorted_fps = np.sort(fps_arr)
    cdf = np.arange(1, len(sorted_fps)+1) / len(sorted_fps)
    ax3c.plot(sorted_fps, cdf * 100, color='navy', linewidth=2, label='CDF')
    ax3.axvline(30, color='orange', linestyle='--', linewidth=1.5, label='30 fps')
    ax3.axvline(60, color='red',    linestyle='--', linewidth=1.5, label='60 fps')
    pct_30 = float(np.mean(fps_arr >= 30) * 100)
    pct_60 = float(np.mean(fps_arr >= 60) * 100)
    ax3.text(0.02, 0.92, '{:.0f}% ≥ 30 fps'.format(pct_30),
             transform=ax3.transAxes, fontsize=8, color='orange')
    ax3.text(0.02, 0.84, '{:.0f}% ≥ 60 fps'.format(pct_60),
             transform=ax3.transAxes, fontsize=8, color='red')
    ax3.set_xlabel('fps')
    ax3.set_ylabel('Density')
    ax3c.set_ylabel('Cumulative % of frames')
    ax3c.set_ylim(0, 105)
    ax3.set_title('fps distribution + CDF')
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3c.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4 — stacked area breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.stackplot(frames, t_est_arr, t_ref_arr,
                  labels=['Estimator', 'Refiner (×{})'.format(ITERATION)],
                  colors=['steelblue', 'darkorange'], alpha=0.75)
    ax4.axhline(1000/30, color='orange', linestyle='--', linewidth=1.5, label='30 fps')
    ax4.axhline(1000/60, color='red',    linestyle='--', linewidth=1.5, label='60 fps')
    ax4.set_xlabel('Frame index')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Estimator vs Refiner time breakdown')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    # tight_layout is incompatible with twinx axes (panel 3 CDF).
    # subplots_adjust gives equivalent spacing without the warning.
    plt.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.10,
                        hspace=0.45, wspace=0.38)

    if not output_path:
        raise ValueError('timing_plots: output_path is required — '
                         'plt.show() is not supported in a headless environment.')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Plot saved: {}'.format(output_path))
    plt.close()


# ── Delta-ADD evolution plot ──────────────────────────────────────────────────

def add_delta_plot(add_seq, success_seq, threshold_mm, title, output_path):
    """Single-panel plot of frame-to-frame ADD change over the sequence.

    Layout (one axis, full width, *two* Y scales):
      - Left Y axis  : Δ ADD = ADD[i] − ADD[i-1], signed (mm). The first
                       frame has no predecessor, so its delta is set to 0.
                         positive → pose error grew (tracker drifting)
                         negative → pose error shrank (tracker recovering)
                       Bars are coloured green if frame i is PASS, red if
                       FAIL — same convention as the ADD timeline panel in
                       timing_plots() so both figures read in tandem.
                       ±threshold_mm (dashed grey) mark the ADD success
                       threshold; a single-frame delta larger than this is
                       enough on its own to flip a PASS into a FAIL.
      - Right Y axis : rolling mean of Δ ADD on its own scale, so smoothed
                       trends remain readable even when isolated spikes
                       dominate the bar range. A solid black baseline at 0
                       is shared by both axes (means and bars are zero in
                       the same place by construction).

    Parameters
    ----------
    add_seq      : list[float] — per-frame ADD distance in mm
    success_seq  : list[int]   — per-frame PASS(1) / FAIL(0) flag
    threshold_mm : float       — ADD success threshold (= 0.1 × diameter)
    title        : str         — figure suptitle
    output_path  : str         — destination file (.png recommended)
    """
    add_arr = np.array(add_seq, dtype=float)
    if add_arr.size == 0:
        raise ValueError('add_delta_plot: empty add_seq')

    # First frame has no predecessor → keep arrays aligned with success_seq
    # by setting delta[0] = 0 rather than dropping it.
    delta   = np.concatenate(([0.0], np.diff(add_arr)))
    frames  = np.arange(len(delta))
    window  = max(1, len(delta) // 20)
    colors  = ['green' if s else 'red' for s in success_seq]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor='white')
    fig.suptitle(title, fontsize=12)

    # Primary axis — raw deltas (bars) + threshold references
    ax.bar(frames, delta, color=colors, alpha=0.6, width=1.0,
           label='_nolegend_')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.axhline( threshold_mm, color='gray', linestyle='--', linewidth=1.2,
               label='+threshold ({:+.2f} mm)'.format( threshold_mm))
    ax.axhline(-threshold_mm, color='gray', linestyle='--', linewidth=1.2,
               label='-threshold ({:+.2f} mm)'.format(-threshold_mm))
    ax.set_xlabel('Frame index')
    ax.set_ylabel(r'$\Delta$ ADD per frame (mm)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.margins(x=0)

    # Secondary axis — rolling mean on its own scale so smoothing-scale
    # trends stay legible even when bar spikes dominate the primary range.
    # The mean is bounded by max|delta| (centred window), so its dynamic
    # range is typically one to two orders of magnitude smaller than the
    # raw bars; sharing the y-axis flattens it visually.
    ax_mean = ax.twinx()
    rolling = np.convolve(delta, np.ones(window) / window, mode='same')
    mean_line, = ax_mean.plot(frames, rolling, color='navy', linewidth=1.8,
                              label='Rolling mean ({}-frame window)'.format(window))
    ax_mean.axhline(0, color='navy', linewidth=0.5, alpha=0.3)
    ax_mean.set_ylabel(r'$\Delta$ ADD rolling mean (mm)', color='navy')
    ax_mean.tick_params(axis='y', labelcolor='navy')
    # Symmetric range around 0 so positive and negative drift read the same.
    mean_abs_max = float(np.max(np.abs(rolling))) if rolling.size else 1.0
    pad          = max(mean_abs_max * 1.15, 1e-6)
    ax_mean.set_ylim(-pad, pad)

    # Synthetic legend entries for the PASS/FAIL colour code so the reader
    # doesn't have to infer it from the bars alone. Combine handles from
    # both axes into a single legend.
    from matplotlib.patches import Patch
    bar_handles  = [Patch(facecolor='green', alpha=0.6, label='PASS frame'),
                    Patch(facecolor='red',   alpha=0.6, label='FAIL frame')]
    line_h, line_l = ax.get_legend_handles_labels()
    mean_h, mean_l = ax_mean.get_legend_handles_labels()
    handles = bar_handles + line_h + mean_h
    labels  = ['PASS frame', 'FAIL frame'] + line_l + mean_l
    ax.legend(handles, labels, fontsize=8, loc='upper right')

    plt.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.12)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Plot saved: {}'.format(output_path))
    plt.close()


# ── OpenCV overlay for video frames ──────────────────────────────────────────

def draw_frame_cv2(img_arr, R_pred, t_pred, R_gt, t_gt,
                   model_pts, bbox, add_val, threshold, frame_idx, t_total,
                   playback_fps=None):
    """Render one composite video frame with OpenCV overlays.

    Draws:
      - 2D bounding box (green=PASS, red=FAIL)
      - 6D pose axes (XYZ arrows) for predicted pose
      - HUD: frame index, inference latency, playback fps, ADD result

    Parameters
    ----------
    playback_fps : float or None
        Effective playback framerate of the output video (the rate at which
        the viewer sees frames advance). Displayed on the HUD as "Playback".
        If None, the playback line is omitted.
        Note: this is *not* the inference rate (1000 / t_total). The inference
        rate measures how fast DenseFusion can process a frame on the GPU and
        can easily exceed 60 fps, while playback is fixed by the writer fps.

    Returns
    -------
    np.ndarray [H, W, 3] uint8 — BGR for cv2.VideoWriter
    """
    out     = img_arr.copy()
    success = add_val < threshold
    rmin, rmax, cmin, cmax = bbox

    # 2D bounding box
    box_col = (0, 255, 0) if success else (0, 0, 255)
    cv2.rectangle(out, (cmin, rmin), (cmax, rmax), box_col, 2)

    # Camera-frame pose axes anchored at the object origin t_pred.
    #   x (red)   → image right
    #   y (green) → image down
    #   z (blue)  → into the scene (optical-axis depth convention)
    # The arrows keep a constant orientation across frames because we
    # project from the identity rotation; only the anchor (O) moves with
    # the object. The z arrow projects very short whenever t_pred lies
    # near the optical axis — that is correct, depth points away from the
    # image plane and therefore foreshortens to almost nothing in 2D.
    length_m = 0.05
    pts_cam  = np.array([[0.0,      0.0,      0.0     ],
                         [length_m, 0.0,      0.0     ],   # +x tip
                         [0.0,      length_m, 0.0     ],   # +y tip
                         [0.0,      0.0,      length_m]])  # +z tip
    proj    = project_points(pts_cam, np.eye(3), t_pred).astype(int)
    origin  = proj[0]

    # OpenCV is BGR: (0,0,255)=red  (0,255,0)=green  (255,0,0)=blue
    for tip, col, lbl in [(proj[1], (0, 0, 255), 'x'),
                          (proj[2], (0, 255, 0), 'y'),
                          (proj[3], (255, 0, 0), 'z')]:
        cv2.arrowedLine(out, tuple(origin), tuple(tip), col, 2, tipLength=0.3)
        # Halo + coloured label so it stays legible on any background.
        cv2.putText(out, lbl, tuple(tip),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, lbl, tuple(tip),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    # Origin marker "O"
    cv2.circle(out, tuple(origin), 4, (255, 255, 255), -1)
    cv2.circle(out, tuple(origin), 4, (0, 0, 0), 1)
    cv2.putText(out, 'O', (origin[0] + 6, origin[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(out, 'O', (origin[0] + 6, origin[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # HUD — inference latency on its own line; playback fps shown separately
    # so the viewer doesn't mistake GPU throughput for video playback speed.
    hud = ['Frame {:04d}'.format(frame_idx),
           'Inference: {:.1f} ms'.format(t_total)]
    if playback_fps is not None:
        hud.append('Playback: {:.0f} fps'.format(playback_fps))
    hud.append('ADD: {:.2f} mm  [{}]'.format(
        add_val * 1000, 'PASS' if success else 'FAIL'))

    for k, line in enumerate(hud):
        y = 20 + k * 22
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
