"""
lib/custom/results.py — Pose result serialisation for custom inference runs.

Schema (one entry per frame × mask pair):
{
    "frame_id":  int,
    "mask_id":   int,
    "obj_id":    int,
    "files": {
        "rgb":  str,
        "npz":  str,
        "mask": str,
        "ply":  str
    },
    "camera": {
        "profile": str,
        "fx": float, "fy": float,
        "cx": float, "cy": float
    },
    "pose_6d": {
        "R": [[3×3]],      # rotation matrix
        "t": [tx, ty, tz]  # translation in metres
    },
    "timestamp": str       # ISO-8601 UTC
}

Extend by passing extra keyword arguments to build_entry() — they are merged
at the top level, making the schema forward-compatible.
"""

import json
import os
from datetime import datetime, timezone


def build_entry(frame_id, mask_id, obj_id,
                R, t,
                rgb_path, npz_path, mask_path, ply_path,
                camera_profile, cam_fx, cam_fy, cam_cx, cam_cy,
                **extra):
    """
    Build a single result dict for one inference run.

    Parameters
    ----------
    frame_id, mask_id, obj_id : int
    R        : np.ndarray (3, 3) — rotation matrix
    t        : np.ndarray (3,)   — translation in metres
    *_path   : str               — source file paths (basenames stored)
    camera_* : str / float       — intrinsics used for back-projection
    **extra  : any               — forwarded verbatim into the top-level dict

    Returns
    -------
    dict
    """
    entry = {
        "frame_id":  int(frame_id),
        "mask_id":   int(mask_id),
        "obj_id":    int(obj_id),
        "files": {
            "rgb":  os.path.basename(rgb_path),
            "npz":  os.path.basename(npz_path),
            "mask": os.path.basename(mask_path),
            "ply":  os.path.basename(ply_path),
        },
        "camera": {
            "profile": camera_profile,
            "fx": float(cam_fx),
            "fy": float(cam_fy),
            "cx": float(cam_cx),
            "cy": float(cam_cy),
        },
        "pose_6d": {
            "R": R.tolist(),
            "t": t.tolist(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry.update(extra)
    return entry


def load(json_path):
    """Load existing results list from *json_path*, or return []."""
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save(json_path, results):
    """Write results list to *json_path* (pretty-printed)."""
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def upsert(json_path, entry):
    """
    Load *json_path*, replace the entry matching (frame_id, mask_id) if it
    exists, otherwise append. Save and return the updated list.
    """
    results = load(json_path)
    key = (entry['frame_id'], entry['mask_id'])
    idx = next((i for i, r in enumerate(results)
                if (r.get('frame_id'), r.get('mask_id')) == key), None)
    if idx is not None:
        results[idx] = entry
    else:
        results.append(entry)
    save(json_path, results)
    return results
