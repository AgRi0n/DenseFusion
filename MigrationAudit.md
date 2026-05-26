# DenseFusion — Migration Audit (Python 3.5 + PyTorch 0.4 → Python 3.10 + PyTorch 2.x)

## Summary

| Change | Critical | Status |
|---|---|---|
| `torch.utils.ffi` KNN extension removed in PyTorch 1.0 | **Yes** | ✅ Fixed |
| `Variable()` deprecated since PyTorch 0.4.1 | No | ✅ Fixed |
| `Function.__init__` / `forward` API changed | No | ✅ Moot (KNN replaced) |
| NumPy 2.x ABI incompatible with PyTorch 2.1 | **Yes** | ✅ Fixed |
| `libGL.so.1` missing from base image | **Yes** | ✅ Fixed |
| `nvidia-docker` deprecated | **Yes** | ✅ Fixed |
| Google Drive `wget` download method broken | **Yes** | ⛔ ONGOING |
| `yaml.load()` requires explicit Loader since PyYAML 6.0 | No | ✅ Fixed |
| Core model / loss math | — | No changes needed |
| Python syntax | — | No changes needed |

---

## Issue 1 — `torch.utils.ffi` KNN extension ⛔ Critical

`torch.utils.ffi` was **completely removed in PyTorch 1.0**. The KNN extension in `lib/knn/`
uses it to wrap a C/CUDA kernel, making the entire module unimportable on any modern PyTorch.

**Affected files:**
```
lib/knn/build_ffi.py            — torch.utils.ffi.create_extension (removed)
lib/knn/knn_pytorch/__init__.py — torch.utils.ffi._wrap_function (removed)
lib/knn/__init__.py             — KNearestNeighbor class (replaced)
lib/knn/Makefile                — calls build_ffi.py (no longer needed)
lib/loss.py                     — imports and calls KNearestNeighbor
lib/loss_refiner.py             — imports and calls KNearestNeighbor
tools/eval_linemod.py           — imports and calls KNearestNeighbor
```

**Fix — replaced with `torch.cdist` (no dependencies, built into PyTorch):**

The KNN was only used to find the nearest neighbour in the symmetric object loss.
`torch.cdist` computes the full pairwise distance matrix; `argmin` selects the closest point.
Note that the old C extension returned **1-based indices** — the `- 1` offset present in all
call sites has been removed since `argmin` is 0-based.

```python
# Before — in lib/loss.py, lib/loss_refiner.py, tools/eval_linemod.py
from lib.knn.__init__ import KNearestNeighbor
knn = KNearestNeighbor(1)
inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
target = torch.index_select(target, 1, inds.view(-1) - 1)  # 1-based offset

# After
target_t = target.transpose(0, 1).unsqueeze(0)  # [1, num_point_mesh, 3]
pred_t_  = pred.transpose(0, 1).unsqueeze(0)    # [1, N, 3]
inds = torch.cdist(pred_t_, target_t).argmin(dim=2)  # [1, N] — 0-based
target = torch.index_select(target, 1, inds.view(-1))  # no - 1
```

The `lib/knn/` directory and its build step can be deleted entirely.

**Files changed:** `lib/loss.py`, `lib/loss_refiner.py`, `tools/eval_linemod.py`

---

## Issue 2 — `Variable()` deprecated

In PyTorch 0.4, `Variable(tensor)` was required to enable autograd. Since PyTorch 0.4.1,
autograd is built into tensors and `Variable()` is a transparent no-op. It was kept for
backward compatibility but is now fully removed from modern usage.

**Affected files (30 occurrences across 5 files):**
```
tools/train.py                — 12 occurrences
tools/eval_linemod.py         —  8 occurrences
tools/eval_ycb.py             —  6 occurrences
vanilla_segmentation/train.py —  2 occurrences
lib/knn/__init__.py           —  2 occurrences (moot, file deleted)
```

**Fix — unwrap `Variable()`, remove imports:**

```python
# Before
from torch.autograd import Variable
points, choose, img = Variable(points).cuda(), Variable(choose).cuda(), Variable(img).cuda()
T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda()

# After
points, choose, img = points.cuda(), choose.cuda(), img.cuda()
T = torch.from_numpy(my_t.astype(np.float32)).cuda()
```

**Files changed:** `tools/train.py`, `tools/eval_linemod.py`, `tools/eval_ycb.py`,
`lib/loss.py`, `lib/loss_refiner.py`

---

## Issue 3 — `Function.__init__` / `forward` API

In `lib/knn/__init__.py`, `KNearestNeighbor` inherited from `torch.autograd.Function`
and stored state in `__init__`, a pattern removed in PyTorch 1.x. This is moot since
the entire KNN module has been replaced (see Issue 1).

---

## Issue 4 — NumPy 2.x ABI incompatibility ⛔ Critical

PyTorch 2.1 was compiled against NumPy 1.x. Installing NumPy without a version constraint
pulls in 2.x, which breaks the binary ABI and causes PyTorch to fail to import.

**Fix — pin NumPy in `Dockerfile.modern`:**
```dockerfile
# Before
numpy \

# After
"numpy<2" \
```

---

## Issue 5 — `libGL.so.1` missing from base image ⛔ Critical

`opencv-python` requires `libGL.so.1` at runtime, which is not included in the
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` base image by default.

**Fix — add `libgl1` to apt dependencies in `Dockerfile.modern`:**
```dockerfile
# Added
libgl1 \
```

---

## Issue 6 — `nvidia-docker` deprecated ⛔ Critical

`nvidia-docker` was deprecated in favour of the `--gpus` flag in Docker 19.03+.
The `run.sh` script used `nvidia-docker run` and `nvidia-docker ps`.

**Fix — `run.sh`:**
```bash
# Before
until nvidia-docker ps
nvidia-docker run --name dense_fusion ...

# After
until docker ps
docker run --name dense_fusion --gpus all ...
```

---

## Issue 7 (ONGOING) — Google Drive `wget` download method broken ⛔ Critical

Google Drive blocked the cookie-based `wget` download method used in `download.sh`.
All three downloads (YCB dataset, LineMOD dataset, checkpoints) silently fail.

**Fix — `download.sh`, replace `wget` blocks with `gdown`:**
```bash
# Before
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(...)" \
  -O dataset.zip && rm -rf /tmp/cookies.txt

# After
gdown <file_id> -O dataset.zip
```

`gdown` is installed via `pipx` to avoid touching the system Python environment:
```bash
sudo apt install pipx && pipx ensurepath
# download.sh installs gdown automatically on first run if not present
```

---

## What does NOT need to change

- **Core model architecture** — `lib/network.py`, `lib/pspnet.py`, `lib/extractors.py` are pure `nn.Module`, fully compatible with PyTorch 2.x
- **Loss math** — all tensor operations in `lib/loss.py` and `lib/loss_refiner.py` are unchanged beyond the KNN and Variable fixes above
- **Datasets** — `datasets/ycb/dataset.py` and `datasets/linemod/dataset.py` use standard `DataLoader`, no changes needed
- **Python syntax** — no Python 3.5-specific constructs; the codebase migrates cleanly to 3.10
- **40 `.cuda()` calls** — still valid in modern PyTorch

---

## Files changed

| File | Changes |
|---|---|
| `Dockerfile` | New file. Python 3.10, PyTorch 2.1 + CUDA 11.8, `numpy<2`, `libgl1` |
| `run.sh` | `nvidia-docker` → `docker --gpus all` |
| `download.sh` | `wget` cookie method → `gdown` |
| `lib/loss.py` | KNN → `torch.cdist`; `Variable` removed |
| `lib/loss_refiner.py` | KNN → `torch.cdist`; `Variable` removed |
| `tools/eval_linemod.py` | KNN → `torch.cdist`; `Variable` removed |
| `tools/eval_linemod.py` | `yaml.load()` → `yaml.load(..., Loader=yaml.SafeLoader)` |
| `tools/eval_ycb.py` | `Variable` removed |
| `tools/train.py` | `Variable` removed |
| `lib/knn/` | Entire directory obsolete — can be deleted |
| `datasets/linemod/dataset.py` | `yaml.load()` → `yaml.load(..., Loader=yaml.SafeLoader)` |

---

## Issue 8 — `yaml.load()` requires explicit Loader since PyYAML 6.0

`yaml.load()` without a `Loader` argument was deprecated in PyYAML 5.1 and made a hard
error in 6.0. All call sites raised `TypeError: load() missing 1 required positional argument: 'Loader'`.

**Affected files:**
```
datasets/linemod/dataset.py — 1 occurrence
tools/eval_linemod.py       — 1 occurrence
```

**Fix:**
```python
# Before
self.meta[item] = yaml.load(meta_file)
meta = yaml.load(meta_file)

# After
self.meta[item] = yaml.load(meta_file, Loader=yaml.SafeLoader)
meta = yaml.load(meta_file, Loader=yaml.SafeLoader)
```

`SafeLoader` is the correct choice for dataset metadata files — it parses standard YAML
without executing arbitrary code.

**Files changed:** `datasets/linemod/dataset.py`, `tools/eval_linemod.py`