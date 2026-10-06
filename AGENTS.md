# AGENTS.md

Project: interactive viewer + manual labeling pipeline for MRA intracranial
stenosis grading, feeding a shallow 3D U-Net (training not yet implemented).

Data shape: 359 cases pre-cropped around a single-slice YOLO stenosis box.
Each case is a 3D cube `(slice, row, col) = (17, 24, 24)`, 1 mm isotropic
(NIfTI `zooms = (1,1,1)`). Filenames are `YYYY-MM-DD_vessel.nii.gz`
(e.g. `2023-05-25_right MCA.nii.gz`), so a case is unique per date+vessel.
Because several vessels can share a date, **any train/test split must be
patient-level by file name, never by index**.

Image data is `float32`; label data is `uint8` (binary vessel mask).

## Where things live

- Repo root = the pipeline workspace. It also contains 134 gitignored patient
  folders (`500 .. 794`) holding raw PNG slices + YOLO `.txt` labels. These are
  the input to `prepare_training_data.py` and are NOT in git.
- `training_data_manual/` — the training set produced by the viewer
  (GITIGNORED, patient-derived data, never committed):
  - `images/*.nii.gz`, `labels/*.nii.gz` — the 359 image/label pairs.
  - `manual_labels.json` — manifest of every saved case + its mask params.
  - `ignored_cases.json` — no-stenosis cases excluded from training (created
    only after the first ignore; that file may not exist yet).
- `training_data/` — auto-generated pseudo-masks from `batch_run.py`
  (gitignored). NOT used for training; it is the sanity-check output.
- `config.json` — the single configuration source. `case_overrides` holds the
  per-case mask parameters the viewer saved; `batch_run.py` consumes them.

IMPORTANT: patient data (folders `500..794`, `training_data_manual/`,
`training_data/`, `Selected/`, `test/`) is gitignored on purpose - never
`git add` it. A fresh clone contains only code/config; data stays local.

## Python environments (paths are machine-specific)

Viewer / pipeline (already installed):
- `C:\ProgramData\MRA_Labeling_Env\Scripts\python.exe` (Python 3.9, shared)
  - packages: numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4, nibabel 5.3.3,
    scikit-image 0.24.0, Pillow 11.3.0.
- Launcher: double-click `Label MRA Cases.bat` (its `PY=` path must point at
  a Python with the packages above; TkAgg GUI, so no headless/Agg run).

Training env (partially provisioned, needs work for anyone else):
- `C:\Users\Ulcer\.conda\envs\Pytorch_Tran\python.exe` — torch 2.0.1. Currently
  broken: torchvision's Pillow import fails (`DLL load failed ... _imaging`)
  and nibabel is not installed. Fix by reinstalling Pillow and
  `pip install nibabel` before running any training code.

## Scripts

- `interactive_viewer.py` — the manual labeling GUI (TkAgg). Main entry point
  for the day-to-day annotation work.
- `prepare_training_data.py` — converts raw PNG slices + YOLO box into the
  3D NIfTI cubes + auto masks. `build_crop_mask(vol, method, ...)` is the
  SINGLE source of truth for mask generation — viewer, `batch_run.py`, and
  `export_manual.py` all call it, so the previewed mask is the one that trains.
- `batch_run.py` — batch pipeline wrapper with progress log + resume; reproducible
  against `config.json`.
- `export_manual.py` — optionally regenerate the manual set
  (`manual_labels.json` manifest) and verify it (`--verify`).
- `view_results.py`, `notes.json`, `classes.txt` — older result viewer, Label
  Studio categories, YOLO class names.

## Viewer behavior / work flow

Windows: window 1 = 2D panels + sliders/buttons; window 2 = two 3D mask views
(v1 = per-slice dots, v2 = smooth marching-cubes mesh). Both update together.

Keys (bound to ALL windows, so a clicked 3D window never swallows a shortcut):
`<- / ->` previous/next case, `[ / ]` slice up/down, `1 / 2` method
otsu/frangi, `G / T / O` gamma/threshold/otsu-scale, `S` save label,
`I` ignore (no stenosis, writes `ignored_cases.json` atomically),
`U` jump to next unlabeled case, `V` cycle 3D view presets
(iso -> top -> front -> side -> diag), `Z` reset to iso.

`S` writes: the uint8 mask + a copy of the image cube into `training_data_manual/`,
a `manual_labels.json` record, AND the params into `config.json` under
`case_overrides`. It also clears any ignore flag for that case (a saved label
and an ignore cannot both exist). Masks with diameter > 12 px are flagged as
implausible (cerebral arteries are ~3-6 px on this protocol).

## 3D geometry conventions (do not "fix" these)

- Volume array order is `(slice, row, col)`, i.e. dims `(17, 24, 24)`.
- In both 3D windows: x = column 0..24, y = row 0..24 INVERTED (row 0 at
  top), z = slice index 0..17. Both windows share this exact frame.
- `marching_cubes` returns verts in ARRAY order `(slice,row,col)`; they must be
  reordered to `(col,row,slice)` before handing to `Poly3DCollection` (i.e.
  `verts = verts[:, [2,1,0]]`). The smooth view was completely wrong until this
  was fixed — the vessel was drawn lying down.
- Spacing is 1 mm isotropic everywhere; do NOT invent a 1.5x z-spacing. The
  wireframe box edge test is: pair of cube corners differing in exactly one
  axis (`delta.max() > 0`).
- matplotlib 3.9.4: `ax.clear()` preserves `elev/azim`. Never call
  `view_init` inside the render functions — it discards the user's camera
  rotation. Slice axis always projects vertically; camera presets only change
  how the cube is viewed, never the direction the slice plane sweeps.

## Rules for agents

- NEVER touch the `500..794` patient folders, `Selected/`, `test/`,
  `training_data/`, `training_data_manual/` — all gitignored, regeneratable,
  but patient-derived data that must stay local.
- The dataset to train on is `training_data_manual/` only.
- Mask generation must go through `prepare_training_data.build_crop_mask`;
  do not fork mask logic in the viewer.
- Edits to `config.json` and the `*.json` files use atomic writes
  (temp file + `os.replace`) as the existing code does.
- No lint/test framework exists. Verify 3D rendering with a planted-blob check
  (mesh centroid must match blob `(col,row,slice)`) and confirm limits are
  identical across the two windows before trusting a visual change.
- Git: branch of record is `master`; push target is
  https://github.com/khanhdc/MRA_Stenosis_Visualization.git