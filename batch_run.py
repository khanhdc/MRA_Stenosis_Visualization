"""
Batch runner with progress tracking and resume capability.

Reads config.json, processes all cases, and writes a progress log
so reruns skip already-processed cases.
"""

import json
import sys
import time
from pathlib import Path

# Import from the main module
sys.path.insert(0, str(Path(__file__).parent))
from prepare_training_data import (
    load_case, expand_box, build_case_mask, _save_visualization,
    natural_sort_key, SLICES_PER_CASE,
)
import numpy as np
import nibabel as nib
import logging

import prepare_training_data as ptd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_config(config_path: Path):
    with open(config_path) as f:
        return json.load(f)


def discover_cases(input_dir: Path):
    skip_dirs = {"Selected", "test", "training_data", "__pycache__"}
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name in skip_dirs:
            continue
        for date_dir in sorted(patient_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for case_dir in sorted(date_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if any(case_dir.glob("*.png")):
                    yield case_dir


def load_progress(progress_path: Path):
    if progress_path.exists():
        with open(progress_path) as f:
            return set(json.load(f))
    return set()


def save_progress(progress_path: Path, done: set):
    with open(progress_path, "w") as f:
        json.dump(sorted(done), f)


def process_one(case_dir: Path, cfg: dict, output_dir: Path, class_names: dict):
    case_id = case_dir.name
    parent_id = case_dir.parent.name
    full_id = f"{parent_id}_{case_id}"

    # Per-case parameter overrides (from config["case_overrides"]).
    # Each override may carry method / otsu_scale / gamma / threshold so a
    # manually tuned mask can be reproduced exactly by --redo-overrides.
    overrides = cfg.get("case_overrides", {}) or {}
    ov = overrides.get(full_id) or {}
    if not isinstance(ov, dict):
        ov = {"method": ov}
    method = ov.get("method", cfg.get("mask_method", "otsu"))
    vov = ov.get("vesselness") or {}
    m_gamma = vov.get("gamma", (cfg.get("vesselness") or {}).get("gamma"))
    otsu_scale = ov.get("otsu_scale", cfg.get("otsu_scale"))
    threshold = ov.get("threshold", cfg.get("threshold_relative"))

    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"
    vis_out = output_dir / "visualizations"

    volume_3d, boxes, slice_files = load_case(case_dir)
    if volume_3d is None:
        return None

    if not boxes:
        log.warning(f"No label: {full_id}")
        return None

    box = boxes[0]
    img_h, img_w = volume_3d.shape[1], volume_3d.shape[2]

    # Crop first, then mask the crop. This makes the saved mask identical to
    # what the interactive viewer renders for the same cube.
    vol_cropped, msk_cropped, (cx0, cy0) = build_case_mask(
        volume_3d, box, method=method, threshold=threshold,
        gamma=m_gamma, otsu_scale=otsu_scale, crop_size=ptd.CROP_SIZE,
    )

    affine = np.eye(4)
    affine[0, 3] = cx0
    affine[1, 3] = cy0

    nib.save(nib.Nifti1Image(vol_cropped.astype(np.float32), affine),
             img_out / f"{full_id}.nii.gz")
    nib.save(nib.Nifti1Image(msk_cropped.astype(np.uint8), affine),
             lbl_out / f"{full_id}.nii.gz")

    cls_name = class_names.get(box["class_id"], f"cls{box['class_id']}")
    if cfg.get("save_visualizations", True):
        bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
        ex1, ey1, ex2, ey2 = box["x1"], box["y1"], box["x2"], box["y2"]
        if ov.get("draw_expanded_box", True):
            margin = cfg.get("box_expansion_margin", 0.25)
            ex1, ey1, ex2, ey2 = expand_box(
                box["x1"], box["y1"], box["x2"], box["y2"],
                img_w, img_h, margin=margin,
            )
        _save_visualization(
            vol_cropped, msk_cropped, box,
            ex1 - cx0, ey1 - cy0, ex2 - cx0, ey2 - cy0,
            cls_name, full_id, vis_out, slice_files, cx0, cy0,
        )

    return full_id


def main():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        log.error(f"config.json not found at {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])

    # Apply config overrides to the processing module
    ptd.MARGIN_RATIO = cfg.get("box_expansion_margin", ptd.MARGIN_RATIO)
    ptd.FRANGI_THRESHOLD = cfg.get("threshold_relative", ptd.FRANGI_THRESHOLD)
    ptd.MIN_COMPONENT_AREA = cfg.get("min_component_area_px", ptd.MIN_COMPONENT_AREA)
    ptd.MASK_METHOD = cfg.get("mask_method", ptd.MASK_METHOD)
    ptd.OTSU_SCALE = cfg.get("otsu_scale", ptd.OTSU_SCALE)
    ptd.OTSU_SIGMA = cfg.get("otsu_sigma", ptd.OTSU_SIGMA)
    ptd.CENTER_SEED_RADIUS = cfg.get("center_seed_radius", ptd.CENTER_SEED_RADIUS)
    ptd.CROP_SIZE = cfg.get("crop_size", ptd.CROP_SIZE)
    # export_manual validates the expected shape from config, so the batch
    # writer must honour the same value or the two disagree.
    ptd.SLICES_PER_CASE = cfg.get("slices_per_case", ptd.SLICES_PER_CASE)
    vcfg = cfg.get("vesselness", {})
    ptd.VESSELNESS_SIGMAS = vcfg.get("sigmas", ptd.VESSELNESS_SIGMAS)
    ptd.VESSELNESS_BLACK_RIDGE = vcfg.get("black_ridges", ptd.VESSELNESS_BLACK_RIDGE)
    ptd.FRANGI_GAMMA = vcfg.get("gamma", ptd.FRANGI_GAMMA)
    mcfg = cfg.get("morphology", {})
    ptd.MORPH_CLOSE_RADIUS = mcfg.get("close_radius", ptd.MORPH_CLOSE_RADIUS)
    log.info(
        f"Config: margin={ptd.MARGIN_RATIO}, threshold={ptd.FRANGI_THRESHOLD}, "
        f"method={ptd.MASK_METHOD}, sigmas={ptd.VESSELNESS_SIGMAS}, "
        f"close_r={ptd.MORPH_CLOSE_RADIUS}"
    )

    # Load classes
    cls_path = input_dir / cfg.get("classes_file", "classes.txt")
    class_names = {}
    if cls_path.exists():
        with open(cls_path) as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()

    # Create output dirs
    for sub in ["images", "labels", "visualizations"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Progress tracking
    progress_path = output_dir / "progress.json"
    done = load_progress(progress_path)

    cases = list(discover_cases(input_dir))
    log.info(f"Total cases found: {len(cases)} | Already processed: {len(done)}")

    success, skip, fail = 0, 0, 0
    t0 = time.time()
    redo_overrides = "--redo-overrides" in sys.argv
    forced = set(cfg.get("case_overrides", {}).keys()) if redo_overrides else set()

    for i, case_dir in enumerate(cases):
        case_id = f"{case_dir.parent.name}_{case_dir.name}"
        if case_id in done and case_id not in forced:
            skip += 1
            continue

        try:
            result = process_one(case_dir, cfg, output_dir, class_names)
            if result:
                done.add(result)
                success += 1
            else:
                fail += 1
        except Exception as e:
            log.error(f"Failed {case_id}: {e}")
            fail += 1

        # Save progress every 10 cases
        if (success + fail) % 10 == 0:
            save_progress(progress_path, done)

        # Progress printout
        if (success + fail) % 25 == 0 and (success + fail) > 0:
            elapsed = time.time() - t0
            rate = (success + fail) / elapsed
            eta = (len(cases) - len(done)) / rate if rate > 0 else 0
            log.info(
                f"Progress: {len(done)}/{len(cases)} "
                f"({100*len(done)/len(cases):.1f}%) | "
                f"Speed: {rate:.1f} cases/s | ETA: {eta/60:.0f} min"
            )

    save_progress(progress_path, done)
    elapsed = time.time() - t0

    log.info(
        f"COMPLETE in {elapsed/60:.1f} min | "
        f"Success: {success} | Skipped: {skip} | Failed: {fail}"
    )


if __name__ == "__main__":
    main()
