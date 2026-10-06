"""
Manual label dataset management for the stenosis training set.

Two modes:

  python export_manual.py            # export: regenerate every manually tuned
                                     # mask from stored params + copy images
  python export_manual.py --verify   # verify: report shapes, empties, drift
                                     # and the manual/auto split

The manual dataset is self-contained:

    training_data_manual/
        images/                 copies of the pipeline cubes
        labels/                 the masks (manual where tuned, auto otherwise)
        manual_labels.json      manifest of manually saved cases + parameters

Every mask is produced by prepare_training_data.build_crop_mask, the same
function the interactive viewer and the batch pipeline use, so a regenerated
mask is identical to the one that was saved.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib

import prepare_training_data as ptd

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.json"
MANIFEST_NAME = "manual_labels.json"
IGNORE_NAME = "ignored_cases.json"
EXPECTED_SHAPE = None  # derived from CROP_SIZE at runtime


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def apply_config(cfg):
    """Push config values into the pipeline module (runtime resolution)."""
    ptd.CROP_SIZE = cfg.get("crop_size", ptd.CROP_SIZE)
    ptd.OTSU_SCALE = cfg.get("otsu_scale", ptd.OTSU_SCALE)
    ptd.OTSU_SIGMA = cfg.get("otsu_sigma", ptd.OTSU_SIGMA)
    ptd.CENTER_SEED_RADIUS = cfg.get("center_seed_radius", ptd.CENTER_SEED_RADIUS)
    ptd.FRANGI_THRESHOLD = cfg.get("threshold_relative", ptd.FRANGI_THRESHOLD)
    ptd.MIN_COMPONENT_AREA = cfg.get("min_component_area_px", ptd.MIN_COMPONENT_AREA)
    vcfg = cfg.get("vesselness", {}) or {}
    ptd.VESSELNESS_SIGMAS = vcfg.get("sigmas", ptd.VESSELNESS_SIGMAS)
    ptd.VESSELNESS_BLACK_RIDGE = vcfg.get("black_ridges", ptd.VESSELNESS_BLACK_RIDGE)
    ptd.FRANGI_GAMMA = vcfg.get("gamma", ptd.FRANGI_GAMMA)
    mcfg = cfg.get("morphology", {}) or {}
    ptd.MORPH_CLOSE_RADIUS = mcfg.get("close_radius", ptd.MORPH_CLOSE_RADIUS)


def dirs_from_config(cfg):
    auto_dir = Path(cfg["output_dir"])
    manual_dir = BASE / cfg.get("manual_data_dir", "training_data_manual")
    return auto_dir, manual_dir


def load_manifest(manual_dir: Path):
    path = manual_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_manifest(manual_dir: Path, manifest: dict):
    with open(manual_dir / MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_ignored(manual_dir: Path):
    """Cases marked 'no stenosis' in the viewer. Excluded from training."""
    path = manual_dir / IGNORE_NAME
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def params_from_record(rec: dict):
    """Extract mask parameters from a manifest record."""
    vov = rec.get("vesselness") or {}
    return {
        "method": rec.get("method", "otsu"),
        "threshold": rec.get("threshold"),
        "gamma": vov.get("gamma"),
        "otsu_scale": rec.get("otsu_scale"),
    }


# ----------------------------------------------------------------- export
def seed_manual_set(auto_dir: Path, manual_dir: Path, overwrite=False):
    """Copy the auto dataset into the manual folder so it is a full set."""
    created, replaced = 0, 0
    for sub in ("images", "labels"):
        (manual_dir / sub).mkdir(parents=True, exist_ok=True)

    for src in sorted((auto_dir / "images").glob("*.nii.gz")):
        dst = manual_dir / "images" / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)
        created += 1

    for src in sorted((auto_dir / "labels").glob("*.nii.gz")):
        dst = manual_dir / "labels" / src.name
        if dst.exists():
            if overwrite:
                shutil.copy2(src, dst)
                replaced += 1
            continue
        shutil.copy2(src, dst)
        created += 1

    return created, replaced


def cmd_export(cfg, manual_dir: Path, auto_dir: Path, seed=False,
               overwrite_auto=True):
    if seed:
        created, replaced = seed_manual_set(auto_dir, manual_dir,
                                            overwrite=overwrite_auto)
        print(f"seeded manual dataset: {created} new, {replaced} overwritten")

    manifest = load_manifest(manual_dir)
    if not manifest:
        print("no manual labels recorded yet (manual_labels.json is empty)")
        return 0

    print(f"exporting {len(manifest)} manual mask(s) -> {manual_dir / 'labels'}")
    fail = 0
    for case_id, rec in sorted(manifest.items()):
        img_path = manual_dir / "images" / f"{case_id}.nii.gz"
        if not img_path.exists():
            src = auto_dir / "images" / f"{case_id}.nii.gz"
            if src.exists():
                shutil.copy2(src, img_path)
            else:
                print(f"  MISSING image for {case_id}")
                fail += 1
                continue

        vol = nib.load(img_path).get_fdata()
        p = params_from_record(rec)
        mask = ptd.build_crop_mask(
            vol,
            method=p["method"],
            threshold=p["threshold"],
            gamma=p["gamma"],
            otsu_scale=p["otsu_scale"],
        )
        nib.save(nib.Nifti1Image(mask.astype(np.uint8),
                                 nib.load(img_path).affine),
                 manual_dir / "labels" / f"{case_id}.nii.gz")
        print(f"  {case_id}: method={p['method']} thr={p['threshold']} "
              f"gamma={p['gamma']} otsu_scale={p['otsu_scale']} "
              f"voxels={int(mask.sum())}")
    return fail


# ----------------------------------------------------------------- verify
def cmd_verify(cfg, manual_dir: Path, auto_dir: Path):
    crop = cfg.get("crop_size", 17)
    expected = (cfg.get("slices_per_case", 17), crop, crop)
    manifest = load_manifest(manual_dir)
    manual_ids = set(manifest)
    ignored = load_ignored(manual_dir)
    ignored_ids = set(ignored)

    imgs = sorted((manual_dir / "images").glob("*.nii.gz"))
    lbls = sorted((manual_dir / "labels").glob("*.nii.gz"))
    img_ids = {p.name for p in imgs}
    lbl_ids = {p.name for p in lbls}
    lbl_cases = {n[:-len(".nii.gz")] for n in lbl_ids}

    print(f"manual dir : {manual_dir}")
    print(f"images     : {len(imgs)}")
    print(f"labels     : {len(lbls)}")
    print(f"manifest   : {len(manual_ids)} manually tuned case(s)")
    present_manual = len(manual_ids & lbl_cases)
    print(f"auto       : {len(lbl_cases) - present_manual} untouched, "
          f"{present_manual} manual")

    both = manual_ids & ignored_ids
    if both:
        print(f"\nWARNING: {len(both)} case(s) are both labelled and ignored; "
              f"the label wins:")
        for n in sorted(both)[:10]:
            print(f"  {n}")

    if ignored_ids:
        print(f"\nignored (no stenosis, EXCLUDED from training): "
              f"{len(ignored_ids)}")
        for n in sorted(ignored_ids)[:10]:
            rec = ignored[n]
            reason = rec.get("reason", "") if isinstance(rec, dict) else ""
            print(f"  {n}" + (f"   [{reason}]" if reason else ""))
        if len(ignored_ids) > 10:
            print(f"  ... and {len(ignored_ids) - 10} more")

    trainable = len(lbl_cases) - len(ignored_ids)
    print(f"\ntrainable  : {trainable} case(s)  "
          f"({len(lbl_cases)} total - {len(ignored_ids)} ignored)")
    print(f"to do      : {len(lbl_cases) - len(manual_ids) - len(ignored_ids)} "
          f"case(s) still need a decision")

    if img_ids - lbl_ids:
        print("\nWARNING: images without labels:")
        for n in sorted(img_ids - lbl_ids):
            print(f"  {n}")
    if lbl_ids - img_ids:
        print("\nWARNING: labels without images:")
        for n in sorted(lbl_ids - img_ids):
            print(f"  {n}")

    bad_shape = []
    empty = []
    counts = []
    mismatched_affine = []

    for lp in lbls:
        lnii = nib.load(lp)
        data = np.asanyarray(lnii.dataobj)
        if data.shape != expected:
            bad_shape.append((lp.name, data.shape))
        n = int(data.sum())
        counts.append(n)
        if n == 0:
            empty.append(lp.name)
        ip = manual_dir / "images" / lp.name
        if ip.exists():
            iaff = nib.load(ip).affine
            if not np.allclose(iaff, lnii.affine):
                mismatched_affine.append(lp.name)

    counts_arr = np.array(counts) if counts else np.array([0])
    print(f"\nexpected shape: {expected}")
    print(f"shape violations : {len(bad_shape)}")
    for name, shp in bad_shape[:10]:
        print(f"  {name}: {shp}")
    print(f"empty masks      : {len(empty)}")
    for name in empty[:10]:
        print(f"  {name}")
    print(f"affine mismatches: {len(mismatched_affine)}")
    for name in mismatched_affine[:10]:
        print(f"  {name}")
    if len(counts):
        print(f"\nvoxels: min={counts_arr.min()} median={int(np.median(counts_arr))} "
              f"mean={counts_arr.mean():.0f} max={counts_arr.max()}")

    # drift: regenerate each manual mask and compare to what is on disk
    drift = []
    for case_id, rec in sorted(manifest.items()):
        lp = manual_dir / "labels" / f"{case_id}.nii.gz"
        ip = manual_dir / "images" / f"{case_id}.nii.gz"
        if not lp.exists() or not ip.exists():
            continue
        p = params_from_record(rec)
        on_disk = np.asanyarray(nib.load(lp).dataobj).astype(np.uint8)
        regen = ptd.build_crop_mask(
            nib.load(ip).get_fdata(), method=p["method"],
            threshold=p["threshold"], gamma=p["gamma"],
            otsu_scale=p["otsu_scale"],
        ).astype(np.uint8)
        if not np.array_equal(on_disk, regen):
            diff = int((on_disk != regen).sum())
            drift.append((case_id, diff))

    print(f"\nmanual masks reproducible from stored params: "
          f"{len(manifest) - len(drift)}/{len(manifest)}")
    for case_id, diff in drift[:10]:
        print(f"  DRIFT {case_id}: {diff} differing voxels")

    ok = not (bad_shape or empty or mismatched_affine or drift)
    print("\nRESULT:", "OK" if ok else "ISSUES FOUND")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify shapes/empties/drift and manual-auto split")
    ap.add_argument("--seed", action="store_true",
                    help="copy the auto dataset into the manual folder first")
    ap.add_argument("--overwrite-auto", action="store_true",
                    help="with --seed, also overwrite manually saved labels")
    args = ap.parse_args()

    cfg = load_config()
    apply_config(cfg)
    auto_dir, manual_dir = dirs_from_config(cfg)

    if args.verify:
        return cmd_verify(cfg, manual_dir, auto_dir)

    return cmd_export(cfg, manual_dir, auto_dir, seed=args.seed,
                      overwrite_auto=args.overwrite_auto)


if __name__ == "__main__":
    sys.exit(main() or 0)