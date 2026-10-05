"""
Prepare pseudo-3D training data for Shallow 3D U-Net.

Converts per-case PNG slices (17 slices, single-slice YOLO annotation)
into 3D NIfTI volumes + pseudo vessel masks.

Data structure expected per case:
    <case_folder>/
        img-XXXXX-YYYYY.png  (17 images, YYYY = slice index)
        img-XXXXX-ZZZZZ.txt  (YOLO label for middle slice = 9th image)

Usage:
    python prepare_training_data.py --input_dir "E:/MRA_Data/MRA_Annotation_Ver2/stenosis labeling"
    python prepare_training_data.py --input_dir . --output_dir ./training_data --margin 0.25
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import disk, closing
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASSES_FILE = "classes.txt"
SLICES_PER_CASE = 17
MIDDLE_SLICE_IDX = 8  # 0-indexed → 9th image
MARGIN_RATIO = 0.25   # expand 2D box by this fraction on each side
CROP_SIZE = 17        # in-plane crop: output cubes are (17, CROP_SIZE, CROP_SIZE)

# Mask generation: "otsu" (adaptive, default) or "frangi" (fixed params)
MASK_METHOD = "otsu"
OTSU_SCALE = 0.80      # final threshold = Otsu(t_otsu) * this scale
OTSU_SIGMA = 1.0       # gaussian smoothing sigma before thresholding
CENTER_SEED_RADIUS = 3 # keep 3D components touching center box on any slice

VESSELNESS_SIGMAS = [1.0, 1.5, 2.0]
VESSELNESS_BLACK_RIDGE = False  # MRA vessels are bright
FRANGI_THRESHOLD = 0.3  # relative threshold on normalised response
FRANGI_GAMMA = 15       # Frangi selectivity (raise to allow curved vessels)

MIN_COMPONENT_AREA = 5  # remove tiny blobs (in pixels)
MORPH_CLOSE_RADIUS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def natural_sort_key(path: Path):
    """Sort key that handles embedded numbers naturally."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_yolo_label(txt_path: Path, img_w: int, img_h: int):
    """
    Parse a YOLO normalised label file.

    Returns list of dicts:
        {"class_id": int, "cx": float, "cy": float, "w": float, "h": float,
         "x1": int, "y1": int, "x2": int, "y2": int}
    """
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int(round((cx - w / 2) * img_w))
            y1 = int(round((cy - h / 2) * img_h))
            x2 = int(round((cx + w / 2) * img_w))
            y2 = int(round((cy + h / 2) * img_h))
            boxes.append({
                "class_id": cls_id,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return boxes


def expand_box(x1, y1, x2, y2, img_w, img_h, margin=MARGIN_RATIO):
    """Expand bounding box by `margin` fraction, clamped to image bounds."""
    bw = x2 - x1
    bh = y2 - y1
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)
    return nx1, ny1, nx2, ny2


def center_crop_window(cx_norm, cy_norm, img_w, img_h, size=CROP_SIZE):
    """size x size window centered on the box center, clamped to image bounds."""
    x0 = int(round(cx_norm * img_w)) - size // 2
    y0 = int(round(cy_norm * img_h)) - size // 2
    x0 = max(0, min(img_w - size, x0))
    y0 = max(0, min(img_h - size, y0))
    return x0, y0, x0 + size, y0 + size


def build_crop_mask(vol_crop, method=None, threshold=None, gamma=None,
                    otsu_scale=None, otsu_sigma=None, sigmas=None,
                    seed_radius=None, close_radius=None, min_area=None):
    """
    Single source of truth for pseudo-mask generation on a cropped cube.

    Shared by the batch pipeline, the interactive viewer and the manual
    export script so the previewed mask is identical to the one that
    trains the model.

    Parameters default to the module-level config constants, resolved at
    call time so runtime overrides are honoured.

    Steps:
        1. Build a candidate vessel mask (Otsu adaptive or Frangi)
        2. Keep only components passing through the stenosis centre
        3. Light per-slice closing (no opening - it erodes faint vessels)
        4. Remove small blobs

    Returns uint8 array with the same shape as vol_crop.
    """
    if method is None:
        method = MASK_METHOD
    if threshold is None:
        threshold = FRANGI_THRESHOLD
    if gamma is None:
        gamma = FRANGI_GAMMA
    if otsu_scale is None:
        otsu_scale = OTSU_SCALE
    if otsu_sigma is None:
        otsu_sigma = OTSU_SIGMA
    if sigmas is None:
        sigmas = VESSELNESS_SIGMAS
    if seed_radius is None:
        seed_radius = CENTER_SEED_RADIUS
    if close_radius is None:
        close_radius = MORPH_CLOSE_RADIUS
    if min_area is None:
        min_area = MIN_COMPONENT_AREA

    vol = np.asarray(vol_crop, dtype=np.float64)
    if vol.size == 0:
        return np.zeros(vol.shape, dtype=np.uint8)

    if method == "otsu":
        sm = gaussian_filter(vol[vol.shape[0] // 2], sigma=otsu_sigma)
        t = threshold_otsu(sm) if sm.max() > sm.min() else 0
        binary = np.zeros(vol.shape, dtype=np.uint8)
        for zi in range(vol.shape[0]):
            s = gaussian_filter(vol[zi], sigma=otsu_sigma)
            binary[zi] = (s >= t * otsu_scale).astype(np.uint8)
    elif method == "frangi":
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-8:
            return np.zeros(vol.shape, dtype=np.uint8)
        vol_norm = (vol - vmin) / (vmax - vmin)
        vesselness = frangi(
            vol_norm, sigmas=sigmas, alpha=0.5, beta=0.5, gamma=gamma,
            black_ridges=VESSELNESS_BLACK_RIDGE,
        )
        t = threshold * vesselness.max() if vesselness.max() > 0 else 0
        binary = (vesselness >= t).astype(np.uint8)
    else:
        raise ValueError("Unknown mask method: %r" % (method,))

    # Keep only component(s) passing through the stenosis centre (any slice).
    # A severe stenosis is dark in MRA (narrowed lumen) and only clearly
    # visible on slices above/below, so seed on ANY slice, not just the middle.
    labelled = sk_label(binary)
    h, w = binary.shape[1], binary.shape[2]
    cx, cy = w // 2, h // 2
    keep = set()
    for region in regionprops(labelled):
        for coord in region.coords:
            _z, y, x = coord
            if abs(x - cx) <= seed_radius and abs(y - cy) <= seed_radius:
                keep.add(region.label)
                break
    if keep:
        out = np.zeros(binary.shape, dtype=np.uint8)
        for lid in keep:
            out[labelled == lid] = 1
        binary = out
    else:
        binary = np.zeros(binary.shape, dtype=np.uint8)

    # Remember which voxels belong to the annotated vessel. Closing below
    # re-labels the components, so component ids cannot be trusted afterwards -
    # the voxel set has to be captured here.
    seed_voxels = (binary > 0).copy()

    # Light per-slice closing (NO opening - it erodes faint vessels).
    if close_radius and close_radius > 0:
        struct = disk(close_radius)
        for i in range(binary.shape[0]):
            binary[i] = closing(binary[i], struct).astype(np.uint8)

    # Remove small blobs, but never the annotated vessel: a severe stenosis can
    # legitimately be only a few voxels, and dropping it would silently produce
    # an empty mask for a case the annotator still has to label.
    if min_area and min_area > 0:
        labelled = sk_label(binary)
        for region in regionprops(labelled):
            if region.area >= min_area:
                continue
            sel = labelled == region.label
            if (sel & seed_voxels).any():
                continue
            binary[sel] = 0

    return binary.astype(np.uint8)


def build_case_mask(volume_3d, box, method=None, threshold=None, gamma=None,
                    otsu_scale=None, crop_size=None):
    """
    Crop a full volume around a YOLO box and build its mask.

    Returns (vol_crop, mask_crop, (x0, y0)) where vol_crop has shape
    (n_slices, crop_size, crop_size).
    """
    if crop_size is None:
        crop_size = CROP_SIZE
    img_h, img_w = volume_3d.shape[1], volume_3d.shape[2]
    x0, y0, x1, y1 = center_crop_window(box["cx"], box["cy"], img_w, img_h,
                                        size=crop_size)
    vol_crop = volume_3d[:, y0:y1, x0:x1]
    mask_crop = build_crop_mask(vol_crop, method=method, threshold=threshold,
                                gamma=gamma, otsu_scale=otsu_scale)
    return vol_crop, mask_crop, (x0, y0)



    if method == "otsu":
        binary = generate_otsu_mask(sub)
    elif method == "frangi":
        binary = generate_frangi_mask(sub)
    else:
        raise ValueError(f"Unknown mask method: {method}")

    # Place back into full volume
    mask_3d[:, y1:y2, x1:x2] = binary.astype(np.uint8)
    return mask_3d


def load_case(case_dir: Path):
    """
    Load a single case directory.

    Returns:
        volume_3d: np.ndarray (n_slices, H, W) uint16 or uint8
        boxes: list of YOLO dicts for the middle slice
        slice_files: sorted list of Paths
    """
    pngs = sorted(case_dir.glob("*.png"), key=natural_sort_key)
    if len(pngs) != SLICES_PER_CASE:
        log.warning(
            f"Expected {SLICES_PER_CASE} slices in {case_dir.name}, "
            f"found {len(pngs)}. Skipping."
        )
        return None, None, None

    # Load images (convert to grayscale if RGBA)
    slices = []
    ref_shape = None
    for p in pngs:
        img = Image.open(p)
        if img.mode == "RGBA":
            img = img.convert("L")
        elif img.mode != "L":
            img = img.convert("L")
        arr = np.array(img)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            log.warning(
                f"Size mismatch in {case_dir.name}: {p.name} is {arr.shape}, "
                f"expected {ref_shape}. Skipping."
            )
            return None, None, None
        slices.append(arr)
    volume_3d = np.stack(slices, axis=0)  # (17, H, W)

    # Parse YOLO label for middle slice
    middle_png = pngs[MIDDLE_SLICE_IDX]
    txt_path = middle_png.with_suffix(".txt")
    img_h, img_w = volume_3d.shape[1], volume_3d.shape[2]
    boxes = parse_yolo_label(txt_path, img_w, img_h)

    return volume_3d, boxes, pngs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_case(case_dir: Path, output_img_dir: Path, output_lbl_dir: Path,
                 output_vis_dir: Path, margin: float, class_names: dict,
                 method: str = MASK_METHOD, threshold: float = None,
                 gamma: float = None, otsu_scale: float = None):
    """Process one case: load → build pseudo-mask → save NIfTI."""
    case_id = case_dir.name
    parent_id = case_dir.parent.name
    full_id = f"{parent_id}_{case_id}"

    volume_3d, boxes, slice_files = load_case(case_dir)
    if volume_3d is None:
        return False

    # Take first box (primary annotation)
    if not boxes:
        log.warning(f"No YOLO label found for {full_id}. Skipping.")
        return False

    box = boxes[0]
    cls_name = class_names.get(box["class_id"], f"cls{box['class_id']}")
    log.info(
        f"Processing {full_id} | class={cls_name} | "
        f"box=({box['x1']},{box['y1']})-({box['x2']},{box['y2']})"
    )

    img_h, img_w = volume_3d.shape[1], volume_3d.shape[2]

    # Expand 2D box
    ex1, ey1, ex2, ey2 = expand_box(
        box["x1"], box["y1"], box["x2"], box["y2"],
        img_w, img_h, margin=margin,
    )

    # Crop first, then mask the crop, so the saved mask is identical to what
    # the interactive viewer renders for the same cube.
    vol_cropped, msk_cropped, (cx0, cy0) = build_case_mask(
        volume_3d, box, method=method, threshold=threshold, gamma=gamma,
        otsu_scale=otsu_scale, crop_size=CROP_SIZE,
    )

    # Build affine with offset to preserve original coordinate origin
    affine = np.eye(4)
    affine[0, 3] = cx0
    affine[1, 3] = cy0

    # Save image volume
    img_nii = nib.Nifti1Image(vol_cropped.astype(np.float32), affine)
    img_path = output_img_dir / f"{full_id}.nii.gz"
    nib.save(img_nii, img_path)

    # Save pseudo-mask
    lbl_nii = nib.Nifti1Image(msk_cropped.astype(np.uint8), affine)
    lbl_path = output_lbl_dir / f"{full_id}.nii.gz"
    nib.save(lbl_nii, lbl_path)

    log.info(f"  Saved: {img_path.name}  |  {lbl_path.name}")

    # Optional: save a QC PNG of the middle slice with overlays
    _save_visualization(
        vol_cropped, msk_cropped, box, ex1 - cx0, ey1 - cy0, ex2 - cx0, ey2 - cy0,
        cls_name, full_id, output_vis_dir, slice_files, cx0, cy0,
    )

    return True


def _save_visualization(volume_3d, mask_3d, box, ex1, ey1, ex2, ey2,
                        cls_name, full_id, output_vis_dir, slice_files,
                        x0=0, y0=0):
    """Save a 3-panel montage of the middle (cropped) slice with overlays.

    volume_3d / mask_3d are already cropped; ex1..ey2 are the expanded-box
    coords shifted into crop space; x0/y0 are the crop origin in the original image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        log.warning("matplotlib not installed; skipping visualization.")
        return

    mid = MIDDLE_SLICE_IDX
    mid_img = volume_3d[mid].astype(np.float32)
    mid_mask = mask_3d[mid]
    h, w = mid_img.shape

    # Normalise for display
    vmin, vmax = np.percentile(mid_img, [2, 98])
    disp = np.clip((mid_img - vmin) / (vmax - vmin + 1e-8), 0, 1)

    def clip_rect(x1, y1, x2, y2):
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title(f"{full_id} | slice {mid}\norigin=({x0},{y0})")
    bx1, by1, bw, bh = clip_rect(box["x1"] - x0, box["y1"] - y0,
                                 box["x2"] - x0, box["y2"] - y0)
    rect = Rectangle((bx1, by1), bw, bh, linewidth=2, edgecolor="red",
                     facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_xlabel("YOLO box (red)")

    axes[1].imshow(disp, cmap="gray")
    axes[1].imshow(mid_mask, cmap="autumn", alpha=0.4)
    axes[1].set_title("Pseudo-mask overlay")
    cx1, cy1, cw, ch = clip_rect(ex1, ey1, ex2, ey2)
    rect2 = Rectangle((cx1, cy1), cw, ch, linewidth=1.5, edgecolor="cyan",
                      facecolor="none", linestyle="--")
    axes[1].add_patch(rect2)

    axes[2].imshow(mid_mask, cmap="gray")
    axes[2].set_title(f"Mask only | {cls_name}\n{int(mid_mask.sum())} px")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    vis_path = output_vis_dir / f"{full_id}_qc.png"
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def discover_cases(input_dir: Path):
    """
    Walk the input directory and yield case folders.
    Expected structure: <input_dir>/<patient_id>/<date>/<case_name>/
    """
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name in ("Selected", "test", "training_data"):
            continue
        for date_dir in sorted(patient_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for case_dir in sorted(date_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                # Check it contains PNGs
                if any(case_dir.glob("*.png")):
                    yield case_dir


def main():
    global CROP_SIZE
    parser = argparse.ArgumentParser(
        description="Prepare pseudo-3D training data from PNG slices + YOLO labels."
    )
    parser.add_argument(
        "--input_dir", type=str,
        default=r"E:\MRA_Data\MRA_Annotation_Ver2\stenosis labeling",
        help="Root input directory containing all case folders.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=None,
        help="Output directory (default: <input_dir>/training_data).",
    )
    parser.add_argument(
        "--margin", type=float, default=MARGIN_RATIO,
        help="Box expansion margin ratio (default: 0.25).",
    )
    parser.add_argument(
        "--method", type=str, default=MASK_METHOD,
        choices=["otsu", "frangi"],
        help="Mask generation method: 'otsu' (adaptive, default) or 'frangi' (fixed).",
    )
    parser.add_argument(
        "--crop_size", type=int, default=CROP_SIZE,
        help=f"In-plane crop size; output is (17, crop_size, crop_size). Default {CROP_SIZE}.",
    )
    parser.add_argument(
        "--classes_file", type=str, default=CLASSES_FILE,
        help="Path to classes.txt (default: classes.txt in input_dir).",
    )
    args = parser.parse_args()

    CROP_SIZE = args.crop_size

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "training_data"

    # Load class names
    cls_path = input_dir / args.classes_file
    class_names = {}
    if cls_path.exists():
        with open(cls_path) as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()

    # Create output dirs
    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"
    vis_out = output_dir / "visualizations"
    for d in [img_out, lbl_out, vis_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Discover and process
    cases = list(discover_cases(input_dir))
    log.info(f"Found {len(cases)} case folders to process.")

    success, fail = 0, 0
    for case_dir in cases:
        ok = process_case(case_dir, img_out, lbl_out, vis_out, args.margin,
                          class_names, method=args.method)
        if ok:
            success += 1
        else:
            fail += 1

    log.info(f"Done. {success} succeeded, {fail} skipped/failed.")
    log.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
