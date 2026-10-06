"""
Interactive threshold viewer with popup window.
Focuses on the stenosis region using the YOLO box from the middle slice.

Window 1: 2D panels (cropped image / overlay / mask) + sliders & buttons.
Window 2: separate 3D cube visualization of the vessel mask.
Adjust threshold + slice in real-time; both windows update together.
"""
import os, sys, glob
import json
from itertools import product
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Single source of truth for mask generation: shared with batch_run.py and
# export_manual.py so the previewed mask is exactly the one that trains.
import prepare_training_data as ptd
from prepare_training_data import build_crop_mask

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "training_data")
IMG_DIR = os.path.join(DATA_DIR, "images")
CONFIG_PATH = os.path.join(BASE, "config.json")


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


CFG = load_config()
MANUAL_DIR = CFG.get("manual_data_dir", os.path.join(BASE, "training_data_manual"))
MANUAL_IMG_DIR = os.path.join(MANUAL_DIR, "images")
MANUAL_LBL_DIR = os.path.join(MANUAL_DIR, "labels")
MANIFEST_PATH = os.path.join(MANUAL_DIR, "manual_labels.json")
IGNORE_PATH = os.path.join(MANUAL_DIR, "ignored_cases.json")

# Defaults come from config.json so the viewer starts on the batch defaults.
DEFAULTS = {
    "method": CFG.get("mask_method", "otsu"),
    "threshold": CFG.get("threshold_relative", 0.15),
    "gamma": (CFG.get("vesselness") or {}).get("gamma", 0.5),
    "otsu_scale": CFG.get("otsu_scale", 0.80),
}
METHOD_RADIO_INDEX = {"otsu": 0, "frangi": 1}

# Volumes on disk are pre-cropped by the pipeline to
# (slices_per_case, crop_size, crop_size). Read both from config so this
# can never drift from what batch_run actually wrote.
SLICES_PER_CASE = CFG.get("slices_per_case", 17)
CROP_SIZE = CFG.get("crop_size", 17)
# Cerebral arteries measure ~3-6 px on this protocol; beyond 12 px the
# mask has almost certainly bled into adjacent bright tissue.
MAX_PLAUSIBLE_DIA_PX = 12.0
ORIGINAL_IMG_SIZE = 320  # YOLO labels are normalized against the original image

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))
if not images:
    print("No images found."); sys.exit(1)


def build_case_map():
    """Map training-case id (date_case) -> YOLO txt path of middle slice."""
    case_map = {}
    skip = {"Selected", "test", "training_data", "__pycache__"}
    for patient in sorted(os.listdir(BASE)):
        pp = os.path.join(BASE, patient)
        if not os.path.isdir(pp) or patient in skip:
            continue
        for d in sorted(os.listdir(pp)):
            dp = os.path.join(pp, d)
            if not os.path.isdir(dp):
                continue
            for c in sorted(os.listdir(dp)):
                cp = os.path.join(dp, c)
                if not os.path.isdir(cp):
                    continue
                pngs = sorted(glob.glob(os.path.join(cp, "*.png")))
                if not pngs:
                    continue
                mid = pngs[8] if len(pngs) > 8 else pngs[0]
                txt = mid[:-4] + ".txt"
                if os.path.exists(txt):
                    case_map[f"{d}_{c}"] = txt
    return case_map


CASE_MAP = build_case_map()


def parse_yolo(txt_path, img_w, img_h):
    """Return pixel box (x1,y1,x2,y2) and normalized center (cx,cy)."""
    with open(txt_path) as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    cx, cy, w, h = map(float, parts[1:5])
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return (x1, y1, x2, y2), (cx, cy, w, h)


def crop_window(cx_norm, cy_norm, img_w, img_h, size=CROP_SIZE):
    """CROP_SIZE window centered on the box center, clamped to image bounds."""
    x0 = int(round(cx_norm * img_w)) - size // 2
    y0 = int(round(cy_norm * img_h)) - size // 2
    x0 = max(0, min(img_w - size, x0))
    y0 = max(0, min(img_h - size, y0))
    return x0, y0


def draw_wireframe(ax, w, h, d, color="0.4", lw=0.8, scale=(1.0, 1.0, 1.0)):
    """Draw 3D bounding-box wireframe, optional per-axis scale."""
    sx, sy, sz = scale
    corners = [np.array(c) * np.array([sx, sy, sz]) for c in product([0, w], [0, h], [0, d])]
    for i in range(8):
        for j in range(i + 1, 8):
            diff = np.abs(corners[i] - corners[j]).sum()
            if diff == max(w, h, d):
                xs = [corners[i][0], corners[j][0]]
                ys = [corners[i][1], corners[j][1]]
                zs = [corners[i][2], corners[j][2]]
                ax.plot(xs, ys, zs, color=color, lw=lw)


SLICE_PLANE_COLOR = "cyan"


def draw_slice_plane(ax, mask_volume, slice_idx, z_scale=1.0):
    """Show where the slice slider currently is, inside the 3D cube.

    Draws a translucent quad across the full cross-section at the active slice,
    plus the mask voxels on that slice so the vessel cross-section is visible in
    3D context rather than only in the 2D panels.

    `z_scale` differs per window: the dots view uses raw slice indices for z,
    while the smooth view builds its mesh with spacing (1.5, 1, 1), so the same
    slice sits at a different z there.

    Takes the whole volume so the slice index is clamped here - callers must
    never index with an unvalidated value.
    """
    if slice_idx is None:
        return
    n_sl, h, w = mask_volume.shape
    if n_sl <= 0:
        return
    slice_idx = int(np.clip(slice_idx, 0, n_sl - 1))
    z_pos = slice_idx * z_scale

    # Full cross-section quad at the active slice.
    quad_z = np.full((2, 2), float(z_pos))
    ax.plot_surface(
        [0, w], [0, h], quad_z,
        color=SLICE_PLANE_COLOR, alpha=0.15, edgecolor=SLICE_PLANE_COLOR,
        linewidth=0.8, shade=False, zorder=1,
    )

    # Mark the mask voxels on this slice so the plane is not just a marker.
    yy, xx = np.nonzero(np.asarray(mask_volume[slice_idx]) > 0)
    if len(xx) > 0:
        ax.scatter(xx, yy, np.full(len(xx), z_pos), c=SLICE_PLANE_COLOR,
                   s=16, alpha=0.95, depthshade=False, linewidths=0,
                   zorder=2)


class ViewerState:
    pass


def load_case(i):
    """Load cropped volume + mask + box info for case index i.

    Volumes on disk are already cropped to
    (SLICES_PER_CASE, CROP_SIZE, CROP_SIZE) by the
    pipeline, with the crop origin stored in the NIfTI affine offset.
    """
    st = ViewerState()
    nii = nib.load(images[i])
    vol = nii.get_fdata()
    st.name = os.path.basename(images[i]).replace(".nii.gz", "")

    # Crop origin in original-image space (written by the pipeline).
    st.x0 = int(round(float(nii.affine[0, 3])))
    st.y0 = int(round(float(nii.affine[1, 3])))

    # Crop size from the volume itself (already cropped on disk).
    st.crop = vol.shape[2]

    # Parse YOLO box in ORIGINAL image space, then shift to crop space.
    txt = CASE_MAP.get(st.name)
    if txt and os.path.exists(txt):
        st.box, st.norm = parse_yolo(txt, ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE)
    else:
        st.box = None
        st.norm = (0.5, 0.5, 0.0, 0.0)

    if st.box:
        # box = (x1,y1,x2,y2) in original space -> crop space
        bx1, by1, bx2, by2 = st.box
        st.box_crop = (bx1 - st.x0, by1 - st.y0, bx2 - st.x0, by2 - st.y0)
    else:
        st.box_crop = None

    st.vol = vol
    st.roi_abs = (st.x0, st.y0, st.x0 + st.crop, st.y0 + st.crop)
    return st


def case_params(case_id):
    """Per-case saved parameters from config overrides, else global defaults."""
    ov = (CFG.get("case_overrides") or {}).get(case_id) or {}
    if not isinstance(ov, dict):
        ov = {"method": ov}
    vov = ov.get("vesselness") or {}
    return {
        "method": ov.get("method", DEFAULTS["method"]),
        "threshold": ov.get("threshold", DEFAULTS["threshold"]),
        "gamma": vov.get("gamma", DEFAULTS["gamma"]),
        "otsu_scale": ov.get("otsu_scale", DEFAULTS["otsu_scale"]),
    }


def build_mask_from(st, params):
    return build_crop_mask(
        st.vol,
        method=params["method"],
        threshold=params["threshold"],
        gamma=params["gamma"],
        otsu_scale=params["otsu_scale"],
    ).astype(np.float64)


def current_mask(st):
    """Rebuild the mask from the live slider/mode state."""
    return build_mask_from(st, {
        "method": get_method(),
        "threshold": thr_slider.val,
        "gamma": gamma_slider.val,
        "otsu_scale": scale_slider.val,
    })


state = load_case(0)
n_sl = state.vol.shape[0]
sl = n_sl // 2
mask = build_mask_from(state, case_params(state.name))

# =========================== WINDOW 1: 2D ===========================
fig2d = plt.figure(figsize=(12, 4.5))
ax_img = fig2d.add_subplot(1, 3, 1)
ax_overlay = fig2d.add_subplot(1, 3, 2)
ax_mask = fig2d.add_subplot(1, 3, 3)
fig2d.subplots_adjust(bottom=0.32, top=0.90, left=0.05, right=0.98)

disp = state.vol[sl].astype(np.float32)
vmin, vmax = np.percentile(disp, [2, 98])
disp_norm = np.clip((disp - vmin) / (vmax - vmin + 1e-8), 0, 1)

im0 = ax_img.imshow(disp_norm, cmap="gray")
ax_img.set_title("Cropped Image")
ax_img.axis("off")

im1 = ax_overlay.imshow(disp_norm, cmap="gray")
im1c = ax_overlay.imshow(mask[sl], cmap="autumn", alpha=0.45)
ax_overlay.set_title("Overlay")
ax_overlay.axis("off")

im2 = ax_mask.imshow(mask[sl], cmap="gray")
ax_mask.set_title("Mask")
ax_mask.axis("off")

box_rect = None
if state.box_crop:
    x1, y1, x2, y2 = state.box_crop
    crop = state.crop
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bw = max(0, min(crop, x2) - bx1)
    bh = max(0, min(crop, y2) - by1)
    if bw > 0 and bh > 0:
        box_rect = Rectangle((bx1, by1), bw, bh, linewidth=1.5,
                             edgecolor="red", facecolor="none", linestyle="--")
        ax_img.add_patch(box_rect)

# =========================== WINDOW 2: 3D V1 (dots) ===========================
fig3d_v1 = plt.figure(figsize=(6, 6))
ax_3d_v1 = fig3d_v1.add_subplot(111, projection="3d")
ax_3d_v1.set_title("V1_3D Mask (dots)")
ax_3d_v1.set_xlabel("X")
ax_3d_v1.set_ylabel("Y")
ax_3d_v1.set_zlabel("Slice")
ax_3d_v1.view_init(elev=25, azim=-60)
fig3d_v1.tight_layout()

dots3d = [None]


def render_3d_v1(mask_volume, slice_idx=None):
    if dots3d[0] is not None:
        dots3d[0].remove()
        dots3d[0] = None
    ax_3d_v1.clear()

    n_sl3, h, w = mask_volume.shape
    ax_3d_v1.set_xlim(0, w)
    ax_3d_v1.set_ylim(h, 0)
    ax_3d_v1.set_zlim(0, n_sl3)

    z, y, x = np.nonzero(mask_volume > 0)
    if len(x) > 0:
        norm = plt.Normalize(z.min(), z.max())
        colors = plt.cm.viridis_r(norm(z))
        dots3d[0] = ax_3d_v1.scatter(x, y, z, c=colors, s=45, alpha=0.9)

    # Uniform scale: z is the raw slice index, so scaling z by 1.5 pushed the
    # box 8.5 units past zlim and clipped its top off.
    draw_wireframe(ax_3d_v1, w, h, n_sl3, scale=(1.0, 1.0, 1.0))
    draw_slice_plane(ax_3d_v1, mask_volume, slice_idx, z_scale=1.0)
    ax_3d_v1.set_title(
        f"V1_3D Mask (dots)  |  voxels={int(mask_volume.sum())}"
        + (f"  |  slice {int(slice_idx)}" if slice_idx is not None else ""))
    ax_3d_v1.view_init(elev=25, azim=-60)


# =========================== WINDOW 3: 3D V2 (smooth) ===========================
fig3d = plt.figure(figsize=(6, 6))
ax_3d = fig3d.add_subplot(111, projection="3d")
ax_3d.set_title("V2_3D Mask (smooth)")
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Slice")
ax_3d.view_init(elev=25, azim=-60)
fig3d.tight_layout()

surface3d = [None]


def render_3d(mask_volume, slice_idx=None):
    if surface3d[0] is not None:
        surface3d[0].remove()
        surface3d[0] = None
    ax_3d.clear()

    n_sl3, h, w = mask_volume.shape
    if int(mask_volume.sum()) < 8:
        ax_3d.set_xlim(0, w); ax_3d.set_ylim(h, 0); ax_3d.set_zlim(0, n_sl3)
        draw_wireframe(ax_3d, w, h, n_sl3, scale=(1.0, 1.0, 1.0))
        # Small masks are exactly the severe-stenosis cases where knowing the
        # slice position matters most, so keep the plane on this branch too.
        draw_slice_plane(ax_3d, mask_volume, slice_idx, z_scale=1.0)
        ax_3d.set_title(
            f"V2_3D Mask (smooth)  |  voxels={int(mask_volume.sum())}"
            + (f"  |  slice {int(slice_idx)}" if slice_idx is not None else ""))
        ax_3d.view_init(elev=25, azim=-60)
        return

    # Smooth the binary mask a bit, then build an iso-surface with marching cubes.
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter

    vol_s = gaussian_filter(mask_volume.astype(np.float64), sigma=0.6, mode="constant")
    spacing = (1.5, 1.0, 1.0)  # z slices are thicker than xy pixels
    verts, faces, normals, _ = marching_cubes(
        vol_s, level=0.5, spacing=spacing, gradient_direction="ascent"
    )

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = Poly3DCollection(verts[faces], alpha=0.9, linewidth=0, edgecolor="none")
    mesh.set_facecolor((0.55, 0.55, 0.55, 1.0))  # gray
    mesh.set_edgecolor("none")
    ax_3d.add_collection3d(mesh)
    surface3d[0] = mesh

    ax_3d.auto_scale_xyz([0, w * spacing[1]], [0, h * spacing[1]], [0, n_sl3 * spacing[0]])
    draw_wireframe(ax_3d, w, h, n_sl3, scale=spacing)
    # The mesh was built with spacing (1.5, 1, 1), so slice z is 1.5x the index.
    draw_slice_plane(ax_3d, mask_volume, slice_idx, z_scale=spacing[0])
    ax_3d.set_title(
        f"V2_3D Mask (smooth)  |  voxels={int(mask_volume.sum())}"
        + (f"  |  slice {int(slice_idx)}" if slice_idx is not None else ""))
    ax_3d.view_init(elev=25, azim=-60)


def render_all_3d(mask_volume, slice_idx=None):
    render_3d_v1(mask_volume, slice_idx)
    render_3d(mask_volume, slice_idx)


# =========================== TITLE ===========================
title = fig2d.suptitle("", fontsize=11)

def estimate_diameter(mask):
    """Return (diameter_px, max_slice_area_fraction) for the visible mask.

    Diameter is the distance-transform radius measured *along the 3D skeleton*,
    taken at the 90th percentile and doubled. Measuring over the whole volume
    badly under-reports, because the boundary shell dominates the voxel count;
    measuring along the skeleton tracks the medial axis and stays within ~5%
    of truth for tubes and spheres of radius 2-6 px (verified on synthetic
    phantoms). This is also the quantity the centerline module will use, so
    the preview readout and the final measurement agree.
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return None, 0.0
    dt = distance_transform_edt(m)
    sk = skeletonize(m)
    on_axis = dt[sk] if sk.any() else dt[m]
    if on_axis.size == 0:
        return None, 0.0
    dia = 2.0 * float(np.percentile(on_axis, 90))

    per_slice = m.reshape(m.shape[0], -1).sum(1)
    nz = per_slice[per_slice > 0]
    area_frac = float(nz.max()) / float(m.shape[1] * m.shape[2]) if len(nz) else 0.0
    return dia, area_frac


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def case_done_map():
    """Map case index -> bool, read straight from the manual manifest."""
    man = load_manifest()
    return {i: (os.path.basename(images[i]).replace(".nii.gz", "") in man)
            for i in range(len(images))}


def load_ignored():
    """Cases the annotator marked as having no stenosis (excluded from training)."""
    if not os.path.exists(IGNORE_PATH):
        return {}
    try:
        with open(IGNORE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_ignored(data):
    """Write atomically so a crash mid-write cannot lose the ignore list."""
    tmp = IGNORE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, IGNORE_PATH)


def case_name(i):
    return os.path.basename(images[i]).replace(".nii.gz", "")


# =========================== WIDGETS (in window 1) ===========================
from matplotlib.widgets import RadioButtons
ax_gamma = fig2d.add_axes([0.15, 0.31, 0.7, 0.03])
ax_thr = fig2d.add_axes([0.15, 0.25, 0.7, 0.03])
ax_scale = fig2d.add_axes([0.15, 0.19, 0.7, 0.03])
ax_sl = fig2d.add_axes([0.15, 0.13, 0.7, 0.03])
ax_prev = fig2d.add_axes([0.03, 0.03, 0.11, 0.055])
ax_next = fig2d.add_axes([0.155, 0.03, 0.11, 0.055])
ax_unlab = fig2d.add_axes([0.28, 0.03, 0.15, 0.055])
ax_save = fig2d.add_axes([0.44, 0.03, 0.19, 0.055])
ax_ign = fig2d.add_axes([0.645, 0.03, 0.15, 0.055])
ax_meth = fig2d.add_axes([0.81, 0.03, 0.16, 0.055])
fig2d.subplots_adjust(bottom=0.38)

readout = fig2d.text(0.5, 0.103, "", ha="center", va="bottom", fontsize=9)
helptext = fig2d.text(
    0.5, 0.095,
    "keys:  <- -> case   [ ] slice   1/2 method   G/T/O gamma/thr/scale   S save   I ignore",
    ha="center", va="bottom", fontsize=7.5, color="0.4")

gamma_slider = Slider(ax_gamma, "Gamma", 0.1, 5.0,
                      valinit=DEFAULTS["gamma"], valstep=0.05)
thr_slider = Slider(ax_thr, "Threshold", 0.01, 1.0,
                    valinit=DEFAULTS["threshold"], valstep=0.005)
scale_slider = Slider(ax_scale, "Otsu Scale", 0.30, 1.20,
                      valinit=DEFAULTS["otsu_scale"], valstep=0.01)
sl_slider = Slider(ax_sl, "Slice", 0, n_sl - 1, valinit=sl, valstep=1)
btn_prev = Button(ax_prev, "< Prev Case")
btn_next = Button(ax_next, "Next Case >")
btn_unlab = Button(ax_unlab, "Next Unlabeled >")
method_radio = RadioButtons(ax_meth, ("otsu", "frangi"),
                            active=METHOD_RADIO_INDEX.get(DEFAULTS["method"], 0))
btn_save = Button(ax_save, "Save Manual Label")
btn_ign = Button(ax_ign, "Ignore Case")

case_idx = [0]
SUPPRESS = set()


def set_sliders_from_case(st):
    """Restore this case's saved parameters, suppressing re-entrant updates."""
    global SUPPRESS
    p = case_params(st.name)
    SUPPRESS = {"gamma", "threshold", "otsu_scale"}
    try:
        method_radio.set_active(METHOD_RADIO_INDEX.get(p["method"], 0))
        gamma_slider.set_val(p["gamma"])
        thr_slider.set_val(p["threshold"])
        scale_slider.set_val(p["otsu_scale"])
    finally:
        SUPPRESS = set()


def make_display(vol_slice):
    vmin, vmax = np.percentile(vol_slice, [2, 98])
    return np.clip((vol_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)


def get_method():
    return method_radio.value_selected


def update_plot():
    global mask
    mask = current_mask(state)
    s = int(sl_slider.val)
    disp = make_display(state.vol[s].astype(np.float32))
    im0.set_data(disp)
    im1.set_data(disp)
    im1c.set_data(mask[s])
    im2.set_data(mask[s])
    render_all_3d(mask, s)
    title.set_text(
        f"{state.name}  |  crop=({state.x0},{state.y0})  |  slice {s}/{n_sl-1}  |  "
        f"mode={get_method()}  |  thr={thr_slider.val:.3f}  |  gamma={gamma_slider.val:.2f}"
        f"  |  otsu_scale={scale_slider.val:.2f}  |  voxels={int(mask.sum())}")

    dia, area_frac = estimate_diameter(mask)
    done = case_done_map()
    n_done = sum(1 for v in done.values() if v)
    if dia is None:
        readout.set_text("EMPTY MASK  |  labeled %d/%d" % (n_done, len(images)))
        readout.set_color("darkred")
    else:
        # Only flag physically implausible geometry. An earlier area-based
        # warning fired on 94% of cases and was useless; cerebral arteries are
        # 3-6 px here, so >12 px means the mask bled into surrounding tissue.
        flags = []
        if dia > MAX_PLAUSIBLE_DIA_PX:
            flags.append("DIA > %.0f px - lower thr / raise scale" % MAX_PLAUSIBLE_DIA_PX)
        if area_frac >= 0.999:
            flags.append("MASK FILLED THE CROP")
        readout.set_text(
            "lumen dia ~ %.1f px   |   widest slice %.0f%%   |   "
            "labeled %d/%d   |   %s%s"
            % (dia, area_frac * 100, n_done, len(images),
               "SAVED" if done.get(case_idx[0]) else "not saved",
               ("   <-- " + "; ".join(flags)) if flags else ""))
        readout.set_color("darkred" if flags else "black")
    fig2d.canvas.draw_idle()
    fig3d_v1.canvas.draw_idle()
    fig3d.canvas.draw_idle()


def on_thr_change(val):
    if "threshold" not in SUPPRESS:
        update_plot()


def on_gamma_change(val):
    if "gamma" not in SUPPRESS:
        update_plot()


def on_scale_change(val):
    if "otsu_scale" not in SUPPRESS:
        update_plot()


def on_sl_change(val):
    update_plot()


def load_new_case(idx):
    global state, n_sl, mask, box_rect
    state = load_case(idx)
    btn_save.label.set_text("Save Manual Label")
    btn_save.color = "0.85"
    if state.name in load_ignored():
        btn_ign.label.set_text("IGNORED")
        btn_ign.color = "0.95"
    else:
        btn_ign.label.set_text("Ignore Case")
        btn_ign.color = "0.85"
    n_sl = state.vol.shape[0]
    sl_slider.valmax = n_sl - 1
    sl_slider.ax.set_xlim(0, n_sl - 1)
    sl_slider.set_val(n_sl // 2)
    set_sliders_from_case(state)
    if box_rect is not None:
        box_rect.remove()
        box_rect = None
    if state.box_crop:
        x1, y1, x2, y2 = state.box_crop
        crop = state.crop
        bx1 = max(0, x1)
        by1 = max(0, y1)
        bw = max(0, min(crop, x2) - bx1)
        bh = max(0, min(crop, y2) - by1)
        if bw > 0 and bh > 0:
            box_rect = Rectangle((bx1, by1), bw, bh, linewidth=1.5,
                                 edgecolor="red", facecolor="none",
                                 linestyle="--")
            ax_img.add_patch(box_rect)
    update_plot()


def prev_case(event):
    case_idx[0] = max(0, case_idx[0] - 1)
    load_new_case(case_idx[0])


def next_case(event):
    case_idx[0] = min(len(images) - 1, case_idx[0] + 1)
    load_new_case(case_idx[0])


def toggle_ignore(event=None):
    """Mark the current case as 'no stenosis, exclude from training', or undo.

    Toggling means an accidental ignore is always one keypress away from being
    reverted, without navigating anywhere.
    """
    data = load_ignored()
    name = state.name
    if name in data:
        del data[name]
        btn_ign.label.set_text("Ignore Case")
        btn_ign.color = "0.85"
        print(f"Un-ignored (back in the training set): {name}")
    else:
        data[name] = {
            "reason": "no stenosis",
            "ignored_at": __import__("datetime").datetime.now().isoformat(
                timespec="seconds"),
        }
        btn_ign.label.set_text("IGNORED")
        btn_ign.color = "0.95"
        print(f"Ignored (excluded from training): {name}")
    save_ignored(data)
    update_plot()


def next_unlabeled(event=None):
    """Jump forward to the next case that has no manual label saved."""
    done = case_done_map()
    ignored = load_ignored()
    n = len(images)
    for step in range(1, n + 1):
        j = (case_idx[0] + step) % n
        if not done.get(j, False) and case_name(j) not in ignored:
            if j != case_idx[0]:
                case_idx[0] = j
                load_new_case(j)
            return
    print("All cases are either labelled or ignored.")


def nudge(widget, delta):
    widget.set_val(float(np.clip(widget.val + delta, widget.valmin, widget.valmax)))


def on_key(event):
    k = (event.key or "").lower()
    if k in ("right", "n"):
        next_case(event)
    elif k in ("left", "p"):
        prev_case(event)
    elif k in ("u",):
        next_unlabeled(event)
    elif k == "s":
        save_manual_label(event)
    elif k in ("i",):
        toggle_ignore(event)
    elif k == "1":
        method_radio.set_active(0)
    elif k == "2":
        method_radio.set_active(1)
    elif k == "]":
        sl_slider.set_val(min(sl_slider.valmax, int(sl_slider.val) + 1))
    elif k == "[":
        sl_slider.set_val(max(0, int(sl_slider.val) - 1))
    elif k == "g":
        nudge(gamma_slider, 0.05)
    elif k == "t":
        nudge(thr_slider, 0.005)
    elif k == "o":
        nudge(scale_slider, 0.01)
    else:
        return
    fig2d.canvas.draw_idle()


def case_class_name(st):
    """Look up the YOLO class name for this case."""
    txt = CASE_MAP.get(st.name)
    if not txt or not os.path.exists(txt):
        return None
    try:
        with open(txt) as f:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) < 5:
            return None
        cls_idx = int(float(parts[0]))
    except Exception:
        return None
    classes_file = os.path.join(BASE, CFG.get("classes_file", "classes.txt"))
    try:
        with open(classes_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names[cls_idx] if 0 <= cls_idx < len(names) else f"cls{cls_idx}"
    except Exception:
        return f"cls{cls_idx}"


def save_manual_label(event):
    """Write the on-screen mask as a training label in the manual dataset."""
    os.makedirs(MANUAL_IMG_DIR, exist_ok=True)
    os.makedirs(MANUAL_LBL_DIR, exist_ok=True)

    method = get_method()
    thr = float(thr_slider.val)
    gam = float(gamma_slider.val)
    osc = float(scale_slider.val)

    # Label: uint8 NIfTI reusing the image affine (preserves crop origin).
    affine = nib.load(images[case_idx[0]]).affine
    lbl_path = os.path.join(MANUAL_LBL_DIR, f"{state.name}.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), lbl_path)

    # Image: exact copy of the pipeline cube so the pair is self-contained.
    img_src = images[case_idx[0]]
    img_path = os.path.join(MANUAL_IMG_DIR, f"{state.name}.nii.gz")
    src_nii = nib.load(img_src)
    nib.save(nib.Nifti1Image(np.asanyarray(src_nii.dataobj).astype(np.float32),
                             src_nii.affine), img_path)

    record = {
        "case_id": state.name,
        "method": method,
        "threshold": round(thr, 4),
        "vesselness": {"gamma": round(gam, 4)},
        "otsu_scale": round(osc, 4),
        "voxels": int(mask.sum()),
        "class_name": case_class_name(state),
        "crop_origin": [int(state.x0), int(state.y0)],
        "shape": [int(v) for v in mask.shape],
        "source_image": os.path.basename(img_src),
        "saved_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    manifest[state.name] = record
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Also record params in config.json so batch_run --redo-overrides matches.
    cfg = load_config()
    overrides = cfg.setdefault("case_overrides", {})
    overrides[state.name] = {
        "method": method,
        "threshold": round(thr, 4),
        "vesselness": {"gamma": round(gam, 4)},
        "otsu_scale": round(osc, 4),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    # Saving a label means the case IS part of the training set, so drop any
    # earlier "no stenosis" ignore rather than leaving the two contradicting.
    ign = load_ignored()
    if state.name in ign:
        del ign[state.name]
        save_ignored(ign)
        btn_ign.label.set_text("Ignore Case")
        btn_ign.color = "0.85"
        print(f"  (cleared ignore flag for {state.name})")

    print(f"Saved manual label: {state.name} | method={method} | "
          f"thr={thr:.3f} gamma={gam:.2f} otsu_scale={osc:.2f} | "
          f"voxels={int(mask.sum())}")
    btn_save.label.set_text(f"Saved: {state.name}")
    fig2d.canvas.draw_idle()


set_sliders_from_case(state)
update_plot()

thr_slider.on_changed(on_thr_change)
gamma_slider.on_changed(on_gamma_change)
scale_slider.on_changed(on_scale_change)
sl_slider.on_changed(on_sl_change)
method_radio.on_clicked(lambda val: update_plot())
btn_prev.on_clicked(prev_case)
btn_next.on_clicked(next_case)
btn_unlab.on_clicked(next_unlabeled)
btn_save.on_clicked(save_manual_label)
btn_ign.on_clicked(toggle_ignore)

fig2d.canvas.mpl_connect("key_press_event", on_key)
fig2d.canvas.draw_idle()

plt.show()