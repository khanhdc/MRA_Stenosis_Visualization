"""Interactive viewer: scroll through slices and cases."""
import os, sys, glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

BASE = os.path.join(os.path.dirname(__file__), "training_data")
IMG_DIR = os.path.join(BASE, "images")
LBL_DIR = os.path.join(BASE, "labels")

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))
if not images:
    print("No images found."); sys.exit(1)

def load(i):
    img = nib.load(images[i]).get_fdata()
    name = os.path.basename(images[i]).replace(".nii.gz","")
    lbl_path = os.path.join(LBL_DIR, os.path.basename(images[i]))
    lbl = nib.load(lbl_path).get_fdata() if os.path.exists(lbl_path) else np.zeros_like(img)
    return img, lbl, name

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(bottom=0.25)

data = {"idx": 0}
img, lbl, name = load(0)
sdim = img.shape[0]
sl = sdim // 2

im0 = axes[0].imshow(img[sl], cmap="gray"); axes[0].set_title("Image"); axes[0].axis("off")
im1 = axes[1].imshow(img[sl], cmap="gray")
im1c = axes[1].imshow(lbl[sl], cmap="autumn", alpha=0.4); axes[1].set_title("Overlay"); axes[1].axis("off")
im2 = axes[2].imshow(lbl[sl], cmap="gray"); axes[2].set_title("Mask"); axes[2].axis("off")
title = fig.suptitle(f"{name}  |  slice {sl}/{sdim-1}", fontsize=13)

ax_slice = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_case = plt.axes([0.15, 0.05, 0.7, 0.03])
sl_slider = Slider(ax_slice, "Slice", 0, sdim-1, valinit=sl, valstep=1)
case_slider = Slider(ax_case, "Case", 0, len(images)-1, valinit=0, valstep=1)

def update_slice(val):
    sl = int(sl_slider.val)
    im0.set_data(img[sl]); im1.set_data(img[sl]); im1c.set_data(lbl[sl]); im2.set_data(lbl[sl])
    title.set_text(f"{name}  |  slice {sl}/{sdim-1}"); fig.canvas.draw_idle()

def update_case(val):
    data["idx"] = int(case_slider.val)
    global img, lbl, name, sdim
    img, lbl, name = load(data["idx"])
    sdim = img.shape[0]
    sl_slider.valmax = sdim - 1; sl_slider.ax.set_xlim(0, sdim-1)
    sl = sdim // 2; sl_slider.set_val(sl)
    im0.set_data(img[sl]); im1.set_data(img[sl]); im1c.set_data(lbl[sl]); im2.set_data(lbl[sl])
    title.set_text(f"{name}  |  slice {sl}/{sdim-1}"); fig.canvas.draw_idle()

sl_slider.on_changed(update_slice)
case_slider.on_changed(update_case)
plt.show()
