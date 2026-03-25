import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FT2M = 0.3048
with open(os.path.join(_SCRIPT_DIR, "japanese_pitchers_ff_2025_sz.json")) as f:
    all_data = json.load(f)
with open(os.path.join(_SCRIPT_DIR, "japanese_pitchers_omega.json")) as f:
    omega_data = json.load(f)

lefties = {"Matsui", "Kikuchi", "Imanaga"}

def do_pca(x, z):
    data = np.column_stack([x, z])
    mean = data.mean(axis=0)
    cov = np.cov((data - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    return mean, evals[idx], evecs[:, idx], evals[idx] / evals[idx].sum()

def draw_pca_overlay(ax, mean, evals, evecs, evratio, scale, ls, alpha):
    angle_pc1 = np.degrees(np.arctan2(evecs[1,0], evecs[0,0]))
    chi2_95 = 5.991
    w = 2*np.sqrt(evals[0]*chi2_95)
    h = 2*np.sqrt(evals[1]*chi2_95)
    ell = Ellipse(xy=mean, width=w, height=h, angle=angle_pc1,
                  edgecolor="purple", facecolor="none", linewidth=1.8, linestyle=ls, alpha=alpha)
    ax.add_patch(ell)
    colors = ["blue","green"]
    labels = ["PC1","PC2"]
    for i in range(2):
        comp = evecs[:,i]
        length = scale * np.sqrt(evals[i])
        p1 = mean - comp*length
        p2 = mean + comp*length
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=colors[i], linewidth=2, linestyle=ls, alpha=alpha)
        if ls == "-":
            ax.text(p2[0]+0.02, p2[1]+0.02, "{} ({:.0f}%)".format(labels[i], evratio[i]*100),
                    color=colors[i], fontsize=8, fontweight="bold", alpha=alpha)

def get_pc1_angle(name, d):
    x = np.array(d["x"])*FT2M
    z = np.array(d["z"])*FT2M
    _, _, evecs, _ = do_pca(x, z)
    if name in lefties:
        ev = evecs.copy(); ev[0,:] = -ev[0,:]
        return np.degrees(np.arctan2(ev[1,0], ev[0,0]))
    return np.degrees(np.arctan2(evecs[1,0], evecs[0,0]))

def plot_panel(ax, name, d, od, pc1_angle):
    x_ft = np.array(d["x"])
    z_ft = np.array(d["z"])
    x = x_ft * FT2M
    z = z_ft * FT2M
    n = d["n"]
    is_lefty = name in lefties

    sz_top_m = d["sz_top"]*FT2M
    sz_bot_m = d["sz_bot"]*FT2M
    sz_left_m = -0.708*FT2M
    sz_right_m = 0.708*FT2M

    in_zone = np.sum((x_ft >= -0.708) & (x_ft <= 0.708) & (z_ft >= d["sz_bot"]) & (z_ft <= d["sz_top"]))
    in_pct = in_zone/n*100
    out_pct = 100-in_pct

    sz = patches.Rectangle((sz_left_m, sz_bot_m), sz_right_m-sz_left_m, sz_top_m-sz_bot_m,
                            linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--")
    ax.add_patch(sz)

    hw = 0.708*FT2M
    plate_x = [0,hw,hw,-hw,-hw,0]
    plate_y = np.array([0,0.35,0.71,0.71,0.35,0])*FT2M - 0.15
    ax.plot(plate_x, plate_y, "k-", linewidth=1)

    mean, evals, evecs, evratio = do_pca(x, z)
    scale = 2.5

    if is_lefty:
        ax.scatter(-x, z, alpha=0.25, s=10, c="red", edgecolors="darkred", linewidth=0.2)
        mean_m = np.array([-mean[0], mean[1]])
        evecs_m = evecs.copy(); evecs_m[0,:] = -evecs_m[0,:]
        ax.plot(mean_m[0], mean_m[1], "ko", markersize=5, zorder=5)
        draw_pca_overlay(ax, mean_m, evals, evecs_m, evratio, scale, "-", 1.0)
        ax.scatter(x, z, alpha=0.1, s=8, c="orange", edgecolors="darkorange", linewidth=0.2)
        ax.plot(mean[0], mean[1], "o", color="gray", markersize=4, zorder=5, alpha=0.5)
        draw_pca_overlay(ax, mean, evals, evecs, evratio, scale, "--", 0.4)
        title_suffix = " [L]"
    else:
        ax.scatter(x, z, alpha=0.25, s=10, c="red", edgecolors="darkred", linewidth=0.2)
        ax.plot(mean[0], mean[1], "ko", markersize=5, zorder=5)
        draw_pca_overlay(ax, mean, evals, evecs, evratio, scale, "-", 1.0)
        title_suffix = " [R]"

    ax.text(0, 0.03, "Zone: {:.1f}% | Out: {:.1f}%".format(in_pct, out_pct),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.85))

    if od["n"] > 0:
        omega_text = "|$\\omega_X$|:{:.0f}% |$\\omega_Y$|:{:.0f}% |$\\omega_Z$|:{:.0f}%".format(
            od["ox_mean"]*100, od["oy_mean"]*100, od["oz_mean"]*100)
        ax.text(0, 1.68, omega_text, ha="center", va="bottom", fontsize=16, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a5276", alpha=0.85))

    ax.set_title("{} (n={}){} PC1={:.0f}deg".format(name, n, title_suffix, pc1_angle),
                 fontsize=9, fontweight="bold")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 1.85)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

all_names = ["Darvish","Ohtani","Yamamoto","Sasaki","Matsui","Sugano","Senga","Kikuchi","Imanaga"]
name_angles = [(n, get_pc1_angle(n, all_data[n])) for n in all_names]
name_angles.sort(key=lambda x: x[1], reverse=True)
order = [na[0] for na in name_angles]
angle_map = dict(name_angles)

fig, axes = plt.subplots(3, 3, figsize=(15, 19))
for i, name in enumerate(order):
    row, col = divmod(i, 3)
    ax = axes[row][col]
    plot_panel(ax, name, all_data[name], omega_data[name], angle_map[name])
    if col == 0:
        ax.set_ylabel("Vertical (m)", fontsize=9)
    if row == 2:
        ax.set_xlabel("Horizontal (m)", fontsize=9)

fig.suptitle("Japanese MLB Pitchers 2025 FF (sorted by PC1 angle)\nPCA, 95% Ellipse, Zone Rate & $|\\omega_i|/|\\omega|$", fontsize=12, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(_SCRIPT_DIR, "japanese_pitchers_sorted.png"), dpi=150)
print("Plot saved")
