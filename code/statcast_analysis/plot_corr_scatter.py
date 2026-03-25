"""
日本人MLB投手9名の2025年FF
ΔRelease Height vs Δvz0(km/h) — 相関係数|r|でソート、散布図+回帰直線
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FT2CM = 30.48
FTS2KMH = 1.09728


def main():
    with open(os.path.join(SCRIPT_DIR, "japanese_pitchers_ff_vy0.json")) as f:
        data = json.load(f)

    # Compute per pitcher
    pitchers = []
    for name, d in data.items():
        if d["n"] < 10:
            continue
        rz = np.array(d["release_z"])
        vz = np.array(d["vz0"])
        dz_cm = (rz - rz.mean()) * FT2CM
        dvz_kmh = (vz - vz.mean()) * FTS2KMH
        r = np.corrcoef(dz_cm, dvz_kmh)[0, 1]
        pitchers.append({
            "name": name, "n": d["n"],
            "dz_cm": dz_cm, "dvz_kmh": dvz_kmh,
            "r": r,
            "mean_rz_m": rz.mean() * FT2CM / 100,
            "mean_vz_kmh": vz.mean() * FTS2KMH,
        })

    # Sort by |r| descending
    pitchers.sort(key=lambda x: abs(x["r"]), reverse=True)

    # Unified axis limits
    all_vz = np.concatenate([p["dvz_kmh"] for p in pitchers])
    all_dz = np.concatenate([p["dz_cm"] for p in pitchers])
    vz_margin = (all_vz.max() - all_vz.min()) * 0.1
    xlim = (all_vz.min() - vz_margin, all_vz.max() + vz_margin)
    dz_max = max(abs(all_dz.min()), abs(all_dz.max())) * 1.3
    ylim = (-dz_max, dz_max)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    fig.suptitle("Japanese MLB Pitchers 2025 — FF (4-Seam Fastball)\n"
                 "ΔRelease Height (cm) vs. Δvz0 (km/h) — sorted by |r|",
                 fontsize=16, fontweight="bold", y=0.99)

    for idx, p in enumerate(pitchers):
        ax = axes[idx // 3][idx % 3]
        x = p["dvz_kmh"]
        y = p["dz_cm"]

        ax.scatter(x, y, s=4, alpha=0.3, color="steelblue", zorder=2)

        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([xlim[0], xlim[1]])
        ax.plot(x_line, slope * x_line + intercept, color="red", lw=2, zorder=3)

        # Title
        r_color = "red" if abs(p["r"]) > 0.1 else "gray"
        title = f"{p['name']} (n={p['n']})"
        ax.set_title(title, fontsize=13, fontweight="bold")

        ax.set_xlabel("Δvz0 (km/h)", fontsize=11)
        ax.set_ylabel("ΔRelease Height (cm)", fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")

        # Stats
        r_str = f"r = {p['r']:+.3f}"
        stats = (f"{r_str}\n"
                 f"Mean Rz: {p['mean_rz_m']:.2f} m\n"
                 f"Mean vz0: {p['mean_vz_kmh']:.1f} km/h")
        fontsize_r = 16
        # r value prominently
        ax.text(0.98, 0.98, f"r = {p['r']:+.3f}", transform=ax.transAxes,
                fontsize=fontsize_r, fontweight="bold", va="top", ha="right",
                color=r_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        # Other stats smaller
        ax.text(0.02, 0.98, f"Mean Rz: {p['mean_rz_m']:.2f} m\n"
                             f"Mean vz0: {p['mean_vz_kmh']:.1f} km/h",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_corr_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    main()
