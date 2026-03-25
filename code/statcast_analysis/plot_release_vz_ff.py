"""
日本人MLB投手9名の2025年FFについて、
リリースポイントの高さ(平均からの差分) vs 鉛直方向速度成分(vz0)の分布を
3x3グリッドでプロット。cm単位、スケール統一、95%PCA楕円付き。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Japanese MLB pitchers (2025)
PITCHERS = {
    "Darvish":  (506433, "R"),
    "Ohtani":   (660271, "R"),
    "Yamamoto": (808967, "R"),
    "Sasaki":   (808963, "R"),
    "Matsui":   (673513, "L"),
    "Sugano":   (608372, "R"),
    "Senga":    (673540, "R"),
    "Kikuchi":  (579328, "L"),
    "Imanaga":  (684007, "L"),
}

FT2M = 0.3048
FT2CM = 30.48
FTS2CMS = 30.48  # ft/s -> cm/s
FTS2KMH = 1.09728  # ft/s -> km/h


def main():
    # Check for cached full data, then filter FF
    cache_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_release_vz.json")
    ff_cache_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_release_vz_ff.json")

    if os.path.exists(ff_cache_path):
        print("Loading cached FF data...")
        with open(ff_cache_path) as f:
            all_data = json.load(f)
    else:
        # Need to fetch from Statcast
        print("Fetching 2025 Statcast data (FF only)...")
        from pybaseball import statcast
        df = statcast(start_dt="2025-03-20", end_dt="2025-10-01")
        print(f"  Total pitches: {len(df)}")

        all_data = {}
        for name, (pid, p_throws) in PITCHERS.items():
            df_p = df[(df["pitcher"] == pid) & (df["pitch_type"] == "FF")].copy()
            df_p = df_p.dropna(subset=["release_pos_z", "vz0"])
            n = len(df_p)
            print(f"  {name}: {n} FF pitches")

            all_data[name] = {
                "release_z": df_p["release_pos_z"].values.tolist(),
                "vz0": df_p["vz0"].values.tolist(),
                "n": n,
                "p_throws": p_throws,
            }

        with open(ff_cache_path, "w") as f:
            json.dump(all_data, f)
        print(f"Saved cache: {ff_cache_path}")

    # Convert to cm and compute deltas from mean release height
    plot_data = {}
    for name, d in all_data.items():
        if d["n"] < 10:
            plot_data[name] = {"vz0_cm": [], "dz_cm": [], "n": d["n"],
                               "mean_rz_ft": 0, "p_throws": d["p_throws"]}
            continue
        rz = np.array(d["release_z"])  # feet
        vz = np.array(d["vz0"])        # ft/s
        mean_rz = rz.mean()
        mean_vz = vz.mean()
        dz_cm = (rz - mean_rz) * FT2CM       # delta from mean, in cm
        dvz_kmh = (vz - mean_vz) * FTS2KMH    # delta from mean, in km/h
        plot_data[name] = {
            "dvz0_kmh": dvz_kmh.tolist(),
            "dz_cm": dz_cm.tolist(),
            "n": d["n"],
            "mean_rz_ft": float(mean_rz),
            "mean_vz0_kmh": float(mean_vz * FTS2KMH),
            "p_throws": d["p_throws"],
        }

    # Compute PCA angle for sorting
    name_angles = []
    for name, d in plot_data.items():
        if d["n"] < 10:
            name_angles.append((name, 0))
            continue
        X = np.column_stack([d["dvz0_kmh"], d["dz_cm"]])
        pca = PCA(n_components=1)
        pca.fit(X)
        angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
        name_angles.append((name, angle))

    name_angles.sort(key=lambda x: x[1], reverse=True)

    # Determine unified axis limits
    all_vz = []
    all_dz = []
    for name, d in plot_data.items():
        if d["n"] < 10:
            continue
        all_vz.extend(d["dvz0_kmh"])
        all_dz.extend(d["dz_cm"])
    all_vz = np.array(all_vz)
    all_dz = np.array(all_dz)

    # Separate ranges for each axis
    vz_margin = (all_vz.max() - all_vz.min()) * 0.1
    xlim = (all_vz.min() - vz_margin, all_vz.max() + vz_margin)

    dz_max = max(abs(all_dz.min()), abs(all_dz.max())) * 1.3
    ylim = (-dz_max, dz_max)

    # Plot 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    fig.suptitle("Japanese MLB Pitchers 2025 — FF (4-Seam Fastball)\n"
                 "ΔRelease Height vs. Δvz0 (both delta from mean)",
                 fontsize=16, fontweight="bold", y=0.99)

    for idx, (name, pc1_angle) in enumerate(name_angles):
        ax = axes[idx // 3][idx % 3]
        d = plot_data[name]

        if d["n"] < 10:
            ax.set_title(f"{name} (n={d['n']})", fontsize=13)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # ax.set_aspect("equal")  # different units, not needed
            continue

        dvz_kmh = np.array(d["dvz0_kmh"])
        dz_cm = np.array(d["dz_cm"])

        ax.scatter(dvz_kmh, dz_cm, s=4, alpha=0.3, color="steelblue", zorder=2)

        # PCA
        X = np.column_stack([dvz_kmh, dz_cm])
        pca = PCA(n_components=2)
        pca.fit(X)
        center = X.mean(axis=0)

        # 95% confidence ellipse (chi2 with 2 dof, 95% = 5.991)
        chi2_95 = np.sqrt(5.991)
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)
        w = 2 * chi2_95 * np.sqrt(pca.explained_variance_[0])
        h = 2 * chi2_95 * np.sqrt(pca.explained_variance_[1])
        ell = Ellipse(xy=center, width=w, height=h, angle=angle_deg,
                      edgecolor="red", facecolor="none", lw=1.5, ls="--", zorder=3)
        ax.add_patch(ell)

        # Draw PCA major and minor axes aligned with the ellipse
        # The Ellipse patch is drawn in display coords, so axes must match.
        # Use the ellipse angle (which accounts for data-to-display transform)
        # to draw lines that visually align with the ellipse.
        # We draw in data coords using the ellipse semi-axes (w/2, h/2) and angle.
        angle_ell_rad = np.radians(angle_deg)
        for i in range(2):
            if i == 0:
                # Major axis: along ellipse angle, length = w/2
                semi = w / 2
                dx = semi * np.cos(angle_ell_rad)
                dy = semi * np.sin(angle_ell_rad)
                color, lw = "red", 2.0
            else:
                # Minor axis: perpendicular to ellipse angle, length = h/2
                semi = h / 2
                dx = semi * np.cos(angle_ell_rad + np.pi/2)
                dy = semi * np.sin(angle_ell_rad + np.pi/2)
                color, lw = "orange", 1.5
            ax.plot([center[0] - dx, center[0] + dx],
                    [center[1] - dy, center[1] + dy],
                    color=color, lw=lw, zorder=4)

        # Title
        title = f"{name} (n={d['n']}, PC1:{pc1_angle:.1f}°)"
        ax.set_title(title, fontsize=13, fontweight="bold")

        ax.set_xlabel("Δvz0 (km/h)", fontsize=11)
        ax.set_ylabel("ΔRelease Height (cm)", fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_aspect("equal")  # different units, not needed
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls=":")

        # Stats
        mean_rz_m = d["mean_rz_ft"] * FT2M
        stats_text = (f"Mean Rz: {d['mean_rz_ft']:.2f} ft ({mean_rz_m:.2f} m)\n"
                      f"Mean vz0: {d['mean_vz0_kmh']:.1f} km/h")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_release_vz_ff.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    main()
