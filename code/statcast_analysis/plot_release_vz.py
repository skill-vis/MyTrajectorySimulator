"""
日本人MLB投手9名の2025年全投球について、
リリースポイントの高さ(release_pos_z) vs 鉛直方向速度成分(vz0)の分布を
3x3グリッドでプロット。PCA傾きでソート。
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pybaseball import statcast
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

def main():
    # Check for cached data
    cache_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_release_vz.json")
    if os.path.exists(cache_path):
        print("Loading cached data...")
        with open(cache_path) as f:
            all_data = json.load(f)
    else:
        print("Fetching 2025 Statcast data...")
        df = statcast(start_dt="2025-03-20", end_dt="2025-10-01")
        print(f"  Total pitches: {len(df)}")

        all_data = {}
        for name, (pid, p_throws) in PITCHERS.items():
            df_p = df[df["pitcher"] == pid].copy()
            df_p = df_p.dropna(subset=["release_pos_z", "vz0"])
            n = len(df_p)
            print(f"  {name}: {n} pitches (all types)")

            all_data[name] = {
                "release_z": df_p["release_pos_z"].values.tolist(),  # feet
                "vz0": df_p["vz0"].values.tolist(),  # ft/s
                "n": n,
                "p_throws": p_throws,
            }

        with open(cache_path, "w") as f:
            json.dump(all_data, f)
        print(f"Saved cache: {cache_path}")

    # Compute PCA angle for each pitcher and sort
    name_angles = []
    for name, d in all_data.items():
        if d["n"] < 10:
            name_angles.append((name, 0))
            continue
        X = np.column_stack([d["vz0"], d["release_z"]])
        pca = PCA(n_components=1)
        pca.fit(X)
        angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
        name_angles.append((name, angle))

    name_angles.sort(key=lambda x: x[1], reverse=True)

    # Plot 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 19))
    fig.suptitle("Japanese MLB Pitchers 2025 — All Pitches\nRelease Height vs. Vertical Velocity (vz0)",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (name, pc1_angle) in enumerate(name_angles):
        ax = axes[idx // 3][idx % 3]
        d = all_data[name]

        if d["n"] < 10:
            ax.set_title(f"{name} (n={d['n']})", fontsize=13)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            continue

        vz0 = np.array(d["vz0"])         # ft/s
        rz = np.array(d["release_z"])     # feet

        # Mirror left-handed pitchers (not needed for height/vz, but keep consistency)
        p_throws = d["p_throws"]

        ax.scatter(vz0, rz, s=4, alpha=0.3, color="steelblue", zorder=2)

        # PCA ellipse
        X = np.column_stack([vz0, rz])
        pca = PCA(n_components=2)
        pca.fit(X)
        center = X.mean(axis=0)

        # Draw PCA axes
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
            length = 2.0 * np.sqrt(var)
            color = "red" if i == 0 else "orange"
            ax.annotate("", xy=(center[0] + comp[0]*length, center[1] + comp[1]*length),
                        xytext=center,
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))

        # 95% confidence ellipse
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)
        w = 2 * 2.0 * np.sqrt(pca.explained_variance_[0])  # 95%
        h = 2 * 2.0 * np.sqrt(pca.explained_variance_[1])
        ell = Ellipse(xy=center, width=w, height=h, angle=angle_deg,
                      edgecolor="red", facecolor="none", lw=1.5, ls="--", zorder=3)
        ax.add_patch(ell)

        # Title with info
        title = f"{name} (n={d['n']}, PC1:{pc1_angle:.1f}°)"
        ax.set_title(title, fontsize=13, fontweight="bold")

        ax.set_xlabel("vz0 (ft/s)", fontsize=11)
        ax.set_ylabel("Release Height (ft)", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Stats text
        stats_text = (f"Mean vz0: {vz0.mean():.1f} ft/s\n"
                      f"Mean Rz: {rz.mean():.2f} ft ({rz.mean()*FT2M:.2f} m)")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_release_vz.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    main()
