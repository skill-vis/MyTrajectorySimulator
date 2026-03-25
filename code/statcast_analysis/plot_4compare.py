"""
4つのグラフ比較:
1. Δvz0 (km/h) vs ΔRelease Height (cm) — 標準化PCA
2. Δvz0 (cm/s) vs ΔRelease Height (cm) — 標準化PCA
3. Δ仰角 (deg) vs ΔRelease Height (cm) — 標準化PCA
4. Δ仰角 (rad) vs ΔRelease Height (cm) — 標準化PCA
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

FT2CM = 30.48
FTS2KMH = 1.09728
FTS2CMS = 30.48


def fetch_data():
    """Fetch FF data with vy0 included."""
    cache_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_ff_vy0.json")
    if os.path.exists(cache_path):
        print("Loading cached data...")
        with open(cache_path) as f:
            return json.load(f)

    print("Fetching 2025 Statcast data...")
    from pybaseball import statcast
    df = statcast(start_dt="2025-03-20", end_dt="2025-10-01")
    print(f"  Total pitches: {len(df)}")

    all_data = {}
    for name, (pid, p_throws) in PITCHERS.items():
        df_p = df[(df["pitcher"] == pid) & (df["pitch_type"] == "FF")].copy()
        df_p = df_p.dropna(subset=["release_pos_z", "vz0", "vy0"])
        n = len(df_p)
        print(f"  {name}: {n} FF pitches")
        all_data[name] = {
            "release_z": df_p["release_pos_z"].values.tolist(),
            "vz0": df_p["vz0"].values.tolist(),
            "vy0": df_p["vy0"].values.tolist(),
            "n": n,
            "p_throws": p_throws,
        }

    with open(cache_path, "w") as f:
        json.dump(all_data, f)
    print(f"Saved: {cache_path}")
    return all_data


def compute_panel_data(raw_data):
    """Compute the 4 x-axis variants for each pitcher."""
    pitchers = {}
    for name, d in raw_data.items():
        if d["n"] < 10:
            continue
        rz = np.array(d["release_z"])
        vz = np.array(d["vz0"])
        vy = np.array(d["vy0"])

        dz_cm = (rz - rz.mean()) * FT2CM

        # 1. Δvz0 (km/h)
        dvz_kmh = (vz - vz.mean()) * FTS2KMH
        # 2. Δvz0 (cm/s)
        dvz_cms = (vz - vz.mean()) * FTS2CMS
        # 3. Δ仰角 (deg) — arctan(vz0/|vy0|) in YZ plane
        elev = np.degrees(np.arctan2(vz, np.abs(vy)))
        delev_deg = elev - elev.mean()
        # 4. Δ仰角 (rad)
        elev_rad = np.arctan2(vz, np.abs(vy))
        delev_rad = elev_rad - elev_rad.mean()

        pitchers[name] = {
            "dz_cm": dz_cm,
            "x_variants": [dvz_kmh, dvz_cms, delev_deg, delev_rad],
            "n": d["n"],
            "mean_rz_m": rz.mean() * FT2CM / 100,
        }
    return pitchers


def plot_one_graph(fig_idx, pitchers, variant_idx, xlabel, title_suffix, out_name):
    """Plot 3x3 grid for one x-axis variant with standardized PCA."""

    # Compute standardized PCA angle for sorting
    name_angles = []
    for name, d in pitchers.items():
        x = d["x_variants"][variant_idx]
        y = d["dz_cm"]
        X_std = np.column_stack([x / x.std(), y / y.std()])
        pca = PCA(n_components=1).fit(X_std)
        angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
        name_angles.append((name, angle))
    name_angles.sort(key=lambda x: x[1], reverse=True)

    # Unified axis limits (raw, not standardized)
    all_x = np.concatenate([pitchers[n]["x_variants"][variant_idx] for n, _ in name_angles])
    all_y = np.concatenate([pitchers[n]["dz_cm"] for n, _ in name_angles])
    x_margin = (all_x.max() - all_x.min()) * 0.1
    xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
    dz_max = max(abs(all_y.min()), abs(all_y.max())) * 1.3
    ylim = (-dz_max, dz_max)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    fig.suptitle(f"Japanese MLB Pitchers 2025 — FF\n"
                 f"ΔRelease Height vs. {title_suffix} (Standardized PCA)",
                 fontsize=16, fontweight="bold", y=0.99)

    for idx, (name, pc1_angle) in enumerate(name_angles):
        ax = axes[idx // 3][idx % 3]
        d = pitchers[name]
        x_raw = d["x_variants"][variant_idx]
        y_raw = d["dz_cm"]

        ax.scatter(x_raw, y_raw, s=4, alpha=0.3, color="steelblue", zorder=2)

        # Standardized PCA (for ellipse and axes)
        x_std = x_raw / x_raw.std()
        y_std = y_raw / y_raw.std()
        X_std = np.column_stack([x_std, y_std])
        pca = PCA(n_components=2).fit(X_std)
        center_std = X_std.mean(axis=0)
        center_raw = np.array([x_raw.mean(), y_raw.mean()])

        # 95% ellipse in standardized space, then transform back
        chi2_95 = np.sqrt(5.991)
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)
        w_std = 2 * chi2_95 * np.sqrt(pca.explained_variance_[0])
        h_std = 2 * chi2_95 * np.sqrt(pca.explained_variance_[1])

        # Scale back to raw coordinates
        sx = x_raw.std()
        sy = y_raw.std()

        # Transform ellipse from standardized to raw coordinates
        # Ellipse in std space: angle=angle_deg, w=w_std, h=h_std
        # In raw space: need to account for sx, sy scaling
        # Sample points on std ellipse, then scale
        theta = np.linspace(0, 2 * np.pi, 200)
        cos_a = np.cos(np.radians(angle_deg))
        sin_a = np.sin(np.radians(angle_deg))
        # Points on ellipse in std space
        ex_std = (w_std / 2) * np.cos(theta) * cos_a - (h_std / 2) * np.sin(theta) * sin_a + center_std[0]
        ey_std = (w_std / 2) * np.cos(theta) * sin_a + (h_std / 2) * np.sin(theta) * cos_a + center_std[1]
        # Transform to raw
        ex_raw = ex_std * sx
        ey_raw = ey_std * sy
        ax.plot(ex_raw, ey_raw, color="red", lw=1.5, ls="--", zorder=3)

        # PCA axes in standardized space, transformed to raw
        for i in range(2):
            comp = pca.components_[i]
            var = pca.explained_variance_[i]
            length = chi2_95 * np.sqrt(var)
            # endpoints in std space
            p1_std = center_std - comp * length
            p2_std = center_std + comp * length
            # transform to raw
            p1_raw = np.array([p1_std[0] * sx, p1_std[1] * sy])
            p2_raw = np.array([p2_std[0] * sx, p2_std[1] * sy])
            color = "red" if i == 0 else "orange"
            lw = 2.0 if i == 0 else 1.5
            ax.plot([p1_raw[0], p2_raw[0]], [p1_raw[1], p2_raw[1]],
                    color=color, lw=lw, zorder=4)

        title = f"{name} (n={d['n']}, PC1:{pc1_angle:.1f}°)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("ΔRelease Height (cm)", fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(SCRIPT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    raw_data = fetch_data()
    pitchers = compute_panel_data(raw_data)

    configs = [
        (0, "Δvz0 (km/h)", "Δvz0 (km/h)", "compare_1_vz0_kmh.png"),
        (1, "Δvz0 (cm/s)", "Δvz0 (cm/s)", "compare_2_vz0_cms.png"),
        (2, "Δelev (deg)",  "Δelevation angle (deg)", "compare_3_elev_deg.png"),
        (3, "Δelev (rad)",  "Δelevation angle (rad)", "compare_4_elev_rad.png"),
    ]

    for variant_idx, xlabel, title_suffix, out_name in configs:
        print(f"Generating {out_name}...")
        plot_one_graph(0, pitchers, variant_idx, xlabel, title_suffix, out_name)

    # Print sort order comparison
    print("\n=== Sort order comparison (Standardized PCA) ===")
    labels = ["km/h", "cm/s", "deg", "rad"]
    orders = {}
    for vi, label in enumerate(labels):
        name_angles = []
        for name, d in pitchers.items():
            x = d["x_variants"][vi]
            y = d["dz_cm"]
            X_std = np.column_stack([x / x.std(), y / y.std()])
            pca = PCA(n_components=1).fit(X_std)
            angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
            name_angles.append((name, angle))
        name_angles.sort(key=lambda x: x[1], reverse=True)
        orders[label] = name_angles

    print(f"{'Rank':>4s}  {'km/h':>15s}  {'cm/s':>15s}  {'deg':>15s}  {'rad':>15s}")
    print("-" * 72)
    for i in range(9):
        row = f"{i+1:4d}"
        for label in labels:
            n, a = orders[label][i]
            row += f"  {n:>9s}({a:5.1f})"
        print(row)


if __name__ == "__main__":
    main()
