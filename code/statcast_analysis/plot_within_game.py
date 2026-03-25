"""
試合内差分による散布図:
ΔRelease Height (cm) vs Δvz0 (km/h) — 各試合の平均からの差分
|r|でソート
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
    with open(os.path.join(SCRIPT_DIR, "japanese_pitchers_ff_gamepk.json")) as f:
        data = json.load(f)

    pitchers = []
    for name, d in data.items():
        if d["n"] < 10:
            continue
        rz = np.array(d["release_z"])
        vz = np.array(d["vz0"])
        gpk = np.array(d["game_pk"])

        # Within-game delta
        dz_list = []
        dvz_list = []
        games = np.unique(gpk)
        for g in games:
            mask = gpk == g
            if mask.sum() < 3:
                continue
            rz_g = rz[mask]
            vz_g = vz[mask]
            dz_list.extend(((rz_g - rz_g.mean()) * FT2CM).tolist())
            dvz_list.extend(((vz_g - vz_g.mean()) * FTS2KMH).tolist())

        dz = np.array(dz_list)
        dvz = np.array(dvz_list)
        r = np.corrcoef(dz, dvz)[0, 1] if len(dz) > 10 else 0

        # Also compute global for comparison
        dz_global = (rz - rz.mean()) * FT2CM
        dvz_global = (vz - vz.mean()) * FTS2KMH
        r_global = np.corrcoef(dz_global, dvz_global)[0, 1]

        pitchers.append({
            "name": name, "n": d["n"], "n_within": len(dz),
            "n_games": len(games),
            "dz": dz, "dvz": dvz,
            "r": r, "r_global": r_global,
            "mean_rz_m": rz.mean() * FT2CM / 100,
        })

    pitchers.sort(key=lambda x: abs(x["r"]), reverse=True)

    # Unified axis limits
    all_dvz = np.concatenate([p["dvz"] for p in pitchers])
    all_dz = np.concatenate([p["dz"] for p in pitchers])
    vz_margin = (all_dvz.max() - all_dvz.min()) * 0.1
    xlim = (all_dvz.min() - vz_margin, all_dvz.max() + vz_margin)
    dz_max = max(abs(all_dz.min()), abs(all_dz.max())) * 1.3
    ylim = (-dz_max, dz_max)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    fig.suptitle("Japanese MLB Pitchers 2025 — FF (4-Seam Fastball)\n"
                 "Within-Game: ΔRelease Height (cm) vs. Δvz0 (km/h) — sorted by |r|",
                 fontsize=16, fontweight="bold", y=0.99)

    for idx, p in enumerate(pitchers):
        ax = axes[idx // 3][idx % 3]
        x = p["dvz"]
        y = p["dz"]

        ax.scatter(x, y, s=4, alpha=0.3, color="steelblue", zorder=2)

        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([xlim[0], xlim[1]])
        ax.plot(x_line, slope * x_line + intercept, color="red", lw=2, zorder=3)

        # Title
        title = f"{p['name']} (n={p['n_within']}, {p['n_games']}games)"
        ax.set_title(title, fontsize=13, fontweight="bold")

        ax.set_xlabel("Δvz0 (km/h) [within-game]", fontsize=10)
        ax.set_ylabel("ΔRelease Height (cm) [within-game]", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")

        # r values
        r_color = "red" if abs(p["r"]) > 0.15 else "gray"
        ax.text(0.98, 0.98,
                f"r = {p['r']:+.3f}\n(global: {p['r_global']:+.3f})",
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right",
                color=r_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.text(0.02, 0.98, f"Mean Rz: {p['mean_rz_m']:.2f} m",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_within_game.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    main()
