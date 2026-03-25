"""
Statcastから日本人MLB投手9名の2025年FFデータを取得し、
statcast_to_sim.pyのパイプラインでBSG分解 → wx,wy,wz を計算し、
plot_sorted.py用のJSONを生成する。
"""
import sys
import os
import json
import math
import numpy as np

# Add parent directory to path for imports
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(PARENT, "API")
sys.path.insert(0, PARENT)
sys.path.insert(0, API_DIR)

from statcast_to_sim import statcast_to_sim_params, statcast_spin_to_nathan, statcast_to_release
from pybaseball import statcast

FT2M = 0.3048
RPM_TO_RADS = math.pi / 30.0

# --- Japanese MLB pitchers (2025) ---
# name: (pitcher_mlbam_id, p_throws)
PITCHERS = {
    "Darvish":  (506433, "R"),
    "Ohtani":   (660271, "R"),  # Ohtani as pitcher
    "Yamamoto": (808967, "R"),
    "Sasaki":   (808963, "R"),
    "Matsui":   (673513, "L"),  # Yuki Matsui
    "Sugano":   (608372, "R"),
    "Senga":    (673540, "R"),
    "Kikuchi":  (579328, "L"),
    "Imanaga":  (684007, "L"),
}


def bsg_to_omega_xyz(backspin_rpm, sidespin_rpm, wg_rpm, theta_deg, phi_deg, v0_mps):
    """
    Convert BSG (backspin, sidespin, gyrospin) to (wx, wy, wz) in absolute coordinates.
    Same transformation as MyBallTrajectorySim_E.py lines 466-473.
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    v0x = v0_mps * math.cos(th) * math.sin(ph)
    v0y = -v0_mps * math.cos(th) * math.cos(ph)
    v0z = v0_mps * math.sin(th)
    v0 = v0_mps

    wx = (-backspin_rpm * math.cos(ph)
          - sidespin_rpm * math.sin(th) * math.sin(ph)
          + wg_rpm * v0x / v0) * RPM_TO_RADS
    wy = (backspin_rpm * math.sin(ph)
          - sidespin_rpm * math.sin(th) * math.cos(ph)
          + wg_rpm * v0y / v0) * RPM_TO_RADS
    wz = (sidespin_rpm * math.cos(th)
          + wg_rpm * v0z / v0) * RPM_TO_RADS
    return wx, wy, wz


def fetch_pitcher_data(name, pitcher_id, p_throws):
    """Fetch 2025 FF data for a single pitcher."""
    print(f"  Fetching {name} (ID={pitcher_id})...")

    # Fetch full 2025 season data
    df = statcast(start_dt="2025-03-20", end_dt="2025-10-01",
                  player_type="pitcher")

    # Filter for this pitcher and FF
    df_p = df[(df["pitcher"] == pitcher_id) & (df["pitch_type"] == "FF")]
    print(f"    Found {len(df_p)} FF pitches")
    return df_p


def main():
    # Fetch all data at once (more efficient than per-pitcher)
    print("Fetching 2025 Statcast data...")
    df = statcast(start_dt="2025-03-20", end_dt="2025-10-01")
    print(f"  Total pitches: {len(df)}")

    all_data = {}
    omega_data = {}

    for name, (pid, p_throws) in PITCHERS.items():
        print(f"\nProcessing {name} (ID={pid}, {p_throws})...")
        df_p = df[(df["pitcher"] == pid) & (df["pitch_type"] == "FF")].copy()
        df_p = df_p.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot",
                                    "release_spin_rate", "spin_axis",
                                    "vx0", "vy0", "vz0", "ax", "ay", "az",
                                    "release_pos_x", "release_pos_z", "release_extension"])
        n = len(df_p)
        print(f"  FF pitches (valid): {n}")

        if n == 0:
            # Store empty
            all_data[name] = {"x": [], "z": [], "n": 0,
                              "sz_top": 3.5, "sz_bot": 1.5}
            omega_data[name] = {"n": 0, "ox_mean": 0, "oy_mean": 0, "oz_mean": 0}
            continue

        # plate_x, plate_z in feet
        plate_x = df_p["plate_x"].values.tolist()
        plate_z = df_p["plate_z"].values.tolist()
        sz_top = df_p["sz_top"].mean()
        sz_bot = df_p["sz_bot"].mean()

        all_data[name] = {
            "x": plate_x,
            "z": plate_z,
            "n": n,
            "sz_top": float(sz_top),
            "sz_bot": float(sz_bot),
        }

        # Compute omega for each pitch
        ox_ratios = []
        oy_ratios = []
        oz_ratios = []

        for _, row in df_p.iterrows():
            pitch_dict = row.to_dict()
            try:
                params = statcast_to_sim_params(pitch_dict)
                wx, wy, wz = bsg_to_omega_xyz(
                    params["backspin_rpm"],
                    params["sidespin_rpm"],
                    params["wg_rpm"],
                    params["theta_deg"],
                    params["phi_deg"],
                    params["v0_mps"],
                )
                omega_norm = math.sqrt(wx**2 + wy**2 + wz**2)
                if omega_norm > 0:
                    ox_ratios.append(abs(wx) / omega_norm)
                    oy_ratios.append(abs(wy) / omega_norm)
                    oz_ratios.append(abs(wz) / omega_norm)
            except Exception as e:
                pass  # skip problematic rows

        omega_data[name] = {
            "n": len(ox_ratios),
            "ox_mean": float(np.mean(ox_ratios)) if ox_ratios else 0,
            "oy_mean": float(np.mean(oy_ratios)) if oy_ratios else 0,
            "oz_mean": float(np.mean(oz_ratios)) if oz_ratios else 0,
        }
        print(f"  omega computed for {len(ox_ratios)} pitches")
        print(f"    |wX|={omega_data[name]['ox_mean']:.3f}, "
              f"|wY|={omega_data[name]['oy_mean']:.3f}, "
              f"|wZ|={omega_data[name]['oz_mean']:.3f}")

    # Save JSONs
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sz_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_ff_2025_sz.json")
    om_path = os.path.join(SCRIPT_DIR, "japanese_pitchers_omega.json")
    with open(sz_path, "w") as f:
        json.dump(all_data, f)
    with open(om_path, "w") as f:
        json.dump(omega_data, f)

    print("\n=== Done ===")
    print("Saved /tmp/japanese_pitchers_ff_2025_sz.json")
    print("Saved /tmp/japanese_pitchers_omega.json")

    # Summary
    print("\n--- Summary ---")
    for name in PITCHERS:
        d = all_data[name]
        od = omega_data[name]
        print(f"  {name:10s}: n={d['n']:4d}, "
              f"|wX|={od['ox_mean']:.2f}, |wY|={od['oy_mean']:.2f}, |wZ|={od['oz_mean']:.2f}")


if __name__ == "__main__":
    main()
