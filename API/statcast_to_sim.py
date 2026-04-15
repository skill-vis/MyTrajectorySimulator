"""
Convert Statcast pitch data to simulator initial conditions.

Based on Alan Nathan's method:
  - Statcast vx0/vy0/vz0 are at y = 50 ft, not at release.
  - Back-propagate to release point using constant acceleration (ax, ay, az).

Reference: https://baseball.physics.illinois.edu/
"""

import math
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FT_TO_M = 0.3048        # feet -> meters
MPH_TO_MS = 0.44704     # mph -> m/s
FPS_TO_MS = 0.3048      # ft/s -> m/s
INCHES_TO_M = 0.0254

BALL_CIRC_IN = 9.125                               # Nathan Excel: circumference (inches)
BALL_RADIUS_M = BALL_CIRC_IN * 0.0254 / (2 * math.pi)  # 0.03689 m
BALL_MASS_KG = 5.125 * 0.0283495                    # Nathan Excel: 5.125 oz = 0.14529 kg
BALL_CROSS_SECTION_M2 = math.pi * BALL_RADIUS_M ** 2
AIR_DENSITY_KG_M3 = 1.225
RELEASE_TO_PLATE_M = 55.0 * FT_TO_M  # ~16.764 m


def statcast_to_release(pitch: Dict) -> Dict:
    """
    Convert Statcast data to release-point values (Nathan's method).

    Parameters
    ----------
    pitch : dict
        Statcast row with keys:
            release_pos_x, release_pos_z, release_extension,
            vx0, vy0, vz0 (ft/s at y=50ft),
            ax, ay, az (ft/s^2),
            release_spin_rate (rpm), spin_axis (deg)

    Returns
    -------
    dict with:
        x0, y0, z0 (m),
        v0_mps (m/s),
        theta_deg (vertical release angle, deg),
        phi_deg (horizontal release direction, deg),
        vxR, vyR, vzR (ft/s at release),
        release_speed_mph
    """
    # --- Position at release (Nathan convention) ---
    x0_ft = float(pitch["release_pos_x"])       # ft
    y0_ft = 60.5 - float(pitch["release_extension"])  # ft
    z0_ft = float(pitch["release_pos_z"])        # ft

    # --- Velocity at y = 50 ft (Statcast) ---
    vx0 = float(pitch["vx0"])   # ft/s
    vy0 = float(pitch["vy0"])   # ft/s
    vz0 = float(pitch["vz0"])   # ft/s

    # --- Acceleration (constant, ft/s^2) ---
    ax = float(pitch["ax"])
    ay = float(pitch["ay"])
    az = float(pitch["az"])

    # --- Back-propagate to release point (Nathan's equations) ---
    # vyR = -sqrt(vy0^2 + 2*ay*(y0 - 50))
    #   (negative because ball moves toward home plate, i.e. -y direction)
    vyR = -math.sqrt(vy0**2 + 2 * ay * (y0_ft - 50.0))

    # tR = (vyR - vy0) / ay
    tR = (vyR - vy0) / ay

    # vxR, vzR at release
    vxR = vx0 + ax * tR
    vzR = vz0 + az * tR

    # --- Release speed, angle, direction ---
    release_speed_fps = math.sqrt(vxR**2 + vyR**2 + vzR**2)
    release_speed_mph = release_speed_fps * 0.6818
    release_speed_mps = release_speed_fps * FPS_TO_MS

    # theta = atan(vzR / sqrt(vxR^2 + vyR^2))  (positive = upward)
    theta_deg = math.degrees(math.atan2(vzR, math.sqrt(vxR**2 + vyR**2)))

    # Simulator: v0x = v0*cos(th)*sin(phi), v0y = -v0*cos(th)*cos(phi)
    # => sin(phi) = vxR / v0h, cos(phi) = -vyR / v0h  (vyR < 0)
    phi_deg = math.degrees(math.atan2(vxR, -vyR))

    # --- Position in meters ---
    x0_m = x0_ft * FT_TO_M
    y0_m = y0_ft * FT_TO_M
    z0_m = z0_ft * FT_TO_M

    return {
        # Simulator inputs (MKS)
        "x0": x0_m,
        "y0": y0_m,
        "z0": z0_m,
        "v0_mps": release_speed_mps,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        # Reference values
        "release_speed_mph": release_speed_mph,
        "vxR_fps": vxR,
        "vyR_fps": vyR,
        "vzR_fps": vzR,
        "tR": tR,
        # Raw Statcast
        "release_spin_rate": pitch.get("release_spin_rate"),
        "spin_axis": pitch.get("spin_axis"),
        "pitch_type": pitch.get("pitch_type"),
    }


# ---------------------------------------------------------------------------
# Spin decomposition: spin_axis + spin_rate -> backspin / sidespin / gyrospin
#
#   Simulator mapping (phi≈0, theta≈0):
#       wx ≈ -backspin,  wz ≈ sidespin
#   Magnus force via ω × v (v ≈ (0, -v, 0)):
#       F_x ∝  sidespin   (horizontal movement → pfx_x)
#       F_z ∝  backspin   (vertical lift       → pfx_z)
#
#   Improved method: use pfx direction to derive effective spin axis,
#   and pfx magnitude to estimate spin efficiency.
# ---------------------------------------------------------------------------

RPM_TO_RADS = 2.0 * math.pi / 60.0
RADS_TO_RPM = 60.0 / (2.0 * math.pi)


def _solve_transverse_spin(a_magnus_mps2: float, v_mps: float,
                           lift_model: str = "nathan_exp") -> float:
    """
    Solve for transverse spin rate (rad/s) from Magnus acceleration.

    Parameters
    ----------
    a_magnus_mps2 : float
        Magnus acceleration magnitude (m/s^2)
    v_mps : float
        Ball speed (m/s)
    lift_model : str
        "nathan_exp" — C_L = A[1 - exp(-BS)], invert: S = -ln(1 - C_L/A) / B
        "rational"  — C_L = S/(0.4 + 2.32S),  invert: S = 0.4K / (1 - 2.32K)
    """
    if v_mps <= 0 or a_magnus_mps2 <= 0:
        return 0.0
    K = (a_magnus_mps2 * 2.0 * BALL_MASS_KG
         / (AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 * v_mps ** 2))

    if lift_model == "rational":
        denom = 1.0 - 2.32 * K
        if denom <= 0:
            return 6000.0 * RPM_TO_RADS
        S = 0.4 * K / denom
    else:  # nathan_exp (default)
        CL_A = 0.336
        CL_B = 6.041
        if K >= CL_A:
            return 6000.0 * RPM_TO_RADS
        S = -math.log(1.0 - K / CL_A) / CL_B

    return S * v_mps / BALL_RADIUS_M


def _estimate_spin_from_accel(pitch: Dict, spin_rate_rpm: float,
                              lift_model: str = "nathan_exp"):
    """
    Nathan (2020) method: estimate transverse spin from Statcast acceleration.

    1. Remove gravity:  a* = a - g
    2. Compute mean velocity <v> from 9P constant-acceleration fit
    3. Remove drag (projection along <v>): a_D = (a* . <v_hat>) <v_hat>
    4. Magnus acceleration: a_M = a* - a_D   (Eq. 6)
    5. Transverse spin direction: omega_T_hat = <v_hat> x a_M_hat  (Eq. 10)
    6. C_L from |a_M|, invert for omega_T  (Eqs. 7-9)

    Returns (omega_T_vec_rads, omega_T_mag_rads, efficiency, a_M_vec_mps2)
            or (None, None, None, None) if data unavailable.
    """
    try:
        # Statcast 9P fit: velocity at y=50ft (ft/s) and constant acceleration (ft/s^2)
        vx0 = float(pitch["vx0"])
        vy0 = float(pitch["vy0"])
        vz0 = float(pitch["vz0"])
        ax = float(pitch["ax"])
        ay = float(pitch["ay"])
        az = float(pitch["az"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, None

    # --- Step 2: Mean velocity (constant acceleration → midpoint velocity) ---
    # Flight time from y=50ft to home plate (y=17/12 ft ≈ 1.417 ft)
    # y(t) = 50 + vy0*t + 0.5*ay*t^2 = y_plate
    # Solve quadratic: 0.5*ay*t^2 + vy0*t + (50 - y_plate) = 0
    y_plate_ft = 17.0 / 12.0  # front of home plate
    A_coeff = 0.5 * ay
    B_coeff = vy0
    C_coeff = 50.0 - y_plate_ft
    disc = B_coeff**2 - 4 * A_coeff * C_coeff
    if disc < 0:
        return None, None, None, None
    # Take smallest positive root (first time ball reaches home plate)
    sqrt_disc = math.sqrt(disc)
    t1 = (-B_coeff - sqrt_disc) / (2 * A_coeff)
    t2 = (-B_coeff + sqrt_disc) / (2 * A_coeff)
    positive_roots = [t for t in (t1, t2) if t > 0]
    if not positive_roots:
        return None, None, None, None
    t_flight = min(positive_roots)

    # Mean velocity = v0 + a*t/2 (midpoint of constant-acceleration trajectory)
    vx_mean = vx0 + ax * t_flight / 2.0
    vy_mean = vy0 + ay * t_flight / 2.0
    vz_mean = vz0 + az * t_flight / 2.0
    # Convert to m/s
    vx_m = vx_mean * FPS_TO_MS
    vy_m = vy_mean * FPS_TO_MS
    vz_m = vz_mean * FPS_TO_MS
    v_mean = math.sqrt(vx_m**2 + vy_m**2 + vz_m**2)
    if v_mean <= 0:
        return None, None, None, None
    v_hat = (vx_m / v_mean, vy_m / v_mean, vz_m / v_mean)

    # --- Step 1: Remove gravity from acceleration ---
    # a* = a - g  (convert ax,ay,az from ft/s^2 to m/s^2)
    g_mps2 = 9.80
    ax_m = ax * FPS_TO_MS  # ft/s^2 → m/s^2 (multiply by 0.3048)
    ay_m = ay * FPS_TO_MS
    az_m = az * FPS_TO_MS
    astar = (ax_m, ay_m, az_m - (-g_mps2))  # az includes -g, so a* = a + g_z
    # Note: Statcast az already includes gravity (az ≈ -32 ft/s^2 for no-spin).
    # a = a_D + a_M + g, so a* = a - g = a_D + a_M
    # In PITCHf/x convention: g = (0, 0, -g), so a*_z = az_m + g
    astar = (ax_m, ay_m, az_m + g_mps2)

    # --- Step 3: Remove drag (projection along mean velocity) ---
    # a_D = (a* . v_hat) * v_hat
    astar_dot_vhat = astar[0]*v_hat[0] + astar[1]*v_hat[1] + astar[2]*v_hat[2]
    a_drag = (astar_dot_vhat * v_hat[0],
              astar_dot_vhat * v_hat[1],
              astar_dot_vhat * v_hat[2])

    # --- Step 4: Magnus acceleration = a* - a_D  (Nathan Eq. 6) ---
    a_M = (astar[0] - a_drag[0],
           astar[1] - a_drag[1],
           astar[2] - a_drag[2])
    a_M_mag = math.sqrt(a_M[0]**2 + a_M[1]**2 + a_M[2]**2)

    if a_M_mag < 1e-6:
        return None, None, None, None

    # --- Step 5: Transverse spin direction  (Nathan Eq. 10) ---
    # omega_T_hat = <v_hat> x a_M_hat
    a_M_hat = (a_M[0] / a_M_mag, a_M[1] / a_M_mag, a_M[2] / a_M_mag)
    wT_hat = (v_hat[1]*a_M_hat[2] - v_hat[2]*a_M_hat[1],
              v_hat[2]*a_M_hat[0] - v_hat[0]*a_M_hat[2],
              v_hat[0]*a_M_hat[1] - v_hat[1]*a_M_hat[0])

    # --- Step 6: Solve for omega_T magnitude via C_L inversion ---
    omega_T_rads = _solve_transverse_spin(a_M_mag, v_mean, lift_model)
    omega_T_rpm = omega_T_rads * RADS_TO_RPM

    # omega_T vector (rad/s)
    omega_T_vec = (omega_T_rads * wT_hat[0],
                   omega_T_rads * wT_hat[1],
                   omega_T_rads * wT_hat[2])

    efficiency = min(omega_T_rpm / spin_rate_rpm, 1.0) if spin_rate_rpm > 0 else 1.0

    return omega_T_vec, omega_T_rads, efficiency, a_M


def _estimate_spin_from_pfx(pitch: Dict, spin_rate_rpm: float,
                            lift_model: str = "nathan_exp"):
    """
    Estimate effective spin axis and spin efficiency from pfx data.

    - Direction: θ_eff = atan2(pfx_x, -pfx_z)
      (pfx_x ∝ sin(θ), pfx_z ∝ -cos(θ))
    - Efficiency: from pfx magnitude → Magnus accel → solve C_L for omega_T

    Returns (theta_eff_rad, spin_efficiency) or (None, None) if data unavailable.
    """
    pfx_x = pitch.get("pfx_x")   # feet (from pybaseball)
    pfx_z = pitch.get("pfx_z")   # feet

    if pfx_x is None or pfx_z is None:
        return None, None

    try:
        pfx_x_f = float(pfx_x)
        pfx_z_f = float(pfx_z)
    except (TypeError, ValueError):
        return None, None

    if pfx_x_f == 0 and pfx_z_f == 0:
        return None, None

    # --- Effective spin axis from pfx direction ---
    theta_eff_rad = math.atan2(pfx_x_f, -pfx_z_f)

    # --- Spin efficiency from pfx magnitude ---
    pfx_mag_m = math.sqrt(pfx_x_f ** 2 + pfx_z_f ** 2) * FT_TO_M

    # Flight time from y=50ft to home plate
    try:
        vy0 = float(pitch["vy0"])   # ft/s (negative)
        # Simple approximation: t ≈ 50 / |vy0|
        t_flight = 50.0 / abs(vy0) if abs(vy0) > 0 else 0.4

        # Average speed (ball decelerates ~5-8% from y=50ft to plate)
        v_50ft = math.sqrt(float(pitch["vx0"]) ** 2
                           + vy0 ** 2
                           + float(pitch["vz0"]) ** 2)
        v_avg_mps = v_50ft * 0.96 * FPS_TO_MS
    except (KeyError, TypeError, ValueError):
        t_flight = 0.4
        v_avg_mps = 38.0  # ~85 mph fallback

    # Magnus acceleration from pfx: pfx = 0.5 * a_M * t^2
    a_magnus_mps2 = 2.0 * pfx_mag_m / (t_flight ** 2) if t_flight > 0 else 0.0

    # Solve for transverse spin
    omega_T_rads = _solve_transverse_spin(a_magnus_mps2, v_avg_mps, lift_model)
    omega_T_rpm = omega_T_rads * RADS_TO_RPM

    efficiency = min(omega_T_rpm / spin_rate_rpm, 1.0) if spin_rate_rpm > 0 else 1.0

    return theta_eff_rad, efficiency


def statcast_spin_to_bsg(pitch: Dict, theta_deg: float, phi_deg: float,
                         use_pfx: bool = True, lift_model: str = "nathan_exp",
                         accel_method: bool = False) -> Dict:
    """
    Convert Statcast spin data to orthonormal BSG (backspin / sidespin / gyrospin).

    Uses orthonormal basis:
        eg = velocity direction
        eb = eg × eZ / |eg × eZ|  (backspin, horizontal, B>0 = hop)
        es = eb × eg               (sidespin)

    Computes ω in XYZ first, then projects onto eb, es, eg.

    Parameters
    ----------
    pitch : dict
        Must contain: release_spin_rate (rpm), spin_axis (deg).
        For pfx method: pfx_x, pfx_z (feet), vy0, vx0, vz0 (ft/s).
        For accel method: ax, ay, az (ft/s^2) additionally.
    theta_deg, phi_deg : float
        Release angles (deg) for BSG basis computation.
    accel_method : bool
        If True, use Nathan (2020) acceleration-based method (3D ω_T).
        Falls back to pfx method if accel data unavailable.

    Returns
    -------
    dict with:
        backspin_rpm, sidespin_rpm, wg_rpm (gyrospin),
        spin_efficiency (0-1), active_spin_rpm,
        theta_eff_deg (effective spin axis used)
    """
    spin_rate = float(pitch["release_spin_rate"])
    spin_axis = float(pitch["spin_axis"])

    # Try Nathan accel method first if requested
    if accel_method:
        result = _nathan_accel_to_bsg(pitch, spin_rate, theta_deg, phi_deg, lift_model)
        if result is not None:
            return result
        # Fall through to pfx method

    theta_eff_rad = None
    efficiency = None

    if use_pfx:
        theta_eff_rad, efficiency = _estimate_spin_from_pfx(pitch, spin_rate, lift_model)

    # Fallback to Statcast spin_axis / 100% efficiency
    if theta_eff_rad is None:
        theta_eff_rad = math.radians(spin_axis)
    if efficiency is None:
        efficiency = 1.0

    efficiency = max(0.0, min(efficiency, 1.0))

    active_spin = spin_rate * efficiency
    gyro_rpm = spin_rate * math.sqrt(max(1.0 - efficiency ** 2, 0.0))

    # Compute ω in XYZ (rad/s)
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)

    # Transverse spin (in XZ plane)
    wx = active_spin * math.cos(theta_eff_rad) * RPM_TO_RADS
    wz = active_spin * math.sin(theta_eff_rad) * RPM_TO_RADS

    # Gyro spin along velocity direction
    ux = cth * sph
    uy = -cth * cph
    uz = sth
    wx += gyro_rpm * ux * RPM_TO_RADS
    wy = gyro_rpm * uy * RPM_TO_RADS
    wz += gyro_rpm * uz * RPM_TO_RADS

    # Project onto orthonormal BSG basis
    eb = (-cph, -sph, 0)
    es = (-sth * sph, sth * cph, cth)
    eg = (ux, uy, uz)

    rpm_per_rad_s = RADS_TO_RPM
    backspin = rpm_per_rad_s * (wx * eb[0] + wy * eb[1] + wz * eb[2])
    sidespin = rpm_per_rad_s * (wx * es[0] + wy * es[1] + wz * es[2])
    wg = rpm_per_rad_s * (wx * eg[0] + wy * eg[1] + wz * eg[2])

    return {
        "backspin_rpm": backspin,
        "sidespin_rpm": sidespin,
        "wg_rpm": wg,
        "spin_efficiency": efficiency,
        "active_spin_rpm": active_spin,
        "theta_eff_deg": math.degrees(theta_eff_rad) % 360,
    }


def _nathan_accel_to_bsg(pitch: Dict, spin_rate_rpm: float,
                         theta_deg: float, phi_deg: float,
                         lift_model: str = "nathan_exp") -> Dict:
    """
    Nathan (2020) full method: acceleration → Magnus → ω_T (3D) → BSG.

    Unlike the pfx method which estimates ω_T direction in 2D (xz plane only),
    this derives a full 3D ω_T vector via the cross product <v_hat> × a_M_hat.
    Gyro spin is computed as the remainder: ω_G = ω − ω_T, along ±<v_hat>.

    Returns dict with backspin_rpm, sidespin_rpm, wg_rpm, spin_efficiency, etc.
    Falls back to pfx method if acceleration data is unavailable.
    """
    result = _estimate_spin_from_accel(pitch, spin_rate_rpm, lift_model)
    if result[0] is None:
        # Fallback to pfx method
        return None

    omega_T_vec, omega_T_rads, efficiency, a_M = result
    efficiency = max(0.0, min(efficiency, 1.0))

    spin_rate_rads = spin_rate_rpm * RPM_TO_RADS
    omega_T_mag = omega_T_rads

    # Gyro spin magnitude: ω_G = sqrt(ω² - ω_T²)
    omega_G_rads = math.sqrt(max(spin_rate_rads**2 - omega_T_mag**2, 0.0))

    # Gyro direction: ±<v_hat>  (Nathan: parallel for RHP, anti-parallel for LHP)
    # We use the pitch velocity direction as the gyro axis
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    cth, sth = math.cos(th), math.sin(th)
    cph, sph = math.cos(ph), math.sin(ph)

    # Velocity unit vector
    ux = cth * sph
    uy = -cth * cph
    uz = sth

    # Determine gyro sign from pitch handedness or use positive convention
    p_throws = str(pitch.get("p_throws", "R")).upper()
    gyro_sign = 1.0 if p_throws == "R" else -1.0

    # Full ω = ω_T + ω_G
    wx = omega_T_vec[0] + gyro_sign * omega_G_rads * ux
    wy = omega_T_vec[1] + gyro_sign * omega_G_rads * uy
    wz = omega_T_vec[2] + gyro_sign * omega_G_rads * uz

    # Project onto orthonormal BSG basis
    eb = (-cph, -sph, 0)
    es = (-sth * sph, sth * cph, cth)
    eg = (ux, uy, uz)

    rpm_per_rad_s = RADS_TO_RPM
    backspin = rpm_per_rad_s * (wx * eb[0] + wy * eb[1] + wz * eb[2])
    sidespin = rpm_per_rad_s * (wx * es[0] + wy * es[1] + wz * es[2])
    wg = rpm_per_rad_s * (wx * eg[0] + wy * eg[1] + wz * eg[2])

    # Compute theta_eff from ω_T direction (projected to xz plane for display)
    theta_eff_rad = math.atan2(omega_T_vec[2], omega_T_vec[0]) if omega_T_mag > 0 else 0.0

    return {
        "backspin_rpm": backspin,
        "sidespin_rpm": sidespin,
        "wg_rpm": wg,
        "spin_efficiency": efficiency,
        "active_spin_rpm": omega_T_mag * RADS_TO_RPM,
        "theta_eff_deg": math.degrees(theta_eff_rad) % 360,
        "a_magnus_mps2": a_M,  # for diagnostics
    }


def statcast_spin_to_omega_direct(pitch: Dict, theta_deg: float, phi_deg: float,
                                   use_pfx: bool = True,
                                   lift_model: str = "nathan_exp",
                                   accel_method: bool = False) -> Dict:
    """
    Convert Statcast spin data directly to wx, wy, wz (rad/s),
    bypassing the BSG decomposition.

    Transverse spin (force-generating) is placed in the XZ plane
    based on spin_axis, gyro spin is placed along the velocity direction.

    If accel_method=True, uses Nathan (2020) acceleration-based 3D ω_T.
    """
    spin_rate = float(pitch["release_spin_rate"])
    spin_axis = float(pitch["spin_axis"])

    # Try Nathan accel method if requested
    if accel_method:
        result = _estimate_spin_from_accel(pitch, spin_rate, lift_model)
        if result[0] is not None:
            omega_T_vec, omega_T_rads, efficiency, a_M = result
            efficiency = max(0.0, min(efficiency, 1.0))
            omega_G_rads = math.sqrt(max((spin_rate * RPM_TO_RADS)**2 - omega_T_rads**2, 0.0))
            th = math.radians(theta_deg)
            ph = math.radians(phi_deg)
            ux = math.cos(th) * math.sin(ph)
            uy = -math.cos(th) * math.cos(ph)
            uz = math.sin(th)
            p_throws = str(pitch.get("p_throws", "R")).upper()
            gyro_sign = 1.0 if p_throws == "R" else -1.0
            return {
                "wx_rad_s": omega_T_vec[0] + gyro_sign * omega_G_rads * ux,
                "wy_rad_s": omega_T_vec[1] + gyro_sign * omega_G_rads * uy,
                "wz_rad_s": omega_T_vec[2] + gyro_sign * omega_G_rads * uz,
                "spin_efficiency": efficiency,
                "active_spin_rpm": omega_T_rads * RADS_TO_RPM,
                "theta_eff_deg": math.degrees(math.atan2(omega_T_vec[2], omega_T_vec[0])) % 360,
            }

    theta_eff_rad = None
    efficiency = None

    if use_pfx:
        theta_eff_rad, efficiency = _estimate_spin_from_pfx(pitch, spin_rate, lift_model)

    if theta_eff_rad is None:
        theta_eff_rad = math.radians(spin_axis)
    if efficiency is None:
        efficiency = 1.0

    efficiency = max(0.0, min(efficiency, 1.0))

    active_spin = spin_rate * efficiency
    gyro_rpm = spin_rate * math.sqrt(max(1.0 - efficiency ** 2, 0.0))

    # Transverse spin → wx, wz directly from spin_axis (2D, catcher's view)
    wx_trans = active_spin * math.cos(theta_eff_rad) * RPM_TO_RADS
    wz_trans = active_spin * math.sin(theta_eff_rad) * RPM_TO_RADS

    # Gyro spin along velocity direction
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    ux = math.cos(th) * math.sin(ph)
    uy = -math.cos(th) * math.cos(ph)
    uz = math.sin(th)

    wx_gyro = gyro_rpm * ux * RPM_TO_RADS
    wy_gyro = gyro_rpm * uy * RPM_TO_RADS
    wz_gyro = gyro_rpm * uz * RPM_TO_RADS

    return {
        "wx_rad_s": wx_trans + wx_gyro,
        "wy_rad_s": wy_gyro,
        "wz_rad_s": wz_trans + wz_gyro,
        "spin_efficiency": efficiency,
        "active_spin_rpm": active_spin,
        "theta_eff_deg": math.degrees(theta_eff_rad) % 360,
    }


def statcast_to_sim_params(pitch: Dict, spin_method: str = "bsg",
                           lift_model: str = "nathan_exp",
                           accel_method: bool = False) -> Dict:
    """
    Full pipeline: Statcast row -> simulator PitchParameters dict.

    Combines position/velocity back-propagation and spin decomposition.

    Parameters
    ----------
    pitch : dict
        Statcast row (from statcast_fetcher.select_pitch)
    spin_method : str
        'bsg' (default, Nathan BSG) or 'direct' (spin_axis → wx,wy,wz)
    lift_model : str
        'nathan_exp' or 'rational' — used for C_L inversion in spin efficiency estimation
    accel_method : bool
        If True, use Nathan (2020) acceleration-based Magnus separation for ω_T.
        Falls back to pfx method if accel data unavailable.

    Returns
    -------
    dict compatible with PitchParameters
    """
    release = statcast_to_release(pitch)

    result = {
        "x0": release["x0"],
        "y0": release["y0"],
        "z0": release["z0"],
        "v0_mps": release["v0_mps"],
        "theta_deg": release["theta_deg"],
        "phi_deg": release["phi_deg"],
        "release_speed_mph": release["release_speed_mph"],
        "pitch_type": pitch.get("pitch_type"),
        "spin_method": spin_method,
    }

    if spin_method == "direct":
        omega = statcast_spin_to_omega_direct(
            pitch, release["theta_deg"], release["phi_deg"],
            lift_model=lift_model, accel_method=accel_method)
        result.update({
            "wx_rad_s": omega["wx_rad_s"],
            "wy_rad_s": omega["wy_rad_s"],
            "wz_rad_s": omega["wz_rad_s"],
            "spin_efficiency": omega["spin_efficiency"],
            "theta_eff_deg": omega.get("theta_eff_deg"),
            # Compute equivalent BSG for display
            "backspin_rpm": 0.0,
            "sidespin_rpm": 0.0,
            "wg_rpm": 0.0,
        })
    else:
        spin = statcast_spin_to_bsg(pitch, release["theta_deg"], release["phi_deg"],
                                    lift_model=lift_model, accel_method=accel_method)
        result.update({
            "backspin_rpm": spin["backspin_rpm"],
            "sidespin_rpm": spin["sidespin_rpm"],
            "wg_rpm": spin["wg_rpm"],
            "spin_efficiency": spin["spin_efficiency"],
            "theta_eff_deg": spin.get("theta_eff_deg"),
        })

    return result


def vectorized_bsg_summary(df, lift_model: str = "nathan_exp") -> "pd.DataFrame":
    """
    Compute BSG decomposition for all rows in a Statcast DataFrame (vectorized).

    Returns a DataFrame with columns:
        backspin_rpm, sidespin_rpm, wg_rpm, spin_efficiency, theta_deg, phi_deg
    Rows with insufficient data are filled with NaN.
    """
    import pandas as pd

    required = ["release_pos_x", "release_pos_z", "release_extension",
                "vx0", "vy0", "vz0", "ax", "ay", "az",
                "release_spin_rate", "spin_axis", "pfx_x", "pfx_z"]
    mask = df[required].notna().all(axis=1)
    out = pd.DataFrame(index=df.index, columns=[
        "backspin_rpm", "sidespin_rpm", "wg_rpm", "spin_efficiency",
        "theta_deg", "phi_deg",
    ], dtype=float)

    if mask.sum() == 0:
        return out

    d = df.loc[mask]

    # --- Release back-propagation (vectorized) ---
    y0_ft = 60.5 - d["release_extension"].values.astype(float)
    vx0 = d["vx0"].values.astype(float)
    vy0 = d["vy0"].values.astype(float)
    vz0 = d["vz0"].values.astype(float)
    ax = d["ax"].values.astype(float)
    ay = d["ay"].values.astype(float)
    az = d["az"].values.astype(float)

    vyR = -np.sqrt(vy0**2 + 2 * ay * (y0_ft - 50.0))
    tR = (vyR - vy0) / ay
    vxR = vx0 + ax * tR
    vzR = vz0 + az * tR

    v_h = np.sqrt(vxR**2 + vyR**2)
    theta = np.arctan2(vzR, v_h)
    phi = np.arctan2(vxR, -vyR)

    # --- Spin efficiency from pfx (vectorized) ---
    spin_rate = d["release_spin_rate"].values.astype(float)
    pfx_x = d["pfx_x"].values.astype(float)
    pfx_z = d["pfx_z"].values.astype(float)

    theta_eff = np.arctan2(pfx_x, -pfx_z)
    pfx_mag_m = np.sqrt(pfx_x**2 + pfx_z**2) * FT_TO_M

    t_flight = np.where(np.abs(vy0) > 0, 50.0 / np.abs(vy0), 0.4)
    v_50ft = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    v_avg_mps = v_50ft * 0.96 * FPS_TO_MS

    a_magnus = np.where(t_flight > 0, 2.0 * pfx_mag_m / t_flight**2, 0.0)

    # Solve for transverse spin: invert C_L(S) model
    K = a_magnus * 2.0 * BALL_MASS_KG / (AIR_DENSITY_KG_M3 * BALL_CROSS_SECTION_M2 * v_avg_mps**2)
    S_cap = 6000.0 * RPM_TO_RADS * BALL_RADIUS_M / v_avg_mps
    if lift_model == "rational":
        denom = 1.0 - 2.32 * K
        S_param = np.where(denom > 0, 0.4 * K / denom, S_cap)
    else:  # nathan_exp
        CL_A, CL_B = 0.336, 6.041
        ratio = np.clip(K / CL_A, 0, 0.9999)
        S_param = np.where(K < CL_A, -np.log(1.0 - ratio) / CL_B, S_cap)
    omega_T = S_param * v_avg_mps / BALL_RADIUS_M
    omega_T_rpm = omega_T * RADS_TO_RPM

    efficiency = np.clip(np.where(spin_rate > 0, omega_T_rpm / spin_rate, 1.0), 0.0, 1.0)

    # --- BSG decomposition (vectorized) ---
    active = spin_rate * efficiency
    gyro = spin_rate * np.sqrt(np.maximum(1.0 - efficiency**2, 0.0))

    # ω in XYZ
    wx = active * np.cos(theta_eff) * RPM_TO_RADS
    wz = active * np.sin(theta_eff) * RPM_TO_RADS

    cth, sth = np.cos(theta), np.sin(theta)
    cph, sph = np.cos(phi), np.sin(phi)
    ux = cth * sph
    uy = -cth * cph
    uz = sth

    wx += gyro * ux * RPM_TO_RADS
    wy = gyro * uy * RPM_TO_RADS
    wz += gyro * uz * RPM_TO_RADS

    # Project onto orthonormal BSG: eb=(-cph,-sph,0), es=(-sth*sph,sth*cph,cth)
    eb_x, eb_y = -cph, -sph
    es_x, es_y, es_z = -sth * sph, sth * cph, cth
    eg_x, eg_y, eg_z = ux, uy, uz

    backspin = (wx * eb_x + wy * eb_y) * RADS_TO_RPM
    sidespin = (wx * es_x + wy * es_y + wz * es_z) * RADS_TO_RPM
    wg = (wx * eg_x + wy * eg_y + wz * eg_z) * RADS_TO_RPM

    out.loc[mask, "backspin_rpm"] = backspin
    out.loc[mask, "sidespin_rpm"] = sidespin
    out.loc[mask, "wg_rpm"] = wg
    out.loc[mask, "spin_efficiency"] = efficiency
    out.loc[mask, "theta_deg"] = np.degrees(theta)
    out.loc[mask, "phi_deg"] = np.degrees(phi)

    return out


def print_conversion(result: Dict):
    """Pretty-print the conversion result."""
    print("\n=== Release Point (Simulator Initial Conditions) ===")
    print(f"  Position:  x0={result['x0']:.4f} m,  y0={result['y0']:.4f} m,  z0={result['z0']:.4f} m")
    print(f"  Speed:     {result['v0_mps']:.2f} m/s  ({result['release_speed_mph']:.1f} mph)")
    print(f"  Angle:     theta={result['theta_deg']:.2f} deg (vertical),  phi={result['phi_deg']:.2f} deg (horizontal)")
    print(f"  Back-prop: tR={result['tR']:.5f} s")
    print(f"  vR (ft/s): ({result['vxR_fps']:.2f}, {result['vyR_fps']:.2f}, {result['vzR_fps']:.2f})")
    print(f"  Spin:      {result.get('release_spin_rate')} rpm,  axis={result.get('spin_axis')} deg")
    print(f"  Type:      {result.get('pitch_type')}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def print_sim_params(params: Dict):
    """Pretty-print full simulator parameters."""
    print("\n=== Simulator PitchParameters ===")
    print(f"  x0         = {params['x0']:.4f} m")
    print(f"  y0         = {params['y0']:.4f} m")
    print(f"  z0         = {params['z0']:.4f} m")
    print(f"  v0_mps     = {params['v0_mps']:.2f} m/s  ({params['release_speed_mph']:.1f} mph)")
    print(f"  theta_deg  = {params['theta_deg']:.2f} deg")
    print(f"  phi_deg    = {params['phi_deg']:.2f} deg")
    print(f"  backspin   = {params['backspin_rpm']:.1f} rpm")
    print(f"  sidespin   = {params['sidespin_rpm']:.1f} rpm")
    print(f"  wg (gyro)  = {params['wg_rpm']:.1f} rpm")
    print(f"  spin_eff   = {params['spin_efficiency']:.3f}")
    print(f"  pitch_type = {params.get('pitch_type')}")


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Test 1: Position/velocity back-propagation
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Test 1: Release-point back-propagation")
    print("=" * 60)

    sample = {
        "release_pos_x": -2.0,       # ft
        "release_pos_z": 6.0,        # ft
        "release_extension": 6.5,    # ft
        "vx0": 5.0,                  # ft/s at y=50ft
        "vy0": -130.0,               # ft/s at y=50ft
        "vz0": -5.0,                 # ft/s at y=50ft
        "ax": -10.0,                 # ft/s^2
        "ay": 28.0,                  # ft/s^2
        "az": -15.0,                 # ft/s^2
        "release_spin_rate": 2300,   # rpm
        "spin_axis": 210,            # deg
        "pitch_type": "FF",
    }

    result = statcast_to_release(sample)
    print_conversion(result)

    # ---------------------------------------------------------------
    # Test 2: Spin decomposition sanity checks
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Spin decomposition (known cases)")
    print("=" * 60)

    # Pure backspin (12 o'clock topspin axis → 6 o'clock = 180°)
    # spin_axis=180° → backspin = -ω*sin(180°) = 0, sidespin = ω*cos(180°) = -ω
    # Actually: spin_axis=270° (9 o'clock axis) → pure backspin
    # spin_axis=270° → backspin = -ω*sin(270°) = ω (positive rise), sidespin = ω*cos(270°) = 0
    test_cases = [
        {"name": "Pure backspin (axis at 9 o'clock)",
         "release_spin_rate": 2000, "spin_axis": 270},
        {"name": "Pure topspin (axis at 3 o'clock)",
         "release_spin_rate": 2000, "spin_axis": 90},
        {"name": "RHP 4-seam (axis ~210°)",
         "release_spin_rate": 2300, "spin_axis": 210},
        {"name": "RHP curveball (axis ~45°)",
         "release_spin_rate": 2800, "spin_axis": 45},
    ]

    for tc in test_cases:
        spin = statcast_spin_to_bsg(tc, theta_deg=-3.0, phi_deg=1.0)
        print(f"\n  {tc['name']}:")
        print(f"    spin_axis={tc['spin_axis']}°, total={tc['release_spin_rate']} rpm")
        print(f"    → backspin={spin['backspin_rpm']:+.1f}, sidespin={spin['sidespin_rpm']:+.1f}, gyro={spin['wg_rpm']:.1f}")

    # ---------------------------------------------------------------
    # Test 3: Full pipeline (Statcast -> sim params)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: Full pipeline (Statcast → Simulator)")
    print("=" * 60)

    full_sample = {
        "release_pos_x": -2.0,
        "release_pos_z": 6.0,
        "release_extension": 6.5,
        "vx0": 5.0,
        "vy0": -130.0,
        "vz0": -5.0,
        "ax": -10.0,
        "ay": 28.0,
        "az": -15.0,
        "release_spin_rate": 2300,
        "spin_axis": 210,
        "pfx_x": -7.5,
        "pfx_z": 15.0,
        "release_speed": 94.0,
        "pitch_type": "FF",
    }

    params = statcast_to_sim_params(full_sample)
    print_sim_params(params)
