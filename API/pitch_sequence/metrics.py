"""
MetricsComputer — stateless computation of all dynamic pitch-sequence metrics.

Metrics:
    1. Tempo differential (arrival time difference)
    2. Tunnel effect (trajectory divergence distance)
    3. Movement vector (plate location shift)
    4. Reaction mismatch (batter timing model)
    5. NR (Natural Release) timing analysis (HawkEye only)
"""

from __future__ import annotations

import bisect
import math
from typing import List, Optional, Tuple

from .models import (
    HawkEyeData,
    MovementVector,
    NRAnalysis,
    Pitch,
    PitchResult,
    ReactionMismatch,
    SimResult,
    TempoDifferential,
    TunnelAnalysis,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_REACTION_S = 0.200            # baseline human reaction time (seconds)
DIVERGENCE_THRESHOLD_M = 0.0254     # 1 inch = 25.4 mm


# ---------------------------------------------------------------------------
# MetricsComputer
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Stateless computation of all dynamic metrics."""

    # ------------------------------------------------------------------
    # 1. Tempo differential
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tempo_differential(
        pitch_a: Pitch,
        pitch_b: Pitch,
        a_idx: int,
        b_idx: int,
    ) -> Optional[TempoDifferential]:
        """
        Compute arrival-time difference between two consecutive pitches.

        Returns None if either pitch lacks sim_result.
        Positive differential_ms means pitch B is slower (arrives later).
        """
        if not pitch_a.sim_result or not pitch_b.sim_result:
            return None

        arr_a = pitch_a.sim_result.arrival_time_s
        arr_b = pitch_b.sim_result.arrival_time_s

        return TempoDifferential(
            pitch_a_idx=a_idx,
            pitch_b_idx=b_idx,
            pitch_a_type=pitch_a.pitch_type,
            pitch_b_type=pitch_b.pitch_type,
            arrival_time_a_s=arr_a,
            arrival_time_b_s=arr_b,
            differential_ms=(arr_b - arr_a) * 1000.0,
        )

    # ------------------------------------------------------------------
    # 2. Tunnel effect
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tunnel(
        pitch_a: Pitch,
        pitch_b: Pitch,
        a_idx: int,
        b_idx: int,
        threshold_m: float = DIVERGENCE_THRESHOLD_M,
    ) -> Optional[TunnelAnalysis]:
        """
        Compute tunnel distance between two pitches.

        Algorithm:
        1. Shift pitch B trajectory to match pitch A release point.
        2. Walk through common time grid (dt=0.001s).
        3. Find first time where 3D distance > threshold.
        4. tunnel_distance = y_release - y_at_tunnel.
        """
        if not pitch_a.sim_result or not pitch_b.sim_result:
            return None

        traj_a = pitch_a.sim_result.trajectory
        traj_b = pitch_b.sim_result.trajectory
        if not traj_a or not traj_b:
            return None

        # Release positions from sim_params
        if not pitch_a.sim_params or not pitch_b.sim_params:
            return None

        rel_a = (pitch_a.sim_params.x0, pitch_a.sim_params.y0, pitch_a.sim_params.z0)
        rel_b = (pitch_b.sim_params.x0, pitch_b.sim_params.y0, pitch_b.sim_params.z0)

        # Offset to align release points
        offset = (rel_a[0] - rel_b[0], rel_a[1] - rel_b[1], rel_a[2] - rel_b[2])

        # Time range
        arr_a = pitch_a.sim_result.arrival_time_s
        arr_b = pitch_b.sim_result.arrival_time_s
        t_max = min(arr_a, arr_b)

        dt = 0.001
        tunnel_time = t_max  # default: no divergence found
        tunnel_pos_a = _interpolate_position(traj_a, t_max)
        tunnel_pos_b_raw = _interpolate_position(traj_b, t_max)
        tunnel_pos_b = (
            tunnel_pos_b_raw[0] + offset[0],
            tunnel_pos_b_raw[1] + offset[1],
            tunnel_pos_b_raw[2] + offset[2],
        )

        t = 0.0
        while t <= t_max:
            pos_a = _interpolate_position(traj_a, t)
            pos_b_raw = _interpolate_position(traj_b, t)
            pos_b = (
                pos_b_raw[0] + offset[0],
                pos_b_raw[1] + offset[1],
                pos_b_raw[2] + offset[2],
            )

            dist = math.sqrt(
                (pos_a[0] - pos_b[0]) ** 2 +
                (pos_a[1] - pos_b[1]) ** 2 +
                (pos_a[2] - pos_b[2]) ** 2
            )

            if dist > threshold_m:
                tunnel_time = t
                tunnel_pos_a = pos_a
                tunnel_pos_b = pos_b
                break

            t += dt

        # Tunnel distance (along y-axis: release → tunnel point)
        tunnel_distance = abs(rel_a[1] - tunnel_pos_a[1])

        # Plate separation
        hp_a = pitch_a.sim_result.home_plate_crossing
        hp_b = pitch_b.sim_result.home_plate_crossing
        plate_sep = 0.0
        if hp_a and hp_b:
            plate_sep = math.sqrt(
                (hp_a["x"] - hp_b["x"] - offset[0]) ** 2 +
                (hp_a["z"] - hp_b["z"] - offset[2]) ** 2
            )

        return TunnelAnalysis(
            pitch_a_idx=a_idx,
            pitch_b_idx=b_idx,
            pitch_a_type=pitch_a.pitch_type,
            pitch_b_type=pitch_b.pitch_type,
            tunnel_distance_m=tunnel_distance,
            tunnel_time_s=tunnel_time,
            divergence_threshold_m=threshold_m,
            tunnel_point_a=tunnel_pos_a,
            tunnel_point_b=tunnel_pos_b,
            plate_separation_m=plate_sep,
        )

    # ------------------------------------------------------------------
    # 3. Movement vector
    # ------------------------------------------------------------------

    @staticmethod
    def compute_movement_vector(
        pitch_a: Pitch,
        pitch_b: Pitch,
        a_idx: int,
        b_idx: int,
    ) -> Optional[MovementVector]:
        """Compute plate-location shift between consecutive pitches."""
        if pitch_a.plate_x is None or pitch_a.plate_z is None:
            return None
        if pitch_b.plate_x is None or pitch_b.plate_z is None:
            return None

        dx = pitch_b.plate_x - pitch_a.plate_x
        dz = pitch_b.plate_z - pitch_a.plate_z
        mag = math.sqrt(dx ** 2 + dz ** 2)
        direction = math.degrees(math.atan2(dz, dx))

        return MovementVector(
            pitch_a_idx=a_idx,
            pitch_b_idx=b_idx,
            dx_m=dx,
            dz_m=dz,
            magnitude_m=mag,
            direction_deg=direction,
        )

    # ------------------------------------------------------------------
    # 4. Reaction mismatch
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reaction_mismatch(
        pitch_a: Pitch,
        pitch_b: Pitch,
        a_idx: int,
        b_idx: int,
    ) -> Optional[ReactionMismatch]:
        """
        Model batter timing mismatch.

        If batter timed to pitch A, how off is pitch B?
        "late" = pitch B is slower (batter swings early).
        "early" = pitch B is faster (batter is late).
        """
        if not pitch_a.sim_result or not pitch_b.sim_result:
            return None

        arr_a = pitch_a.sim_result.arrival_time_s
        arr_b = pitch_b.sim_result.arrival_time_s
        mismatch_ms = (arr_b - arr_a) * 1000.0
        direction = "late" if arr_b > arr_a else "early"

        return ReactionMismatch(
            pitch_a_idx=a_idx,
            pitch_b_idx=b_idx,
            timing_mismatch_ms=mismatch_ms,
            decision_window_a_ms=(arr_a - HUMAN_REACTION_S) * 1000.0,
            decision_window_b_ms=(arr_b - HUMAN_REACTION_S) * 1000.0,
            mismatch_direction=direction,
            pitch_b_result=pitch_b.description,
        )

    # ------------------------------------------------------------------
    # 5. NR (Natural Release) timing analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_nr_analysis(
        pitch: Pitch,
        tunnel: Optional[TunnelAnalysis] = None,
    ) -> Optional[NRAnalysis]:
        """
        Analyze Natural Release timing relative to pitch arrival and tunnel.

        HawkEye only — returns None if grip_max_time unavailable.

        Time reference conversion:
            grip_max_time, impact_time: absolute HawkEye time
            release_time: absolute HawkEye time
            arrival_time: simulator time (relative to release = 0)

        NR relative to release:
            nr_rel = grip_max_time - release_time
        """
        if not pitch.hawkeye or pitch.hawkeye.grip_max_time is None:
            return None
        if not pitch.sim_result:
            return None
        if pitch.hawkeye.release_time is None:
            return None

        he = pitch.hawkeye
        release_t = he.release_time
        nr_t = he.grip_max_time
        impact_t = he.impact_time
        arrival_s = pitch.sim_result.arrival_time_s

        # Swing duration
        swing_duration_ms = None
        if impact_t is not None:
            swing_duration_ms = (impact_t - nr_t) * 1000.0

        decision_window_ms = (arrival_s - HUMAN_REACTION_S) * 1000.0

        # Timing margin
        timing_margin_ms = None
        if swing_duration_ms is not None:
            timing_margin_ms = decision_window_ms - swing_duration_ms

        # NR before tunnel?
        nr_before_tunnel = None
        if tunnel is not None:
            nr_rel = nr_t - release_t  # NR time relative to release
            nr_before_tunnel = nr_rel < tunnel.tunnel_time_s

        return NRAnalysis(
            pitch_idx=pitch.pitch_number_in_ab - 1,  # 0-based
            grip_max_time_s=nr_t,
            impact_time_s=impact_t,
            arrival_time_s=arrival_s,
            swing_duration_ms=swing_duration_ms,
            decision_window_ms=decision_window_ms,
            nr_before_tunnel=nr_before_tunnel,
            timing_margin_ms=timing_margin_ms,
        )


# ---------------------------------------------------------------------------
# Trajectory interpolation utility
# ---------------------------------------------------------------------------

def _interpolate_position(
    trajectory: List[dict],
    t: float,
) -> Tuple[float, float, float]:
    """
    Linear interpolation of trajectory position at time *t*.

    Uses binary search to find bracketing points, then lerp.
    trajectory must be sorted by 't' (ascending).
    """
    if not trajectory:
        return (0.0, 0.0, 0.0)

    times = [p["t"] for p in trajectory]

    # Clamp to trajectory bounds
    if t <= times[0]:
        p = trajectory[0]
        return (p["x"], p["y"], p["z"])
    if t >= times[-1]:
        p = trajectory[-1]
        return (p["x"], p["y"], p["z"])

    # Binary search
    idx = bisect.bisect_right(times, t) - 1
    p0 = trajectory[idx]
    p1 = trajectory[idx + 1]

    dt = p1["t"] - p0["t"]
    if dt == 0:
        return (p0["x"], p0["y"], p0["z"])

    frac = (t - p0["t"]) / dt
    return (
        p0["x"] + frac * (p1["x"] - p0["x"]),
        p0["y"] + frac * (p1["y"] - p0["y"]),
        p0["z"] + frac * (p1["z"] - p0["z"]),
    )
