"""
Data models for pitch sequence analysis.

Provides a common schema for both HawkEye and Statcast data sources,
plus dynamic metrics dataclasses for tempo differential, tunnel effect,
movement vectors, reaction mismatch, and NR timing analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataSource(Enum):
    HAWKEYE = "hawkeye"
    STATCAST = "statcast"


class PitchResult(Enum):
    """Per-pitch outcome."""
    BALL = "ball"
    CALLED_STRIKE = "called_strike"
    SWINGING_STRIKE = "swinging_strike"
    SWINGING_STRIKE_BLOCKED = "swinging_strike_blocked"
    FOUL = "foul"
    FOUL_TIP = "foul_tip"
    HIT_INTO_PLAY = "hit_into_play"
    HIT_INTO_PLAY_NO_OUT = "hit_into_play_no_out"
    HIT_INTO_PLAY_SCORE = "hit_into_play_score"
    HIT_BY_PITCH = "hit_by_pitch"
    MISSED_BUNT = "missed_bunt"
    FOUL_BUNT = "foul_bunt"
    PITCHOUT = "pitchout"


class AtBatResult(Enum):
    """At-bat outcome."""
    STRIKEOUT = "strikeout"
    STRIKEOUT_DOUBLE_PLAY = "strikeout_double_play"
    WALK = "walk"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"
    FIELD_OUT = "field_out"
    FORCE_OUT = "force_out"
    GROUNDED_INTO_DP = "grounded_into_double_play"
    DOUBLE_PLAY = "double_play"
    FIELDERS_CHOICE = "fielders_choice"
    SAC_FLY = "sac_fly"
    SAC_BUNT = "sac_bunt"
    SAC_FLY_DOUBLE_PLAY = "sac_fly_double_play"
    HIT_BY_PITCH = "hit_by_pitch"
    FIELD_ERROR = "field_error"
    CATCHER_INTERF = "catcher_interf"
    OTHER = "other"


class AtBatResultCategory(Enum):
    """Broad classification of at-bat outcome."""
    STRIKEOUT = "strikeout"
    GROUNDBALL = "groundball"
    FLYBALL = "flyball"
    LINEDRIVE = "linedrive"
    WALK = "walk"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class SimParameters:
    """Simulator input parameters (mirrors PitchParameters)."""
    x0: float
    y0: float
    z0: float
    v0_mps: float
    theta_deg: float
    phi_deg: float
    backspin_rpm: float
    sidespin_rpm: float
    wg_rpm: float
    batter_hand: str = "R"


@dataclass
class SimResult:
    """Cached simulator output."""
    trajectory: List[dict]                      # [{t,x,y,z,vx,vy,vz,...}, ...]
    home_plate_crossing: Optional[dict]         # {t,x,y,z,vx,vy,vz,v,v_mph}
    arrival_time_s: float                       # home_plate_crossing['t']
    arrival_speed_mps: float                    # home_plate_crossing['v']


@dataclass
class HawkEyeData:
    """HawkEye-specific rich data fields."""
    ball_time: Optional[List[float]] = None
    ball_pos: Optional[List[List[float]]] = None
    bat_head: Optional[List[List[float]]] = None
    bat_handle: Optional[List[List[float]]] = None
    grip_max_time: Optional[float] = None       # NR time (absolute)
    impact_time: Optional[float] = None         # impact time (absolute)
    release_time: Optional[float] = None        # release time (absolute)
    release_pos: Optional[List[float]] = None
    vel_time: Optional[List[float]] = None
    vel_head: Optional[List[float]] = None      # head speed (km/h)
    vel_grip: Optional[List[float]] = None      # grip speed (km/h)
    hit_ball_time: Optional[List[float]] = None
    hit_ball_pos: Optional[List[List[float]]] = None
    isa_time: Optional[List[float]] = None
    isa_pos: Optional[List[List[float]]] = None
    isa_axis: Optional[List[List[float]]] = None
    isa_omega: Optional[List[float]] = None


@dataclass
class Pitch:
    """Single pitch data — common model for HawkEye and Statcast."""

    # --- Identity ---
    pitch_id: str                               # "{source}_{game}_{ab}_{pitch}"
    source: DataSource

    # --- Sequence ---
    at_bat_number: int
    pitch_number_in_ab: int                     # 1-indexed
    balls: int
    strikes: int

    # --- Classification ---
    pitch_type: str                             # FF, SL, CU, CH, SI, ...

    # --- Release ---
    release_speed_mps: Optional[float] = None
    release_pos: Optional[Tuple[float, float, float]] = None  # (x,y,z) meters

    # --- Plate crossing ---
    plate_x: Optional[float] = None             # meters
    plate_z: Optional[float] = None             # meters
    sz_top: Optional[float] = None              # meters
    sz_bot: Optional[float] = None              # meters

    # --- Spin ---
    spin_rate_rpm: Optional[float] = None
    spin_axis_deg: Optional[float] = None
    backspin_rpm: Optional[float] = None
    sidespin_rpm: Optional[float] = None
    gyrospin_rpm: Optional[float] = None

    # --- Result ---
    description: Optional[PitchResult] = None
    is_whiff: bool = False
    is_in_zone: bool = False

    # --- Batter ---
    batter_id: Optional[int] = None
    batter_name: Optional[str] = None
    stand: Optional[str] = None                 # R / L

    # --- Batted ball ---
    launch_speed: Optional[float] = None        # mph
    launch_angle: Optional[float] = None        # deg

    # --- Simulator ---
    sim_params: Optional[SimParameters] = None
    sim_result: Optional[SimResult] = None      # populated by SequenceAnalyzer

    # --- Source-specific ---
    hawkeye: Optional[HawkEyeData] = None
    statcast_raw: Optional[dict] = None


@dataclass
class AtBat:
    """Single at-bat containing an ordered sequence of pitches."""
    at_bat_id: str
    at_bat_number: int
    pitcher_name: str
    batter_name: str
    stand: str                                  # R / L
    pitches: List[Pitch]                        # ordered by pitch_number_in_ab
    result: Optional[AtBatResult] = None
    result_category: Optional[AtBatResultCategory] = None
    inning: Optional[int] = None

    # Populated by SequenceAnalyzer
    sequence_metrics: Optional[SequenceMetrics] = None


# ---------------------------------------------------------------------------
# Dynamic metrics
# ---------------------------------------------------------------------------

@dataclass
class TempoDifferential:
    """Arrival-time difference between consecutive pitches."""
    pitch_a_idx: int
    pitch_b_idx: int
    pitch_a_type: str
    pitch_b_type: str
    arrival_time_a_s: float
    arrival_time_b_s: float
    differential_ms: float                      # (B - A) * 1000; positive = slower


@dataclass
class TunnelAnalysis:
    """Tunnel effect between two pitches."""
    pitch_a_idx: int
    pitch_b_idx: int
    pitch_a_type: str
    pitch_b_type: str
    tunnel_distance_m: float                    # release → divergence point
    tunnel_time_s: float                        # time at divergence
    divergence_threshold_m: float               # threshold used (default 1 inch)
    tunnel_point_a: Tuple[float, float, float]
    tunnel_point_b: Tuple[float, float, float]
    plate_separation_m: float                   # 3D distance at plate


@dataclass
class MovementVector:
    """Plate-location shift between consecutive pitches."""
    pitch_a_idx: int
    pitch_b_idx: int
    dx_m: float
    dz_m: float
    magnitude_m: float
    direction_deg: float                        # atan2(dz, dx)


@dataclass
class ReactionMismatch:
    """Batter reaction-time model output."""
    pitch_a_idx: int
    pitch_b_idx: int
    timing_mismatch_ms: float                   # (arrival_b - arrival_a) * 1000
    decision_window_a_ms: float                 # (arrival_a - 0.200) * 1000
    decision_window_b_ms: float                 # (arrival_b - 0.200) * 1000
    mismatch_direction: str                     # "late" or "early"
    pitch_b_result: Optional[PitchResult] = None


@dataclass
class NRAnalysis:
    """Natural Release timing analysis (HawkEye only)."""
    pitch_idx: int
    grip_max_time_s: float                      # NR absolute time
    impact_time_s: Optional[float]              # impact absolute time
    arrival_time_s: float                       # sim arrival time (from release)
    swing_duration_ms: Optional[float]          # (impact - NR) * 1000
    decision_window_ms: float                   # (arrival - 0.200) * 1000
    nr_before_tunnel: Optional[bool] = None     # NR before tunnel divergence?
    timing_margin_ms: Optional[float] = None    # decision_window - swing_duration


@dataclass
class SequenceMetrics:
    """All dynamic metrics for one at-bat."""
    tempo_differentials: List[TempoDifferential] = field(default_factory=list)
    tunnel_analyses: List[TunnelAnalysis] = field(default_factory=list)
    movement_vectors: List[MovementVector] = field(default_factory=list)
    reaction_mismatches: List[ReactionMismatch] = field(default_factory=list)
    nr_analyses: List[NRAnalysis] = field(default_factory=list)
