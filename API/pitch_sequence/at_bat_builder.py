"""
AtBatBuilder — group individual pitch data into AtBat objects.

Supports Statcast (via pybaseball DataFrame) and HawkEye CSV (placeholder).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import (
    AtBat,
    AtBatResult,
    AtBatResultCategory,
    DataSource,
    HawkEyeData,
    Pitch,
    PitchResult,
    SimParameters,
)

logger = logging.getLogger(__name__)

FT_TO_M = 0.3048


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_DESCRIPTION_MAP: Dict[str, PitchResult] = {
    "ball": PitchResult.BALL,
    "blocked_ball": PitchResult.BALL,
    "called_strike": PitchResult.CALLED_STRIKE,
    "swinging_strike": PitchResult.SWINGING_STRIKE,
    "swinging_strike_blocked": PitchResult.SWINGING_STRIKE_BLOCKED,
    "foul": PitchResult.FOUL,
    "foul_tip": PitchResult.FOUL_TIP,
    "foul_bunt": PitchResult.FOUL_BUNT,
    "missed_bunt": PitchResult.MISSED_BUNT,
    "hit_into_play": PitchResult.HIT_INTO_PLAY,
    "hit_into_play_no_out": PitchResult.HIT_INTO_PLAY_NO_OUT,
    "hit_into_play_score": PitchResult.HIT_INTO_PLAY_SCORE,
    "hit_by_pitch": PitchResult.HIT_BY_PITCH,
    "pitchout": PitchResult.PITCHOUT,
}

_EVENT_MAP: Dict[str, AtBatResult] = {
    "strikeout": AtBatResult.STRIKEOUT,
    "strikeout_double_play": AtBatResult.STRIKEOUT_DOUBLE_PLAY,
    "walk": AtBatResult.WALK,
    "single": AtBatResult.SINGLE,
    "double": AtBatResult.DOUBLE,
    "triple": AtBatResult.TRIPLE,
    "home_run": AtBatResult.HOME_RUN,
    "field_out": AtBatResult.FIELD_OUT,
    "force_out": AtBatResult.FORCE_OUT,
    "grounded_into_double_play": AtBatResult.GROUNDED_INTO_DP,
    "double_play": AtBatResult.DOUBLE_PLAY,
    "fielders_choice": AtBatResult.FIELDERS_CHOICE,
    "fielders_choice_out": AtBatResult.FIELDERS_CHOICE,
    "sac_fly": AtBatResult.SAC_FLY,
    "sac_bunt": AtBatResult.SAC_BUNT,
    "sac_fly_double_play": AtBatResult.SAC_FLY_DOUBLE_PLAY,
    "hit_by_pitch": AtBatResult.HIT_BY_PITCH,
    "field_error": AtBatResult.FIELD_ERROR,
    "catcher_interf": AtBatResult.CATCHER_INTERF,
}

_WHIFF_DESCRIPTIONS = {
    PitchResult.SWINGING_STRIKE,
    PitchResult.SWINGING_STRIKE_BLOCKED,
    PitchResult.MISSED_BUNT,
}


# ---------------------------------------------------------------------------
# AtBatBuilder
# ---------------------------------------------------------------------------

class AtBatBuilder:
    """Group pitches into AtBat objects from various data sources."""

    # ------------------------------------------------------------------
    # Statcast
    # ------------------------------------------------------------------

    @staticmethod
    def from_statcast(
        df: pd.DataFrame,
        pitcher_name: str = "",
        game_pk: Optional[int] = None,
    ) -> List[AtBat]:
        """
        Build AtBat list from a Statcast DataFrame (typically one game).

        Parameters
        ----------
        df : pd.DataFrame
            Statcast pitch-level data.  Must contain at least:
            at_bat_number, pitch_number, pitch_type, balls, strikes,
            description, plate_x, plate_z, sz_top, sz_bot, stand.
        pitcher_name : str
            Pitcher display name (for AtBat.pitcher_name).
        game_pk : int, optional
            Game identifier for pitch_id generation.

        Returns
        -------
        List[AtBat]
        """
        df = df.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)

        at_bats: List[AtBat] = []

        for ab_num, grp in df.groupby("at_bat_number", sort=True):
            pitches: List[Pitch] = []
            for seq, (_, row) in enumerate(grp.iterrows(), start=1):
                pitches.append(AtBatBuilder._statcast_row_to_pitch(
                    row, seq, game_pk=game_pk,
                ))

            # Determine AB result from last pitch's events column
            last_row = grp.iloc[-1]
            ab_result = AtBatBuilder._map_event(last_row.get("events"))
            bb_type = last_row.get("bb_type") if pd.notna(last_row.get("bb_type")) else None
            ab_category = AtBatBuilder._infer_category(ab_result, bb_type)

            batter_name = str(last_row.get("batter_name", "")) if pd.notna(last_row.get("batter_name")) else ""
            stand = str(last_row.get("stand", "")) if pd.notna(last_row.get("stand")) else ""
            inning = int(last_row["inning"]) if pd.notna(last_row.get("inning")) else None

            at_bats.append(AtBat(
                at_bat_id=f"sc_{game_pk or 0}_{ab_num}",
                at_bat_number=int(ab_num),
                pitcher_name=pitcher_name,
                batter_name=batter_name,
                stand=stand,
                pitches=pitches,
                result=ab_result,
                result_category=ab_category,
                inning=inning,
            ))

        return at_bats

    # ------------------------------------------------------------------
    # HawkEye CSV (placeholder — implement after CSV structure confirmed)
    # ------------------------------------------------------------------

    @staticmethod
    def from_hawkeye_csv(csv_path: str) -> List[AtBat]:
        """
        Build AtBat list from HawkEye CSV (Snowflake-converted).

        Not yet implemented — waiting for CSV column structure.
        """
        raise NotImplementedError(
            "HawkEye CSV reader is not yet implemented.  "
            "Waiting for Snowflake-converted CSV column structure."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _statcast_row_to_pitch(
        row: pd.Series,
        pitch_number_in_ab: int,
        game_pk: Optional[int] = None,
    ) -> Pitch:
        """Convert a single Statcast DataFrame row to a Pitch object."""

        ab_num = int(row["at_bat_number"])
        pitch_type = str(row.get("pitch_type", "UN"))

        # Description → PitchResult
        desc = AtBatBuilder._map_description(row.get("description"))
        is_whiff = desc in _WHIFF_DESCRIPTIONS if desc else False

        # Plate location (feet → meters)
        plate_x = float(row["plate_x"]) * FT_TO_M if pd.notna(row.get("plate_x")) else None
        plate_z = float(row["plate_z"]) * FT_TO_M if pd.notna(row.get("plate_z")) else None
        sz_top = float(row["sz_top"]) * FT_TO_M if pd.notna(row.get("sz_top")) else None
        sz_bot = float(row["sz_bot"]) * FT_TO_M if pd.notna(row.get("sz_bot")) else None

        # Zone check
        is_in_zone = False
        if plate_x is not None and plate_z is not None and sz_top is not None and sz_bot is not None:
            hw = 0.708 * FT_TO_M  # half-width of strike zone
            is_in_zone = (-hw <= plate_x <= hw) and (sz_bot <= plate_z <= sz_top)

        # Spin
        spin_rate = float(row["release_spin_rate"]) if pd.notna(row.get("release_spin_rate")) else None
        spin_axis = float(row["spin_axis"]) if pd.notna(row.get("spin_axis")) else None

        # Release speed
        release_speed_mps = None
        if pd.notna(row.get("release_speed")):
            release_speed_mps = float(row["release_speed"]) * 0.44704  # mph → m/s

        # Release position
        release_pos = None
        if (pd.notna(row.get("release_pos_x")) and
                pd.notna(row.get("release_pos_z")) and
                pd.notna(row.get("release_extension"))):
            release_pos = (
                float(row["release_pos_x"]) * FT_TO_M,
                (60.5 - float(row["release_extension"])) * FT_TO_M,
                float(row["release_pos_z"]) * FT_TO_M,
            )

        # SimParameters — built lazily by SequenceAnalyzer using statcast_to_sim_params
        # We store statcast_raw so the analyzer can call it later.
        sim_params = None
        try:
            sim_params = AtBatBuilder._try_build_sim_params(row)
        except Exception:
            pass  # insufficient data for simulation

        # Batter info
        batter_id = int(row["batter"]) if pd.notna(row.get("batter")) else None
        batter_name = str(row.get("batter_name", "")) if pd.notna(row.get("batter_name")) else None
        stand = str(row.get("stand", "")) if pd.notna(row.get("stand")) else None

        # Batted ball
        launch_speed = float(row["launch_speed"]) if pd.notna(row.get("launch_speed")) else None
        launch_angle = float(row["launch_angle"]) if pd.notna(row.get("launch_angle")) else None

        return Pitch(
            pitch_id=f"sc_{game_pk or 0}_{ab_num}_{pitch_number_in_ab}",
            source=DataSource.STATCAST,
            at_bat_number=ab_num,
            pitch_number_in_ab=pitch_number_in_ab,
            balls=int(row["balls"]) if pd.notna(row.get("balls")) else 0,
            strikes=int(row["strikes"]) if pd.notna(row.get("strikes")) else 0,
            pitch_type=pitch_type,
            release_speed_mps=release_speed_mps,
            release_pos=release_pos,
            plate_x=plate_x,
            plate_z=plate_z,
            sz_top=sz_top,
            sz_bot=sz_bot,
            spin_rate_rpm=spin_rate,
            spin_axis_deg=spin_axis,
            description=desc,
            is_whiff=is_whiff,
            is_in_zone=is_in_zone,
            batter_id=batter_id,
            batter_name=batter_name,
            stand=stand,
            launch_speed=launch_speed,
            launch_angle=launch_angle,
            sim_params=sim_params,
            statcast_raw=row.to_dict(),
        )

    @staticmethod
    def _try_build_sim_params(row: pd.Series) -> Optional[SimParameters]:
        """Try to build SimParameters from Statcast row using statcast_to_sim_params."""
        required = ["release_pos_x", "release_pos_z", "release_extension",
                     "vx0", "vy0", "vz0", "ax", "ay", "az",
                     "release_spin_rate", "spin_axis", "pfx_x", "pfx_z"]
        for col in required:
            if pd.isna(row.get(col)):
                return None

        from statcast_to_sim import statcast_to_sim_params
        params = statcast_to_sim_params(row.to_dict())
        return SimParameters(
            x0=params["x0"],
            y0=params["y0"],
            z0=params["z0"],
            v0_mps=params["v0_mps"],
            theta_deg=params["theta_deg"],
            phi_deg=params["phi_deg"],
            backspin_rpm=params.get("backspin_rpm", 0.0),
            sidespin_rpm=params.get("sidespin_rpm", 0.0),
            wg_rpm=params.get("wg_rpm", 0.0),
            batter_hand=str(row.get("stand", "R")),
        )

    @staticmethod
    def _map_description(desc) -> Optional[PitchResult]:
        """Map Statcast description string to PitchResult enum."""
        if pd.isna(desc) or desc is None:
            return None
        key = str(desc).strip().lower()
        return _DESCRIPTION_MAP.get(key)

    @staticmethod
    def _map_event(event) -> Optional[AtBatResult]:
        """Map Statcast events string to AtBatResult enum."""
        if pd.isna(event) or event is None:
            return None
        key = str(event).strip().lower()
        return _EVENT_MAP.get(key, AtBatResult.OTHER)

    @staticmethod
    def _infer_category(
        result: Optional[AtBatResult],
        bb_type: Optional[str],
    ) -> Optional[AtBatResultCategory]:
        """
        Infer broad category from AtBatResult + batted ball type.

        bb_type (Statcast): "ground_ball", "fly_ball", "line_drive", "popup"
        """
        if result is None:
            return None

        # Direct mapping for non-contact outcomes
        if result in (AtBatResult.STRIKEOUT, AtBatResult.STRIKEOUT_DOUBLE_PLAY):
            return AtBatResultCategory.STRIKEOUT
        if result in (AtBatResult.WALK, AtBatResult.HIT_BY_PITCH, AtBatResult.CATCHER_INTERF):
            return AtBatResultCategory.WALK

        # Batted-ball type driven
        if bb_type:
            bb = bb_type.strip().lower()
            if bb == "ground_ball":
                return AtBatResultCategory.GROUNDBALL
            if bb in ("fly_ball", "popup"):
                return AtBatResultCategory.FLYBALL
            if bb == "line_drive":
                return AtBatResultCategory.LINEDRIVE

        return AtBatResultCategory.OTHER
