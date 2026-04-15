"""
SequenceQueryEngine — high-level query API for pitch sequence analysis.

Integrates AtBatBuilder, SequenceAnalyzer, PatternMatcher, and
SequenceVisualizer into a single user-facing interface.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import pandas as pd

from .at_bat_builder import AtBatBuilder
from .models import (
    AtBat,
    AtBatResult,
    AtBatResultCategory,
    PitchResult,
)
from .pattern_matcher import PatternMatcher, PatternQuery
from .sequence_analyzer import SequenceAnalyzer

logger = logging.getLogger(__name__)


class SequenceQueryEngine:
    """High-level interface combining builder, analyzer, and matcher."""

    def __init__(self, simulator_factory: Optional[Callable] = None):
        self.builder = AtBatBuilder()
        self.analyzer = SequenceAnalyzer(simulator_factory)
        self.matcher = PatternMatcher()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_statcast_game(
        self,
        mlbam_id: int,
        year: int,
        date: str,
        analyze: bool = True,
    ) -> List[AtBat]:
        """
        Fetch Statcast data for a game and build analyzed at-bats.

        Parameters
        ----------
        mlbam_id : int
            Pitcher MLBAM ID.
        year : int
            Season year.
        date : str
            Game date (YYYY-MM-DD).
        analyze : bool
            If True, run SequenceAnalyzer on all at-bats.
        """
        from statcast_fetcher import StatcastFetcher

        fetcher = StatcastFetcher()
        df = fetcher.search_pitcher_by_id(mlbam_id, year)
        df_game = fetcher.filter_by_date(df, date)

        pitcher_name = ""
        if "player_name" in df_game.columns and len(df_game) > 0:
            pitcher_name = str(df_game.iloc[0]["player_name"])

        game_pk = None
        if "game_pk" in df_game.columns and len(df_game) > 0:
            game_pk = int(df_game.iloc[0]["game_pk"])

        at_bats = AtBatBuilder.from_statcast(df_game, pitcher_name, game_pk)

        if analyze:
            self.analyzer.analyze_game(at_bats)

        return at_bats

    def load_statcast_df(
        self,
        df: pd.DataFrame,
        pitcher_name: str = "",
        analyze: bool = True,
    ) -> List[AtBat]:
        """
        Build at-bats from an existing Statcast DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-filtered Statcast data (e.g., one game for one pitcher).
        pitcher_name : str
            Pitcher display name.
        analyze : bool
            If True, run SequenceAnalyzer on all at-bats.
        """
        game_pk = None
        if "game_pk" in df.columns and len(df) > 0:
            game_pk = int(df.iloc[0]["game_pk"])

        at_bats = AtBatBuilder.from_statcast(df, pitcher_name, game_pk)

        if analyze:
            self.analyzer.analyze_game(at_bats)

        return at_bats

    def load_hawkeye_csv(
        self,
        csv_path: str,
        analyze: bool = True,
    ) -> List[AtBat]:
        """Load HawkEye CSV — not yet implemented."""
        at_bats = AtBatBuilder.from_hawkeye_csv(csv_path)
        if analyze:
            self.analyzer.analyze_game(at_bats)
        return at_bats

    # ------------------------------------------------------------------
    # Analysis queries
    # ------------------------------------------------------------------

    def strikeout_sequences(
        self,
        at_bats: List[AtBat],
        batter_name: Optional[str] = None,
    ) -> List[AtBat]:
        """Find all strikeout at-bats, optionally filtered by batter."""
        q = PatternQuery(
            result_category=AtBatResultCategory.STRIKEOUT,
            batter_name=batter_name,
        )
        return self.matcher.match(at_bats, q)

    def groundball_sequences(
        self,
        at_bats: List[AtBat],
        batter_name: Optional[str] = None,
    ) -> List[AtBat]:
        """Find all ground-ball at-bats."""
        q = PatternQuery(
            result_category=AtBatResultCategory.GROUNDBALL,
            batter_name=batter_name,
        )
        return self.matcher.match(at_bats, q)

    def flyball_sequences(
        self,
        at_bats: List[AtBat],
        batter_name: Optional[str] = None,
    ) -> List[AtBat]:
        """Find all fly-ball at-bats."""
        q = PatternQuery(
            result_category=AtBatResultCategory.FLYBALL,
            batter_name=batter_name,
        )
        return self.matcher.match(at_bats, q)

    def transition_matrix(
        self,
        at_bats: List[AtBat],
        count: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Pitch-type transition frequency matrix.

        Returns DataFrame with from_type as index and to_type as columns.
        """
        raw = self.matcher.count_transitions(at_bats, from_count=count)
        if not raw:
            return pd.DataFrame()

        rows = []
        for (from_t, to_t), n in raw.items():
            rows.append({"from": from_t, "to": to_t, "count": n})
        df = pd.DataFrame(rows)
        return df.pivot_table(index="from", columns="to",
                              values="count", fill_value=0)

    def tempo_filtered_results(
        self,
        at_bats: List[AtBat],
        min_tempo_ms: float = 40.0,
        result_filter: Optional[PitchResult] = None,
    ) -> pd.DataFrame:
        """
        Find pitch pairs with tempo differential above threshold.

        Optionally filter by pitch B's result (e.g., SWINGING_STRIKE).
        """
        rows = []
        for ab in at_bats:
            if not ab.sequence_metrics:
                continue
            for td in ab.sequence_metrics.tempo_differentials:
                if abs(td.differential_ms) < min_tempo_ms:
                    continue
                pb = ab.pitches[td.pitch_b_idx]
                if result_filter and pb.description != result_filter:
                    continue
                rows.append({
                    "pitcher": ab.pitcher_name,
                    "batter": ab.batter_name,
                    "ab_number": ab.at_bat_number,
                    "inning": ab.inning,
                    "pitch_a_type": td.pitch_a_type,
                    "pitch_b_type": td.pitch_b_type,
                    "tempo_diff_ms": td.differential_ms,
                    "pitch_b_result": pb.description.value if pb.description else None,
                    "count": f"{pb.balls}-{pb.strikes}",
                })
        return pd.DataFrame(rows)

    def tunnel_distribution(
        self,
        at_bats: List[AtBat],
        type_a: str = "FF",
        type_b: str = "SL",
    ) -> pd.DataFrame:
        """
        Tunnel distance distribution for a specific pitch-type pair.
        """
        rows = []
        for ab in at_bats:
            if not ab.sequence_metrics:
                continue
            for t in ab.sequence_metrics.tunnel_analyses:
                if t.pitch_a_type == type_a and t.pitch_b_type == type_b:
                    pb = ab.pitches[t.pitch_b_idx]
                    rows.append({
                        "batter": ab.batter_name,
                        "ab_number": ab.at_bat_number,
                        "tunnel_distance_m": t.tunnel_distance_m,
                        "tunnel_time_s": t.tunnel_time_s,
                        "plate_separation_m": t.plate_separation_m,
                        "pitch_b_result": pb.description.value if pb.description else None,
                        "is_whiff": pb.is_whiff,
                    })
        return pd.DataFrame(rows)

    def nr_timing_comparison(self, at_bats: List[AtBat]) -> pd.DataFrame:
        """
        Compare NR timing between whiffs and contact (HawkEye only).

        Groups by pitch result type and returns:
            result_type, n, mean_swing_duration_ms, mean_timing_margin_ms,
            pct_nr_before_tunnel
        """
        rows = []
        for ab in at_bats:
            if not ab.sequence_metrics:
                continue
            for nr in ab.sequence_metrics.nr_analyses:
                p = ab.pitches[nr.pitch_idx]
                result_type = "other"
                if p.is_whiff:
                    result_type = "whiff"
                elif p.description and p.description in (
                    PitchResult.HIT_INTO_PLAY,
                    PitchResult.HIT_INTO_PLAY_NO_OUT,
                    PitchResult.HIT_INTO_PLAY_SCORE,
                    PitchResult.FOUL,
                    PitchResult.FOUL_TIP,
                ):
                    result_type = "contact"

                rows.append({
                    "result_type": result_type,
                    "swing_duration_ms": nr.swing_duration_ms,
                    "timing_margin_ms": nr.timing_margin_ms,
                    "nr_before_tunnel": nr.nr_before_tunnel,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        summary = df.groupby("result_type").agg(
            n=("result_type", "count"),
            mean_swing_duration_ms=("swing_duration_ms", "mean"),
            mean_timing_margin_ms=("timing_margin_ms", "mean"),
            pct_nr_before_tunnel=("nr_before_tunnel", "mean"),
        ).reset_index()

        return summary
