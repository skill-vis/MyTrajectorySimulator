"""
PatternMatcher — search at-bats by declarative pattern queries.

Supports filtering by count, pitch type sequence, result, dynamic metrics,
and custom predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .models import (
    AtBat,
    AtBatResult,
    AtBatResultCategory,
    Pitch,
    PitchResult,
    SequenceMetrics,
)


@dataclass
class PatternQuery:
    """Declarative pattern specification for at-bat search."""
    count: Optional[Tuple[int, int]] = None             # (balls, strikes) before last pitch
    pitch_types: Optional[List[str]] = None             # full sequence of pitch types
    result: Optional[AtBatResult] = None
    result_category: Optional[AtBatResultCategory] = None
    final_pitch_result: Optional[PitchResult] = None
    min_pitches: Optional[int] = None
    max_pitches: Optional[int] = None
    batter_hand: Optional[str] = None                   # "R" or "L"
    batter_name: Optional[str] = None
    min_tempo_diff_ms: Optional[float] = None
    min_tunnel_distance_m: Optional[float] = None
    custom_filter: Optional[Callable[[AtBat], bool]] = None


class PatternMatcher:
    """Search for at-bats matching declarative patterns."""

    def match(self, at_bats: List[AtBat], query: PatternQuery) -> List[AtBat]:
        """Return at-bats matching ALL specified criteria (AND logic)."""
        results = []
        for ab in at_bats:
            if self._matches(ab, query):
                results.append(ab)
        return results

    def find_subsequence(
        self,
        at_bats: List[AtBat],
        pitch_type_pattern: List[str],
    ) -> List[AtBat]:
        """
        Find at-bats containing a sub-sequence of pitch types.

        E.g., pitch_type_pattern=["FF", "FF", "SL"] finds at-bats where
        three consecutive pitches are FF, FF, SL anywhere in the AB.
        """
        n = len(pitch_type_pattern)
        results = []
        for ab in at_bats:
            types = [p.pitch_type for p in ab.pitches]
            for i in range(len(types) - n + 1):
                if types[i:i + n] == pitch_type_pattern:
                    results.append(ab)
                    break
        return results

    def count_transitions(
        self,
        at_bats: List[AtBat],
        from_count: Optional[Tuple[int, int]] = None,
    ) -> Dict[Tuple[str, str], int]:
        """
        Count pitch-type transitions across all at-bats.

        Returns {("FF", "SL"): 15, ("FF", "CH"): 8, ...}

        Parameters
        ----------
        from_count : (balls, strikes), optional
            If given, only count transitions where pitch_a was thrown
            at that specific count.
        """
        counts: Dict[Tuple[str, str], int] = {}
        for ab in at_bats:
            for i in range(len(ab.pitches) - 1):
                pa = ab.pitches[i]
                pb = ab.pitches[i + 1]

                if from_count is not None:
                    if (pa.balls, pa.strikes) != from_count:
                        continue

                key = (pa.pitch_type, pb.pitch_type)
                counts[key] = counts.get(key, 0) + 1

        return counts

    def aggregate_metrics(
        self,
        at_bats: List[AtBat],
        group_by: str = "pitch_type_pair",
    ) -> pd.DataFrame:
        """
        Aggregate dynamic metrics across at-bats.

        group_by : "pitch_type_pair" | "count" | "result_category"

        Returns DataFrame with columns:
            group_key, n, mean_tempo_diff_ms, mean_tunnel_distance_m,
            mean_timing_mismatch_ms, whiff_rate
        """
        rows = []
        for ab in at_bats:
            if not ab.sequence_metrics:
                continue

            sm = ab.sequence_metrics

            for td in sm.tempo_differentials:
                row = {
                    "pitch_type_pair": f"{td.pitch_a_type}->{td.pitch_b_type}",
                    "count": f"{ab.pitches[td.pitch_b_idx].balls}-{ab.pitches[td.pitch_b_idx].strikes}",
                    "result_category": ab.result_category.value if ab.result_category else "unknown",
                    "tempo_diff_ms": td.differential_ms,
                    "pitch_b_type": td.pitch_b_type,
                }

                # Find matching tunnel for this pair
                tunnel = next((t for t in sm.tunnel_analyses
                               if t.pitch_a_idx == td.pitch_a_idx and
                               t.pitch_b_idx == td.pitch_b_idx), None)
                row["tunnel_distance_m"] = tunnel.tunnel_distance_m if tunnel else None

                # Find matching reaction mismatch
                react = next((r for r in sm.reaction_mismatches
                              if r.pitch_a_idx == td.pitch_a_idx and
                              r.pitch_b_idx == td.pitch_b_idx), None)
                row["timing_mismatch_ms"] = react.timing_mismatch_ms if react else None

                # Whiff on pitch B?
                pb = ab.pitches[td.pitch_b_idx]
                row["is_whiff"] = pb.is_whiff

                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        agg_funcs = {
            "tempo_diff_ms": "mean",
            "tunnel_distance_m": "mean",
            "timing_mismatch_ms": "mean",
            "is_whiff": ["sum", "count"],
        }

        grouped = df.groupby(group_by).agg(agg_funcs)
        grouped.columns = ["mean_tempo_diff_ms", "mean_tunnel_distance_m",
                           "mean_timing_mismatch_ms", "whiff_count", "n"]
        grouped["whiff_rate"] = grouped["whiff_count"] / grouped["n"]
        grouped = grouped.drop(columns=["whiff_count"]).reset_index()
        grouped = grouped.rename(columns={group_by: "group_key"})
        grouped = grouped.sort_values("n", ascending=False)

        return grouped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _matches(self, ab: AtBat, q: PatternQuery) -> bool:
        """Check if at-bat matches all query criteria."""

        if q.result is not None and ab.result != q.result:
            return False

        if q.result_category is not None and ab.result_category != q.result_category:
            return False

        if q.batter_hand is not None and ab.stand != q.batter_hand:
            return False

        if q.batter_name is not None:
            if q.batter_name.lower() not in ab.batter_name.lower():
                return False

        n = len(ab.pitches)

        if q.min_pitches is not None and n < q.min_pitches:
            return False
        if q.max_pitches is not None and n > q.max_pitches:
            return False

        if q.count is not None:
            last = ab.pitches[-1]
            if (last.balls, last.strikes) != q.count:
                return False

        if q.pitch_types is not None:
            types = [p.pitch_type for p in ab.pitches]
            if types != q.pitch_types:
                return False

        if q.final_pitch_result is not None:
            last_desc = ab.pitches[-1].description
            if last_desc != q.final_pitch_result:
                return False

        # Dynamic metrics filters
        if q.min_tempo_diff_ms is not None and ab.sequence_metrics:
            if not any(abs(td.differential_ms) >= q.min_tempo_diff_ms
                       for td in ab.sequence_metrics.tempo_differentials):
                return False

        if q.min_tunnel_distance_m is not None and ab.sequence_metrics:
            if not any(t.tunnel_distance_m >= q.min_tunnel_distance_m
                       for t in ab.sequence_metrics.tunnel_analyses):
                return False

        if q.custom_filter is not None and not q.custom_filter(ab):
            return False

        return True
