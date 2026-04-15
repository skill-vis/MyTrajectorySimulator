"""
SequenceVisualizer — visualization for pitch sequences and dynamic metrics.

Charts:
    1. At-bat strike-zone chart with sequence arrows
    2. Tunnel overlay (two trajectories with divergence point)
    3. Tempo differential box plot
    4. NR timing timeline
    5. Reaction mismatch scatter
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .models import (
    AtBat,
    Pitch,
    SequenceMetrics,
    TunnelAnalysis,
)

# ---------------------------------------------------------------------------
# Pitch type color map (MLB standard)
# ---------------------------------------------------------------------------

PITCH_COLORS: Dict[str, Tuple[str, str]] = {
    "FF": ("#d62728", "4-Seam"),
    "SI": ("#ff7f0e", "Sinker"),
    "FC": ("#8c564b", "Cutter"),
    "SL": ("#2ca02c", "Slider"),
    "CU": ("#1f77b4", "Curveball"),
    "CH": ("#9467bd", "Changeup"),
    "FS": ("#e377c2", "Splitter"),
    "FO": ("#e377c2", "Forkball"),
    "ST": ("#17becf", "Sweeper"),
    "KC": ("#aec7e8", "Knuckle-CV"),
    "SV": ("#98df8a", "Slurve"),
}

FT_TO_M = 0.3048
_DEFAULT_SZ_TOP_M = 3.5 * FT_TO_M   # ~1.067 m
_DEFAULT_SZ_BOT_M = 1.5 * FT_TO_M   # ~0.457 m
_ZONE_HW_M = 0.708 * FT_TO_M        # half-width ~0.216 m


def _pitch_color(pitch_type: str) -> str:
    return PITCH_COLORS.get(pitch_type, ("#999999", pitch_type))[0]


def _pitch_label(pitch_type: str) -> str:
    return PITCH_COLORS.get(pitch_type, ("#999999", pitch_type))[1]


# ---------------------------------------------------------------------------
# Strike zone drawing
# ---------------------------------------------------------------------------

def draw_strike_zone(
    ax: plt.Axes,
    sz_top: float = _DEFAULT_SZ_TOP_M,
    sz_bot: float = _DEFAULT_SZ_BOT_M,
) -> None:
    """Draw strike zone rectangle and home plate outline."""
    hw = _ZONE_HW_M
    rect = patches.Rectangle(
        (-hw, sz_bot), 2 * hw, sz_top - sz_bot,
        linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)
    # Home plate polygon
    plate_x = [0, hw, hw, -hw, -hw, 0]
    plate_y = np.array([0, 0.35, 0.71, 0.71, 0.35, 0]) * FT_TO_M - 0.15
    ax.plot(plate_x, plate_y, "k-", linewidth=1)


# ---------------------------------------------------------------------------
# SequenceVisualizer
# ---------------------------------------------------------------------------

class SequenceVisualizer:
    """Visualization for pitch sequences and dynamic metrics."""

    # ------------------------------------------------------------------
    # 1. At-bat strike-zone chart
    # ------------------------------------------------------------------

    def plot_at_bat_chart(
        self,
        ab: AtBat,
        ax: Optional[plt.Axes] = None,
        show_zone: bool = True,
        show_arrows: bool = True,
        show_count: bool = True,
    ) -> plt.Figure:
        """
        Plot all pitches in an at-bat on the strike zone.

        - Marker color = pitch type
        - Pitch number annotated
        - Arrows connect consecutive pitches (movement vectors)
        - Count displayed per pitch
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        else:
            fig = ax.figure

        # Strike zone
        sz_top = _DEFAULT_SZ_TOP_M
        sz_bot = _DEFAULT_SZ_BOT_M
        for p in ab.pitches:
            if p.sz_top is not None:
                sz_top = p.sz_top
                break
        for p in ab.pitches:
            if p.sz_bot is not None:
                sz_bot = p.sz_bot
                break

        if show_zone:
            draw_strike_zone(ax, sz_top, sz_bot)

        xs, zs = [], []
        for i, p in enumerate(ab.pitches):
            if p.plate_x is None or p.plate_z is None:
                continue

            color = _pitch_color(p.pitch_type)
            marker = "o" if not p.is_whiff else "X"
            size = 120

            ax.scatter(p.plate_x, p.plate_z, c=color, s=size,
                       marker=marker, edgecolors="white", linewidths=0.5,
                       zorder=10 + i)

            # Pitch number
            ax.annotate(
                str(i + 1),
                (p.plate_x, p.plate_z),
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=20 + i,
            )

            # Count label
            if show_count:
                label = f"{p.balls}-{p.strikes} {p.pitch_type}"
                ax.annotate(
                    label,
                    (p.plate_x, p.plate_z),
                    xytext=(8, -8), textcoords="offset points",
                    fontsize=6, color=color, alpha=0.8,
                )

            xs.append(p.plate_x)
            zs.append(p.plate_z)

        # Arrows between consecutive pitches
        if show_arrows and len(xs) >= 2:
            for i in range(len(xs) - 1):
                ax.annotate(
                    "", xy=(xs[i + 1], zs[i + 1]), xytext=(xs[i], zs[i]),
                    arrowprops=dict(arrowstyle="->", color="#666666",
                                    lw=1.0, alpha=0.5),
                    zorder=5,
                )

        # Result annotation
        result_str = ab.result.value if ab.result else ""
        ax.set_title(
            f"vs {ab.batter_name} ({ab.stand})  [{result_str}]",
            fontsize=10, fontweight="bold",
        )

        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0.0, 1.4)
        ax.set_xlabel("Horizontal (m)")
        ax.set_ylabel("Vertical (m)")
        ax.grid(True, alpha=0.2)

        return fig

    # ------------------------------------------------------------------
    # 2. Tunnel overlay
    # ------------------------------------------------------------------

    def plot_tunnel_overlay(
        self,
        pitch_a: Pitch,
        pitch_b: Pitch,
        tunnel: TunnelAnalysis,
        view: str = "side",
    ) -> plt.Figure:
        """
        Overlay two simulated trajectories highlighting the tunnel point.

        view: "side" (y-z), "top" (x-y), "catcher" (x-z)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if not pitch_a.sim_result or not pitch_b.sim_result:
            ax.text(0.5, 0.5, "No simulation data", ha="center", va="center")
            return fig

        traj_a = pitch_a.sim_result.trajectory
        traj_b = pitch_b.sim_result.trajectory

        # Offset for release alignment
        offset = (0.0, 0.0, 0.0)
        if pitch_a.sim_params and pitch_b.sim_params:
            offset = (
                pitch_a.sim_params.x0 - pitch_b.sim_params.x0,
                pitch_a.sim_params.y0 - pitch_b.sim_params.y0,
                pitch_a.sim_params.z0 - pitch_b.sim_params.z0,
            )

        # Extract coordinates
        def get_coords(traj, off=(0, 0, 0)):
            return (
                [p["x"] + off[0] for p in traj],
                [p["y"] + off[1] for p in traj],
                [p["z"] + off[2] for p in traj],
                [p["t"] for p in traj],
            )

        xa, ya, za, ta = get_coords(traj_a)
        xb, yb, zb, tb = get_coords(traj_b, offset)

        color_a = _pitch_color(pitch_a.pitch_type)
        color_b = _pitch_color(pitch_b.pitch_type)

        # Select axes based on view
        if view == "side":
            ax_data_a, ay_data_a = ya, za
            ax_data_b, ay_data_b = yb, zb
            xlabel, ylabel = "Y - Distance to plate (m)", "Z - Height (m)"
            tp_x = tunnel.tunnel_point_a[1]
            tp_y_a = tunnel.tunnel_point_a[2]
            tp_y_b = tunnel.tunnel_point_b[2]
        elif view == "top":
            ax_data_a, ay_data_a = ya, xa
            ax_data_b, ay_data_b = yb, xb
            xlabel, ylabel = "Y - Distance to plate (m)", "X - Horizontal (m)"
            tp_x = tunnel.tunnel_point_a[1]
            tp_y_a = tunnel.tunnel_point_a[0]
            tp_y_b = tunnel.tunnel_point_b[0]
        else:  # "catcher"
            ax_data_a, ay_data_a = xa, za
            ax_data_b, ay_data_b = xb, zb
            xlabel, ylabel = "X - Horizontal (m)", "Z - Height (m)"
            tp_x = tunnel.tunnel_point_a[0]
            tp_y_a = tunnel.tunnel_point_a[2]
            tp_y_b = tunnel.tunnel_point_b[2]

        # Find tunnel split index
        tunnel_idx_a = 0
        for i, t in enumerate(ta):
            if t >= tunnel.tunnel_time_s:
                tunnel_idx_a = i
                break

        tunnel_idx_b = 0
        for i, t in enumerate(tb):
            if t >= tunnel.tunnel_time_s:
                tunnel_idx_b = i
                break

        # Plot tunnel (shared) portion
        ax.plot(ax_data_a[:tunnel_idx_a + 1], ay_data_a[:tunnel_idx_a + 1],
                color=color_a, linewidth=2, label=f"{_pitch_label(pitch_a.pitch_type)}")
        ax.plot(ax_data_b[:tunnel_idx_b + 1], ay_data_b[:tunnel_idx_b + 1],
                color=color_b, linewidth=2, label=f"{_pitch_label(pitch_b.pitch_type)}")

        # Diverging portion (dashed)
        ax.plot(ax_data_a[tunnel_idx_a:], ay_data_a[tunnel_idx_a:],
                color=color_a, linewidth=2, linestyle="--", alpha=0.6)
        ax.plot(ax_data_b[tunnel_idx_b:], ay_data_b[tunnel_idx_b:],
                color=color_b, linewidth=2, linestyle="--", alpha=0.6)

        # Tunnel point marker
        ax.scatter([tp_x], [tp_y_a], c="black", s=80, zorder=20,
                   marker="D", edgecolors="white", linewidths=1)
        ax.annotate(
            f"Tunnel: {tunnel.tunnel_distance_m:.1f}m\n"
            f"t={tunnel.tunnel_time_s*1000:.0f}ms",
            (tp_x, tp_y_a), xytext=(10, 10), textcoords="offset points",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Tunnel: {pitch_a.pitch_type} vs {pitch_b.pitch_type}  "
            f"(divergence={tunnel.tunnel_distance_m:.2f}m, "
            f"plate sep={tunnel.plate_separation_m*100:.1f}cm)",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        if view == "catcher":
            ax.set_aspect("equal")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Tempo differential box plot
    # ------------------------------------------------------------------

    def plot_tempo_boxplot(
        self,
        at_bats: List[AtBat],
        group_by: str = "pitch_type_pair",
    ) -> plt.Figure:
        """Box plot of tempo differentials grouped by pitch-type pair."""
        data: Dict[str, list] = {}

        for ab in at_bats:
            if not ab.sequence_metrics:
                continue
            for td in ab.sequence_metrics.tempo_differentials:
                key = f"{td.pitch_a_type}->{td.pitch_b_type}"
                data.setdefault(key, []).append(td.differential_ms)

        if not data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No tempo data", ha="center", va="center")
            return fig

        # Sort by median
        sorted_keys = sorted(data.keys(),
                             key=lambda k: np.median(data[k]), reverse=True)
        sorted_data = [data[k] for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(max(6, len(sorted_keys) * 0.8), 5))
        bp = ax.boxplot(sorted_data, labels=sorted_keys, patch_artist=True)

        for patch, key in zip(bp["boxes"], sorted_keys):
            from_type = key.split("->")[0]
            color = _pitch_color(from_type)
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Tempo Differential (ms)")
        ax.set_xlabel("Pitch Pair")
        ax.set_title("Tempo Differential by Pitch-Type Pair")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. NR timing timeline
    # ------------------------------------------------------------------

    def plot_nr_timeline(self, ab: AtBat) -> plt.Figure:
        """
        Timeline for one at-bat (HawkEye only):
        - Release → arrival time per pitch
        - Tunnel divergence point
        - NR and impact markers
        - Decision window shading
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        if not ab.sequence_metrics:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
            return fig

        sm = ab.sequence_metrics
        y_positions = list(range(len(ab.pitches)))

        for i, p in enumerate(ab.pitches):
            y = i
            color = _pitch_color(p.pitch_type)

            if p.sim_result:
                arrival = p.sim_result.arrival_time_s * 1000  # ms

                # Flight bar (release to arrival)
                ax.barh(y, arrival, height=0.3, left=0,
                        color=color, alpha=0.3, edgecolor=color)

                # Decision window shading
                dw_start = 200  # HUMAN_REACTION_S * 1000
                if arrival > dw_start:
                    ax.barh(y, arrival - dw_start, height=0.3, left=dw_start,
                            color=color, alpha=0.15, hatch="//")

                ax.text(arrival + 2, y, f"{arrival:.0f}ms",
                        va="center", fontsize=7, color=color)

            # NR marker
            nr = next((n for n in sm.nr_analyses if n.pitch_idx == i), None)
            if nr and p.hawkeye and p.hawkeye.release_time is not None:
                nr_rel_ms = (nr.grip_max_time_s - p.hawkeye.release_time) * 1000
                ax.plot(nr_rel_ms, y, "v", color="red", markersize=8, zorder=15)
                ax.annotate("NR", (nr_rel_ms, y), xytext=(0, -12),
                            textcoords="offset points", fontsize=6,
                            ha="center", color="red")

                if nr.impact_time_s is not None:
                    impact_rel_ms = (nr.impact_time_s - p.hawkeye.release_time) * 1000
                    ax.plot(impact_rel_ms, y, "s", color="black",
                            markersize=6, zorder=15)

            # Label
            count_str = f"{p.balls}-{p.strikes}"
            result_str = p.description.value[:6] if p.description else ""
            ax.text(-5, y, f"#{i+1} {p.pitch_type} ({count_str}) {result_str}",
                    va="center", ha="right", fontsize=7)

        # Tunnel points
        for t in sm.tunnel_analyses:
            tunnel_ms = t.tunnel_time_s * 1000
            y_mid = (t.pitch_a_idx + t.pitch_b_idx) / 2
            ax.plot(tunnel_ms, y_mid, "D", color="goldenrod",
                    markersize=6, zorder=20)

        ax.set_xlabel("Time from release (ms)")
        ax.set_yticks([])
        ax.set_title(
            f"NR Timeline: {ab.pitcher_name} vs {ab.batter_name} "
            f"[{ab.result.value if ab.result else ''}]",
            fontsize=10,
        )
        ax.set_xlim(-80, 550)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 5. Reaction mismatch scatter
    # ------------------------------------------------------------------

    def plot_reaction_scatter(self, at_bats: List[AtBat]) -> plt.Figure:
        """
        Scatter: x = timing_mismatch_ms, y = result type.
        Color by pitch-type pair.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for ab in at_bats:
            if not ab.sequence_metrics:
                continue
            for rm in ab.sequence_metrics.reaction_mismatches:
                pb = ab.pitches[rm.pitch_b_idx]
                if pb.description is None:
                    continue

                # Y mapping
                if pb.is_whiff:
                    y_val = 2
                    y_label = "Whiff"
                elif pb.description.value.startswith("foul"):
                    y_val = 1
                    y_label = "Foul"
                elif pb.description.value.startswith("hit_into_play"):
                    y_val = 0
                    y_label = "In Play"
                elif pb.description == PitchResult.CALLED_STRIKE:
                    y_val = 1.5
                    y_label = "Called K"
                elif pb.description == PitchResult.BALL:
                    y_val = -0.5
                    y_label = "Ball"
                else:
                    continue

                pair_key = f"{ab.pitches[rm.pitch_a_idx].pitch_type}->{pb.pitch_type}"
                color = _pitch_color(pb.pitch_type)

                ax.scatter(rm.timing_mismatch_ms, y_val + np.random.uniform(-0.15, 0.15),
                           c=color, alpha=0.5, s=30, edgecolors="white", linewidths=0.3)

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_yticks([-0.5, 0, 1, 1.5, 2])
        ax.set_yticklabels(["Ball", "In Play", "Foul", "Called K", "Whiff"])
        ax.set_xlabel("Timing Mismatch (ms)")
        ax.set_ylabel("Pitch B Result")
        ax.set_title("Reaction Mismatch vs Outcome")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig
