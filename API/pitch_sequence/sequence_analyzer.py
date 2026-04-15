"""
SequenceAnalyzer — orchestrate simulation and metric computation for at-bats.

Runs BallTrajectorySimulator2 for each pitch (caching results), then
computes all dynamic metrics via MetricsComputer.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from .metrics import MetricsComputer
from .models import (
    AtBat,
    Pitch,
    SequenceMetrics,
    SimParameters,
    SimResult,
)

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """Compute sequence-level metrics for at-bats."""

    def __init__(self, simulator_factory: Optional[Callable] = None):
        """
        Parameters
        ----------
        simulator_factory : callable, optional
            Returns a configured BallTrajectorySimulator2 instance.
            Default: RK4 integration + NATHAN_EXP lift model.
        """
        self._sim_factory = simulator_factory or _default_simulator_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_at_bat(self, ab: AtBat) -> SequenceMetrics:
        """
        Run full analysis for one at-bat.

        1. Ensure all pitches have sim_result (run simulator if needed).
        2. For each consecutive pair (i, i+1): compute tempo, tunnel,
           movement, reaction metrics.
        3. For HawkEye pitches: compute NR analysis.
        4. Return SequenceMetrics.
        """
        # Step 1: simulate all pitches
        for p in ab.pitches:
            self._ensure_sim_result(p)

        metrics = SequenceMetrics()

        # Step 2: consecutive-pair metrics
        for i in range(len(ab.pitches) - 1):
            pa = ab.pitches[i]
            pb = ab.pitches[i + 1]

            tempo = MetricsComputer.compute_tempo_differential(pa, pb, i, i + 1)
            if tempo:
                metrics.tempo_differentials.append(tempo)

            tunnel = MetricsComputer.compute_tunnel(pa, pb, i, i + 1)
            if tunnel:
                metrics.tunnel_analyses.append(tunnel)

            mv = MetricsComputer.compute_movement_vector(pa, pb, i, i + 1)
            if mv:
                metrics.movement_vectors.append(mv)

            react = MetricsComputer.compute_reaction_mismatch(pa, pb, i, i + 1)
            if react:
                metrics.reaction_mismatches.append(react)

        # Step 3: NR analysis (HawkEye only)
        for i, p in enumerate(ab.pitches):
            # Find the tunnel involving this pitch as pitch_b
            tunnel_for_pitch = None
            for t in metrics.tunnel_analyses:
                if t.pitch_b_idx == i:
                    tunnel_for_pitch = t
                    break

            nr = MetricsComputer.compute_nr_analysis(p, tunnel=tunnel_for_pitch)
            if nr:
                metrics.nr_analyses.append(nr)

        return metrics

    def analyze_game(self, at_bats: List[AtBat]) -> List[AtBat]:
        """Analyze all at-bats, populating sequence_metrics on each."""
        for ab in at_bats:
            ab.sequence_metrics = self.analyze_at_bat(ab)
        return at_bats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_sim_result(self, pitch: Pitch) -> None:
        """Run simulator if pitch.sim_result is None and sim_params exist."""
        if pitch.sim_result is not None:
            return
        if pitch.sim_params is None:
            return

        try:
            pitch.sim_result = self._run_simulation(pitch.sim_params)
        except Exception as e:
            logger.warning("Simulation failed for %s: %s", pitch.pitch_id, e)

    def _run_simulation(self, params: SimParameters) -> SimResult:
        """Execute trajectory simulation and wrap result."""
        from MyBallTrajectorySim_E import PitchParameters, EnvironmentParameters

        sim = self._sim_factory()
        pitch_params = PitchParameters(
            x0=params.x0,
            y0=params.y0,
            z0=params.z0,
            v0_mps=params.v0_mps,
            theta_deg=params.theta_deg,
            phi_deg=params.phi_deg,
            backspin_rpm=params.backspin_rpm,
            sidespin_rpm=params.sidespin_rpm,
            wg_rpm=params.wg_rpm,
            batter_hand=params.batter_hand,
        )
        env = EnvironmentParameters()

        trajectory = sim.simulate(pitch=pitch_params, env=env,
                                  max_time=1.0, save_interval=1)

        hp = sim.home_plate_crossing
        arrival_time = hp["t"] if hp else 0.5
        arrival_speed = hp["v"] if hp else 0.0

        return SimResult(
            trajectory=trajectory,
            home_plate_crossing=hp,
            arrival_time_s=arrival_time,
            arrival_speed_mps=arrival_speed,
        )


def _default_simulator_factory():
    """Create a default BallTrajectorySimulator2 with RK4 + NATHAN_EXP."""
    from MyBallTrajectorySim_E import (
        BallTrajectorySimulator2,
        IntegrationMethod,
        LiftModel,
    )
    return BallTrajectorySimulator2(
        integration_method=IntegrationMethod.RK4,
        lift_model=LiftModel.NATHAN_EXP,
    )
