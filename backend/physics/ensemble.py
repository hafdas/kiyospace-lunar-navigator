from __future__ import annotations

import copy
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from backend.physics.ephemeris import EphemerisPoint
from backend.physics.horizon import HorizonProfile
from backend.physics.simulation import (
    LanderConfig,
    MissionSimulationSummary,
    run_mission_simulation,
)

@dataclass
class EnsembleResult:
    nominal_summary: MissionSimulationSummary
    worst_case_soc: float
    best_case_soc: float
    mean_soc: float
    soc_std_dev: float
    worst_case_blackout_h: float
    strict_feasibility_status: str
    convergence_passed: bool

def generate_synthetic_clone_profiles(
    base_profile: HorizonProfile, num_clones: int = 10, error_margin_deg: float = 0.45
) -> List[HorizonProfile]:
    """
    Simulates horizon variations derived from LOLA terrain clones (z_j = z_nominal + delta_z_j).
    In a full offline pipeline, this reads the actual *clone_001.tif to *clone_010.tif files.
    """
    clones = []
    np.random.seed(42)  # Deterministic seed for reproducibility
    
    for _ in range(num_clones):
        clone = copy.deepcopy(base_profile)
        # Apply a smooth random walk variation to simulate correlated terrain elevation errors
        noise = np.random.normal(0, error_margin_deg / 3.0, len(clone.horizon_elevations_deg))
        smoothed_noise = np.convolve(noise, np.ones(5)/5.0, mode='same')
        clone.horizon_elevations_deg += smoothed_noise
        clones.append(clone)
        
    return clones

def run_ensemble_evaluation(
    ephemeris_series: List[EphemerisPoint],
    nominal_horizon: HorizonProfile,
    config: LanderConfig,
    num_clones: int = 10
) -> EnsembleResult:
    """
    Runs the mission simulation across the nominal DEM and all terrain clones to 
    determine the strict 3-state feasibility status.
    """
    # 1. Run Nominal Simulation
    nominal_sim = run_mission_simulation(ephemeris_series, nominal_horizon, config)
    
    # 2. Generate and Run Terrain Clones
    clones = generate_synthetic_clone_profiles(nominal_horizon, num_clones)
    clone_sims: List[MissionSimulationSummary] = []
    
    for clone_prof in clones:
        clone_sims.append(run_mission_simulation(ephemeris_series, clone_prof, config))
        
    # 3. Aggregate Statistical Uncertainty
    all_min_socs = [sim.min_battery_soc for sim in clone_sims]
    all_blackouts = [sim.max_dte_blackout_hours for sim in clone_sims]
    
    worst_soc = float(np.min(all_min_socs))
    best_soc = float(np.max(all_min_socs))
    mean_soc = float(np.mean(all_min_socs))
    std_soc = float(np.std(all_min_socs))
    worst_blackout = float(np.max(all_blackouts))
    
    # 4. Strict 3-State Logic & Convergence Criteria
    # FEASIBLE: Nominal AND Worst-case clones survive above min_soc
    # UNRESOLVED: Nominal survives, but worst-case clone fails (Uncertainty overlap)
    # INFEASIBLE: Nominal fails outright
    
    if nominal_sim.min_battery_soc <= config.min_soc or nominal_sim.total_unmet_load_wh > 0:
        status = "INFEASIBLE"
    elif worst_soc <= config.min_soc or worst_blackout > 24.0:
        status = "UNRESOLVED"
    else:
        status = "FEASIBLE"
        
    # Convergence Check: Energy balance residual must be <= 0.1% across ALL runs
    convergence_passed = all(sim.energy_balance_residual_pct <= 0.1 for sim in clone_sims)
    
    return EnsembleResult(
        nominal_summary=nominal_sim,
        worst_case_soc=worst_soc,
        best_case_soc=best_soc,
        mean_soc=mean_soc,
        soc_std_dev=std_soc,
        worst_case_blackout_h=worst_blackout,
        strict_feasibility_status=status,
        convergence_passed=convergence_passed
    )