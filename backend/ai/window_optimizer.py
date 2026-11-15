from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Dict, List, Optional

import optuna

from backend.physics.ephemeris import LunarEphemerisEngine
from backend.physics.horizon import HorizonProfile
from backend.physics.simulation import LanderConfig, run_mission_simulation

# Suppress verbose trial logs in Uvicorn console
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationCandidate:
    rank: int
    site_id: str
    site_name: str
    lat_deg: float
    lon_deg: float
    recommended_start_utc: str
    offset_hours_from_base: float
    strict_status: str
    optimization_score: float
    min_battery_soc_pct: float
    final_battery_soc_pct: float
    mean_sun_visible_pct: float
    total_pv_generation_wh: float
    max_dte_blackout_hours: float
    dte_availability_pct: float
    data_label: str = "SIMULATED (AI-Guided Bayesian Optimization via Optuna)"


def _classify_status(
    min_soc: float,
    unmet_load_wh: float,
    max_blackout_h: float,
    cfg: LanderConfig,
    max_allowed_blackout_h: float,
) -> str:
    """Applies strict KiyoSpace feasibility criteria."""
    if unmet_load_wh > 1e-3 or min_soc <= cfg.min_soc + 1e-4:
        return "INFEASIBLE"
    if max_blackout_h > max_allowed_blackout_h:
        return "UNRESOLVED"
    if min_soc < cfg.min_soc + 0.05:
        return "UNRESOLVED"
    return "FEASIBLE"


def optimize_launch_windows(
    base_start_utc: str,
    search_window_days: int,
    duration_hours: float,
    step_minutes: float,
    candidate_sites: Dict[str, Dict[str, Any]],
    horizon_cache: Dict[str, HorizonProfile],
    ephemeris_engine: LunarEphemerisEngine,
    spice_lock: threading.Lock,
    lander_config: LanderConfig,
    max_allowed_blackout_hours: float = 24.0,
    n_trials: int = 40,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Uses Bayesian Optimization (Optuna TPE) over deterministic SPICE + LOLA Horizon
    physics simulations to identify the safest lunar landing windows and sites.
    """
    base_dt = datetime.fromisoformat(base_start_utc.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    max_offset_hours = max(6, search_window_days * 24)
    site_ids = list(candidate_sites.keys())

    evaluated_records: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    def evaluate_point(site_id: str, offset_h: int) -> float:
        key = (site_id, offset_h)
        site_meta = candidate_sites[site_id]
        horizon = horizon_cache[site_id]
        cand_dt = base_dt + timedelta(hours=offset_h)
        cand_utc = cand_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with spice_lock:
            series = ephemeris_engine.evaluate_series(
                start_utc=cand_utc,
                duration_hours=duration_hours,
                step_minutes=step_minutes,
                lat_deg=site_meta["lat_deg"],
                lon_deg=site_meta["lon_deg"],
                elev_m=horizon.ground_elevation_m,
            )

        summary = run_mission_simulation(
            ephemeris_series=series,
            horizon_profile=horizon,
            config=lander_config,
        )

        status = _classify_status(
            min_soc=summary.min_battery_soc,
            unmet_load_wh=summary.total_unmet_load_wh,
            max_blackout_h=summary.max_dte_blackout_hours,
            cfg=lander_config,
            max_allowed_blackout_h=max_allowed_blackout_hours,
        )

        # Multi-objective scalarization:
        # 1. Maximize minimum SOC margin above safe threshold
        # 2. Penalize unmet load heavily
        # 3. Penalize DTE blackout duration
        # 4. Reward average solar illumination
        soc_margin_pct = (summary.min_battery_soc - lander_config.min_soc) * 100.0
        unmet_penalty = (summary.total_unmet_load_wh / max(lander_config.battery_capacity_wh, 1.0)) * 200.0
        blackout_penalty = max(0.0, summary.max_dte_blackout_hours - max_allowed_blackout_hours) * 1.5
        sun_bonus = summary.mean_sun_visible_fraction * 25.0

        score = soc_margin_pct + sun_bonus - unmet_penalty - blackout_penalty

        if key not in seen_keys:
            seen_keys.add(key)
            evaluated_records.append(
                {
                    "site_id": site_id,
                    "site_name": site_meta["name"],
                    "lat_deg": site_meta["lat_deg"],
                    "lon_deg": site_meta["lon_deg"],
                    "cand_dt": cand_dt,
                    "recommended_start_utc": cand_utc,
                    "offset_hours_from_base": float(offset_h),
                    "strict_status": status,
                    "optimization_score": round(score, 2),
                    "min_battery_soc_pct": round(summary.min_battery_soc * 100.0, 2),
                    "final_battery_soc_pct": round(summary.final_battery_soc * 100.0, 2),
                    "mean_sun_visible_pct": round(summary.mean_sun_visible_fraction * 100.0, 2),
                    "total_pv_generation_wh": round(summary.total_pv_generation_wh, 1),
                    "max_dte_blackout_hours": round(summary.max_dte_blackout_hours, 2),
                    "dte_availability_pct": round(summary.dte_availability_pct, 2),
                    "data_label": "SIMULATED (AI-Guided Bayesian Optimization via Optuna)",
                }
            )

        return score

    def objective(trial: optuna.Trial) -> float:
        site_id = trial.suggest_categorical("site_id", site_ids)
        # Search in 6-hour increments across the window
        offset_h = trial.suggest_int("offset_hours", 0, max_offset_hours, step=6)
        return evaluate_point(site_id, offset_h)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Sort all evaluated unique windows from highest score to lowest
    evaluated_records.sort(key=lambda r: r["optimization_score"], reverse=True)

    # Deduplicate recommendations so top_k windows are separated by at least 24 hours (per site)
    distinct_top: List[Dict[str, Any]] = []
    for rec in evaluated_records:
        is_too_close = any(
            rec["site_id"] == chosen["site_id"]
            and abs((rec["cand_dt"] - chosen["cand_dt"]).total_seconds()) < 24 * 3600
            for chosen in distinct_top
        )
        if not is_too_close:
            distinct_top.append(rec)
        if len(distinct_top) >= top_k:
            break

    recommendations = []
    for idx, item in enumerate(distinct_top, start=1):
        clean_item = {k: v for k, v in item.items() if k != "cand_dt"}
        clean_item["rank"] = idx
        recommendations.append(clean_item)

    return {
        "search_space": {
            "base_start_utc": base_start_utc,
            "search_window_days": search_window_days,
            "candidate_sites": site_ids,
            "trials_executed": n_trials,
            "unique_windows_evaluated": len(evaluated_records),
        },
        "recommendations": recommendations,
        "provenance": {
            "optimizer": "Optuna TPE Bayesian Sampler",
            "physics_solver": "NAIF SPICE DE421/MOON_ME + LOLA 5m DEM Ray-Traced Horizon",
            "compliance_note": "All candidate scores are computed by deterministic physics simulations, not surrogate approximations.",
        },
    }