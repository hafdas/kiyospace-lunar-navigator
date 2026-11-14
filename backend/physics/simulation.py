from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from backend.physics.ephemeris import EphemerisPoint
from backend.physics.horizon import HorizonProfile

# Konstanta fluks surya rata-rata pada jarak 1 AU (W/m^2)
SOLAR_CONSTANT_WM2 = 1361.0


@dataclass
class LanderConfig:
    """Parameter konfigurasi wahana pendarat (lander) yang dapat diatur pengguna."""
    pv_area_m2: float = 4.0                 # Luas efektif panel surya (m^2)
    pv_efficiency: float = 0.28             # Efisiensi sel surya (mis. triple-junction GaAs 28%)
    pv_tracking: bool = True                # True = panel mengikuti azimuth Matahari (sun-tracking)
    pv_fixed_azimuth_deg: float = 0.0       # Dipakai jika pv_tracking = False
    pv_tilt_deg: float = 85.0               # Kemiringan panel dari bidang horizontal (vertikal di kutub)
    battery_capacity_wh: float = 5000.0     # Kapasitas total baterai (Wh)
    initial_soc: float = 0.80               # Status pengisian awal (0.0 - 1.0)
    min_soc: float = 0.20                   # Batas aman pengosongan minimum baterai (DoD 80%)
    charge_efficiency: float = 0.92         # Efisiensi pengisian baterai
    discharge_efficiency: float = 0.92      # Efisiensi pengosongan baterai
    nominal_load_w: float = 250.0           # Beban daya rata-rata wahana (W)
    min_dte_elevation_margin_deg: float = 2.0  # Margin sudut minimum di atas horizon untuk DTE


@dataclass
class MissionStepResult:
    utc: str
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_horizon_deg: float
    sun_visible_fraction: float
    pv_power_w: float
    load_power_w: float
    battery_energy_wh: float
    battery_soc: float
    curtailment_wh: float
    unmet_load_wh: float
    earth_azimuth_deg: float
    earth_elevation_deg: float
    earth_horizon_deg: float
    dte_margin_deg: float
    dte_link_active: bool


@dataclass
class MissionSimulationSummary:
    duration_hours: float
    step_minutes: float
    mean_sun_visible_fraction: float
    total_pv_generation_wh: float
    total_load_demand_wh: float
    total_curtailment_wh: float
    total_unmet_load_wh: float
    min_battery_soc: float
    final_battery_soc: float
    energy_balance_residual_pct: float
    dte_availability_pct: float
    max_dte_blackout_hours: float
    earthrise_timestamps: List[str] = field(default_factory=list)
    earthset_timestamps: List[str] = field(default_factory=list)
    steps: List[MissionStepResult] = field(default_factory=list)


def interpolate_horizon_at_azimuth(
    horizon_profile: HorizonProfile, azimuth_deg: float
) -> float:
    """Interpolasi linear siklik (0°-360°) untuk sudut horizon pada azimuth tertentu."""
    az = azimuth_deg % 360.0
    az_grid = np.append(horizon_profile.azimuths_deg, 360.0)
    hor_grid = np.append(
        horizon_profile.horizon_elevations_deg,
        horizon_profile.horizon_elevations_deg[0],
    )
    return float(np.interp(az, az_grid, hor_grid))


def compute_solar_visible_fraction(
    sun_el_deg: float,
    horizon_el_deg: float,
    sun_angular_radius_deg: float,
    n_quadrature: int = 64,
) -> float:
    """
    Menghitung fraksi luas piringan Matahari berhingga V(t) in [0, 1] yang tampak
    di atas garis horizon lokal menggunakan kuadratur irisan horizontal piringan.
    """
    r = max(sun_angular_radius_deg, 1e-6)
    # Seluruh piringan berada di atas horizon
    if sun_el_deg - r >= horizon_el_deg:
        return 1.0
    # Seluruh piringan terbenam di bawah horizon
    if sun_el_deg + r <= horizon_el_deg:
        return 0.0

    # Integrasi numerik irisan piringan lingkaran berjari-jari r
    # y diukur dari pusat Matahari (-r hingga +r)
    y = np.linspace(-r, r, n_quadrature)
    dy = y[1] - y[0]
    chord_widths = 2.0 * np.sqrt(np.maximum(0.0, r**2 - y**2))
    total_area = np.sum(chord_widths) * dy

    visible_mask = (sun_el_deg + y) > horizon_el_deg
    visible_area = np.sum(chord_widths[visible_mask]) * dy

    return float(np.clip(visible_area / total_area, 0.0, 1.0))


def compute_pv_power_w(
    sun_az_deg: float,
    sun_el_deg: float,
    visible_fraction: float,
    config: LanderConfig,
) -> float:
    """Menghitung daya listrik panel surya P_PV(t) dalam Watt."""
    if visible_fraction <= 0.0 or sun_el_deg <= -1.0:
        return 0.0

    el_rad = np.radians(max(0.0, sun_el_deg))
    tilt_rad = np.radians(config.pv_tilt_deg)

    if config.pv_tracking:
        d_az_rad = 0.0
    else:
        d_az_rad = np.radians(sun_az_deg - config.pv_fixed_azimuth_deg)

    # Faktor kosinus sudut datang sinar Matahari pada bidang panel miring
    cos_incidence = (
        np.sin(el_rad) * np.cos(tilt_rad)
        + np.cos(el_rad) * np.sin(tilt_rad) * np.cos(d_az_rad)
    )
    cos_incidence = max(0.0, float(cos_incidence))

    return (
        SOLAR_CONSTANT_WM2
        * config.pv_area_m2
        * config.pv_efficiency
        * visible_fraction
        * cos_incidence
    )


def run_mission_simulation(
    ephemeris_series: List[EphemerisPoint],
    horizon_profile: HorizonProfile,
    config: LanderConfig | None = None,
) -> MissionSimulationSummary:
    """
    Menjalankan simulasi fisika misi menyeluruh (pencahayaan, PV, baterai, DTE)
    selama deret waktu 48–168 jam dan memverifikasi kekekalan energi baterai.
    """
    if len(ephemeris_series) < 2:
        raise ValueError("Deret waktu efemeris minimal berisi 2 titik waktu.")

    cfg = config if config is not None else LanderConfig()
    dt_hours = (ephemeris_series[1].et - ephemeris_series[0].et) / 3600.0
    if dt_hours <= 0:
        raise ValueError("Langkah waktu efemeris harus bernilai positif.")

    e_max = cfg.battery_capacity_wh
    e_min = cfg.min_soc * e_max
    e_curr = float(np.clip(cfg.initial_soc * e_max, e_min, e_max))
    e_initial = e_curr

    total_pv_wh = 0.0
    total_load_wh = 0.0
    total_curtail_wh = 0.0
    total_unmet_wh = 0.0
    total_charge_loss_wh = 0.0
    total_discharge_loss_wh = 0.0

    min_soc_seen = e_curr / e_max
    step_results: List[MissionStepResult] = []

    earthrise_times: List[str] = []
    earthset_times: List[str] = []
    prev_dte_active: bool | None = None
    current_blackout_hours = 0.0
    max_blackout_hours = 0.0
    dte_active_steps = 0

    for pt in ephemeris_series:
        # 1. Pencahayaan Matahari
        sun_hor = interpolate_horizon_at_azimuth(horizon_profile, pt.sun_azimuth_deg)
        vis_frac = compute_solar_visible_fraction(
            sun_el_deg=pt.sun_elevation_deg,
            horizon_el_deg=sun_hor,
            sun_angular_radius_deg=pt.sun_angular_radius_deg,
        )

        # 2. Pembangkitan PV & Neraca Energi Baterai
        p_pv = compute_pv_power_w(
            sun_az_deg=pt.sun_azimuth_deg,
            sun_el_deg=pt.sun_elevation_deg,
            visible_fraction=vis_frac,
            config=cfg,
        )
        p_load = cfg.nominal_load_w
        e_pv_step = p_pv * dt_hours
        e_load_step = p_load * dt_hours

        curtail_step = 0.0
        unmet_step = 0.0

        net_wh = e_pv_step - e_load_step
        if net_wh >= 0.0:
            # Surplus daya mengisi baterai
            stored_gain = net_wh * cfg.charge_efficiency
            room_available = e_max - e_curr
            if stored_gain <= room_available:
                e_curr += stored_gain
                total_charge_loss_wh += net_wh * (1.0 - cfg.charge_efficiency)
            else:
                accepted_input = room_available / cfg.charge_efficiency
                curtail_step = net_wh - accepted_input
                total_charge_loss_wh += accepted_input * (1.0 - cfg.charge_efficiency)
                e_curr = e_max
        else:
            # Defisit daya diambil dari baterai
            deficit_wh = -net_wh
            required_draw = deficit_wh / cfg.discharge_efficiency
            available_draw = e_curr - e_min
            if required_draw <= available_draw:
                e_curr -= required_draw
                total_discharge_loss_wh += required_draw - deficit_wh
            else:
                delivered_wh = available_draw * cfg.discharge_efficiency
                unmet_step = deficit_wh - delivered_wh
                total_discharge_loss_wh += available_draw - delivered_wh
                e_curr = e_min

        total_pv_wh += e_pv_step
        total_load_wh += e_load_step
        total_curtail_wh += curtail_step
        total_unmet_wh += unmet_step

        soc_curr = e_curr / e_max
        min_soc_seen = min(min_soc_seen, soc_curr)

        # 3. Komunikasi Direct-to-Earth (DTE) L1
        earth_hor = interpolate_horizon_at_azimuth(horizon_profile, pt.earth_azimuth_deg)
        dte_margin = pt.earth_elevation_deg - earth_hor
        dte_active = dte_margin >= cfg.min_dte_elevation_margin_deg

        if dte_active:
            dte_active_steps += 1
            current_blackout_hours = 0.0
            if prev_dte_active is False:
                earthrise_times.append(pt.utc)
        else:
            current_blackout_hours += dt_hours
            max_blackout_hours = max(max_blackout_hours, current_blackout_hours)
            if prev_dte_active is True:
                earthset_times.append(pt.utc)

        prev_dte_active = dte_active

        step_results.append(
            MissionStepResult(
                utc=pt.utc,
                sun_azimuth_deg=pt.sun_azimuth_deg,
                sun_elevation_deg=pt.sun_elevation_deg,
                sun_horizon_deg=sun_hor,
                sun_visible_fraction=vis_frac,
                pv_power_w=p_pv,
                load_power_w=p_load,
                battery_energy_wh=e_curr,
                battery_soc=soc_curr,
                curtailment_wh=curtail_step,
                unmet_load_wh=unmet_step,
                earth_azimuth_deg=pt.earth_azimuth_deg,
                earth_elevation_deg=pt.earth_elevation_deg,
                earth_horizon_deg=earth_hor,
                dte_margin_deg=dte_margin,
                dte_link_active=dte_active,
            )
        )

    # Audit hukum kekekalan energi:
    # E_akhir - E_awal == Total_PV - Total_Load - Curtailment + Unmet_Load - Rugi_Efisiensi
    lhs_delta_e = e_curr - e_initial
    rhs_net_energy = (
        total_pv_wh
        - total_load_wh
        - total_curtail_wh
        + total_unmet_wh
        - total_charge_loss_wh
        - total_discharge_loss_wh
    )
    scale_ref = max(e_max, total_pv_wh, total_load_wh, 1.0)
    residual_pct = abs(lhs_delta_e - rhs_net_energy) / scale_ref * 100.0

    duration_hours = dt_hours * (len(ephemeris_series) - 1)
    mean_vis = float(np.mean([s.sun_visible_fraction for s in step_results]))
    dte_avail_pct = (dte_active_steps / len(step_results)) * 100.0

    return MissionSimulationSummary(
        duration_hours=duration_hours,
        step_minutes=dt_hours * 60.0,
        mean_sun_visible_fraction=mean_vis,
        total_pv_generation_wh=total_pv_wh,
        total_load_demand_wh=total_load_wh,
        total_curtailment_wh=total_curtail_wh,
        total_unmet_load_wh=total_unmet_wh,
        min_battery_soc=min_soc_seen,
        final_battery_soc=e_curr / e_max,
        energy_balance_residual_pct=residual_pct,
        dte_availability_pct=dte_avail_pct,
        max_dte_blackout_hours=max_blackout_hours,
        earthrise_timestamps=earthrise_times,
        earthset_timestamps=earthset_times,
        steps=step_results,
    )


if __name__ == "__main__":
    from pathlib import Path
    from backend.physics.ephemeris import LunarEphemerisEngine
    from backend.physics.horizon import compute_horizon_from_geotiff

    site01_dem = Path("data/raw/site01/Site01_final_adj_5mpp_surf.tif")
    if site01_dem.exists():
        horizon = compute_horizon_from_geotiff(
            dem_path=site01_dem,
            lat_deg=-89.5,
            lon_deg=225.0,
            observer_height_m=1.5,
            azimuth_step_deg=1.0,
            max_radius_m=10_000.0,
        )
        ephem = LunarEphemerisEngine()
        series = ephem.evaluate_series(
            start_utc="2026-11-15T00:00:00Z",
            duration_hours=72.0,
            step_minutes=30.0,
            lat_deg=-89.5,
            lon_deg=225.0,
            elev_m=horizon.ground_elevation_m,
        )
        summary = run_mission_simulation(series, horizon)
        print(f"Simulasi Site01 (72 Jam) Selesai:")
        print(f"  Mean Sun Visible Fraction : {summary.mean_sun_visible_fraction * 100:.2f}%")
        print(f"  Total PV / Load           : {summary.total_pv_generation_wh:.1f} Wh / {summary.total_load_demand_wh:.1f} Wh")
        print(f"  SOC Minimum / Akhir       : {summary.min_battery_soc * 100:.1f}% / {summary.final_battery_soc * 100:.1f}%")
        print(f"  Residual Kekekalan Energi : {summary.energy_balance_residual_pct:.6f}% (Target <= 0.1%)")
        print(f"  Ketersediaan DTE          : {summary.dte_availability_pct:.1f}% (Max Blackout: {summary.max_dte_blackout_hours:.1f} jam)")