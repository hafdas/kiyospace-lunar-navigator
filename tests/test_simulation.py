import numpy as np
from backend.physics.ephemeris import EphemerisPoint
from backend.physics.horizon import HorizonProfile
from backend.physics.simulation import (
    LanderConfig,
    compute_solar_visible_fraction,
    run_mission_simulation,
)


def test_finite_solar_disk_visible_fraction():
    """Memastikan integrasi piringan Matahari berhingga menghasilkan 0, 0.5, dan 1 secara akurat."""
    r_sun = 0.265
    assert compute_solar_visible_fraction(sun_el_deg=1.0, horizon_el_deg=0.0, sun_angular_radius_deg=r_sun) == 1.0
    assert compute_solar_visible_fraction(sun_el_deg=-1.0, horizon_el_deg=0.0, sun_angular_radius_deg=r_sun) == 0.0
    half_vis = compute_solar_visible_fraction(sun_el_deg=0.5, horizon_el_deg=0.5, sun_angular_radius_deg=r_sun)
    assert np.isclose(half_vis, 0.5, atol=0.01)


def test_battery_energy_conservation_and_dte_tracking():
    """
    Memastikan residual kekekalan energi baterai <= 0.1% dan deteksi Earthrise/Earthset
    serta durasi blackout DTE bekerja sesuai hukum fisika.
    """
    az = np.arange(0.0, 360.0, 1.0)
    hor = np.zeros_like(az)  # Horizon datar 0°
    profile = HorizonProfile(
        azimuths_deg=az,
        horizon_elevations_deg=hor,
        observer_lat_deg=-89.5,
        observer_lon_deg=225.0,
        observer_height_m=1.5,
        ground_elevation_m=1000.0,
    )

    # Buat 10 titik waktu (masing-masing selang 1 jam):
    # Jam 0-4: Matahari di +2° (terang), Bumi di +5° (DTE aktif)
    # Jam 5-9: Matahari di -2° (gelap), Bumi di -1° (DTE blackout)
    series = []
    for h in range(10):
        day_phase = h < 5
        series.append(
            EphemerisPoint(
                utc=f"2026-11-15T0{h}:00:00Z",
                et=float(h * 3600),
                sun_azimuth_deg=90.0,
                sun_elevation_deg=2.0 if day_phase else -2.0,
                sun_distance_km=149_600_000.0,
                sun_angular_radius_deg=0.266,
                earth_azimuth_deg=180.0,
                earth_elevation_deg=5.0 if day_phase else -1.0,
                earth_distance_km=384_400.0,
            )
        )

    cfg = LanderConfig(
        pv_area_m2=5.0,
        battery_capacity_wh=4000.0,
        initial_soc=0.50,
        nominal_load_w=300.0,
        min_dte_elevation_margin_deg=2.0,
    )
    summary = run_mission_simulation(series, profile, cfg)

    assert summary.energy_balance_residual_pct <= 0.1
    assert summary.total_pv_generation_wh > 0.0
    assert summary.max_dte_blackout_hours == 5.0
    assert summary.earthset_timestamps == ["2026-11-15T05:00:00Z"]