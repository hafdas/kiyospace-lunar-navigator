import numpy as np
import pytest
from backend.physics.ephemeris import LunarEphemerisEngine


@pytest.fixture(scope="module")
def ephemeris_engine():
    return LunarEphemerisEngine(kernel_dir="data/raw/spice")


def test_enu_basis_orthogonality():
    """Memastikan vektor basis East, North, Up saling tegak lurus dan satuan."""
    _, e_hat, n_hat, u_hat = LunarEphemerisEngine._enu_basis_and_position(
        lat_deg=-89.5, lon_deg=225.0, elev_m=1000.0
    )
    assert np.isclose(np.linalg.norm(e_hat), 1.0)
    assert np.isclose(np.linalg.norm(n_hat), 1.0)
    assert np.isclose(np.linalg.norm(u_hat), 1.0)
    assert np.isclose(np.dot(e_hat, n_hat), 0.0, atol=1e-12)
    assert np.isclose(np.dot(e_hat, u_hat), 0.0, atol=1e-12)
    assert np.isclose(np.dot(n_hat, u_hat), 0.0, atol=1e-12)


def test_south_pole_physical_bounds(ephemeris_engine):
    """
    Memastikan hasil perhitungan di Site01 (-89.5°, 225°) berada dalam rentang
    fisika astronomi yang valid (elevasi Matahari di kutub selamanya rendah: -3° s.d. +3°).
    """
    pt = ephemeris_engine.evaluate_single(
        utc_str="2026-11-15T00:00:00Z",
        lat_deg=-89.5,
        lon_deg=225.0,
        elev_m=1200.0,
    )
    assert 0.0 <= pt.sun_azimuth_deg < 360.0
    assert -3.5 <= pt.sun_elevation_deg <= 3.5
    assert 0.25 <= pt.sun_angular_radius_deg <= 0.28
    assert 0.0 <= pt.earth_azimuth_deg < 360.0
    assert -10.0 <= pt.earth_elevation_deg <= 10.0


def test_time_series_length(ephemeris_engine):
    """Memastikan generator deret waktu 48 jam dengan langkah 60 menit menghasilkan 49 titik."""
    series = ephemeris_engine.evaluate_series(
        start_utc="2026-11-15T00:00:00Z",
        duration_hours=48.0,
        step_minutes=60.0,
        lat_deg=-89.5,
        lon_deg=225.0,
    )
    assert len(series) == 49