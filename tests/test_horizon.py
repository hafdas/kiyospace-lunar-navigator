import numpy as np
from backend.physics.horizon import compute_horizon_from_grid, MOON_RADIUS_M


def test_flat_lunar_sphere_horizon():
    """
    On a perfectly flat plain (elevation 0 m) with observer height h = 1.5 m,
    the theoretical horizon dip is -arccos(R / (R + h)) ~ -0.0753°.
    """
    grid_size = 401
    pixel_m = 5.0
    dem = np.zeros((grid_size, grid_size), dtype=float)
    center = grid_size // 2

    az, hor, ground_z = compute_horizon_from_grid(
        dem_m=dem,
        pixel_size_m=pixel_m,
        obs_row=center,
        obs_col=center,
        observer_height_m=1.5,
        azimuth_step_deg=5.0,
        max_radius_m=950.0,
    )

    assert ground_z == 0.0
    assert len(az) == 72
    # At 950 m distance and 1.5 m height, the horizon angle should be negative (below horizontal).
    assert np.all(hor < 0.0)
    assert np.all(hor > -0.25)


def test_isolated_east_peak_obstruction():
    """
    Place a 101.5 m hill exactly 1000 m to the east (azimuth = 90°).
    The peak elevation angle should be close to arctan(100 / 1000) ~ 5.71°.
    """
    grid_size = 501
    pixel_m = 5.0
    dem = np.zeros((grid_size, grid_size), dtype=float)
    center = grid_size // 2

    # 1000 m east = +200 columns.
    dem[center - 1 : center + 2, center + 200] = 101.5

    az, hor, _ = compute_horizon_from_grid(
        dem_m=dem,
        pixel_size_m=pixel_m,
        obs_row=center,
        obs_col=center,
        observer_height_m=1.5,
        azimuth_step_deg=1.0,
        max_radius_m=1200.0,
    )

    east_idx = int(np.where(np.isclose(az, 90.0))[0][0])
    expected_deg = np.degrees(
        np.arctan2(100.0 - (1000.0**2) / (2.0 * MOON_RADIUS_M), 1000.0)
    )
    assert np.isclose(hor[east_idx], expected_deg, atol=0.05)