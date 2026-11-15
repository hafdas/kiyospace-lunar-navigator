from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer

MOON_RADIUS_M = 1_737_400.0


@dataclass
class HorizonProfile:
    azimuths_deg: np.ndarray
    horizon_elevations_deg: np.ndarray
    observer_lat_deg: float
    observer_lon_deg: float
    observer_height_m: float
    ground_elevation_m: float


def compute_horizon_from_grid(
    dem_m: np.ndarray,
    pixel_size_m: float,
    obs_row: float,
    obs_col: float,
    observer_height_m: float = 1.5,
    azimuth_step_deg: float = 1.0,
    radial_step_m: float | None = None,
    max_radius_m: float | None = None,
    nodata_val: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the horizon angle profile H(azimuth), 0°–360°, on the local elevation grid,
    accounting for lunar surface curvature (R = 1737.4 km).
    South Polar Stereographic grid convention (X = Easting, Y = Northing upward / rows downward):
      - Azimuth 0°   (North / +Y) -> d_row < 0, d_col = 0
      - Azimuth 90°  (East / +X) -> d_row = 0, d_col > 0
      - Azimuth 180° (South / -Y) -> d_row > 0, d_col = 0
      - Azimuth 270° (West / -X) -> d_row = 0, d_col < 0
    """
    rows, cols = dem_m.shape
    r0 = int(round(obs_row))
    c0 = int(round(obs_col))

    if not (0 <= r0 < rows and 0 <= c0 < cols):
        raise ValueError(f"Observer point ({obs_row}, {obs_col}) is outside DEM bounds {dem_m.shape}.")

    ground_z = float(dem_m[r0, c0])
    if nodata_val is not None and np.isclose(ground_z, nodata_val):
        raise ValueError("The observer point falls on a NoData pixel.")

    z_obs = ground_z + observer_height_m
    step_m = float(radial_step_m) if radial_step_m is not None else float(pixel_size_m)

    # Default to the maximum distance to the raster edge when no limit is specified.
    if max_radius_m is None:
        max_px = min(r0, rows - 1 - r0, c0, cols - 1 - c0)
        max_radius_m = max_px * pixel_size_m

    num_steps = max(1, int(np.floor(max_radius_m / step_m)))
    distances_m = np.arange(1, num_steps + 1, dtype=float) * step_m

    # Lunar curvature correction: elevation drop caused by spherical curvature.
    curvature_drop_m = (distances_m ** 2) / (2.0 * MOON_RADIUS_M)

    n_az = int(round(360.0 / azimuth_step_deg))
    azimuths_deg = np.linspace(0.0, 360.0, n_az, endpoint=False)
    horizon_deg = np.full(n_az, -90.0, dtype=float)

    for idx, az_deg in enumerate(azimuths_deg):
        az_rad = np.radians(az_deg)
        # Pixel direction vector in the stereographic projection raster.
        d_col = (distances_m * np.sin(az_rad)) / pixel_size_m
        d_row = -(distances_m * np.cos(az_rad)) / pixel_size_m

        sample_cols = np.rint(obs_col + d_col).astype(int)
        sample_rows = np.rint(obs_row + d_row).astype(int)

        valid = (
            (sample_rows >= 0)
            & (sample_rows < rows)
            & (sample_cols >= 0)
            & (sample_cols < cols)
        )
        if not np.any(valid):
            continue

        s_rows = sample_rows[valid]
        s_cols = sample_cols[valid]
        s_dist = distances_m[valid]
        s_curv = curvature_drop_m[valid]

        z_terrain = dem_m[s_rows, s_cols].astype(float)
        if nodata_val is not None:
            mask_ok = ~np.isclose(z_terrain, nodata_val) & np.isfinite(z_terrain)
            if not np.any(mask_ok):
                continue
            z_terrain = z_terrain[mask_ok]
            s_dist = s_dist[mask_ok]
            s_curv = s_curv[mask_ok]

        # Terrain peak elevation angle relative to the observer's local horizontal.
        dz_apparent = (z_terrain - s_curv) - z_obs
        elev_angles = np.degrees(np.arctan2(dz_apparent, s_dist))
        horizon_deg[idx] = float(np.max(elev_angles))

    return azimuths_deg, horizon_deg, ground_z


def compute_horizon_from_geotiff(
    dem_path: str | Path,
    lat_deg: float,
    lon_deg: float,
    observer_height_m: float = 1.5,
    azimuth_step_deg: float = 1.0,
    max_radius_m: float = 15_000.0,
) -> HorizonProfile:
    """
    Read a LOLA/SDEM GeoTIFF in the South Polar Stereographic projection,
    convert (lat, lon) in MOON_ME to pixel coordinates, and compute a 360° horizon profile.
    """
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")

    with rasterio.open(path) as src:
        # Transform lunar geographic coordinates (MOON_ME) to the raster CRS.
        geo_crs = src.crs.geodetic_crs
        transformer = Transformer.from_crs(geo_crs, src.crs, always_xy=True)
        x_proj, y_proj = transformer.transform(lon_deg, lat_deg)

        obs_row, obs_col = src.index(x_proj, y_proj)
        pixel_size_m = float(abs(src.res[0]))

        # Read a local window around the observer to reduce memory usage.
        radius_px = int(np.ceil(max_radius_m / pixel_size_m)) + 2
        row_start = max(0, obs_row - radius_px)
        row_stop = min(src.height, obs_row + radius_px + 1)
        col_start = max(0, obs_col - radius_px)
        col_stop = min(src.width, obs_col + radius_px + 1)

        window = rasterio.windows.Window.from_slices(
            (row_start, row_stop), (col_start, col_stop)
        )
        dem_patch = src.read(1, window=window)

        local_row = float(obs_row - row_start)
        local_col = float(obs_col - col_start)

        az_deg, hor_deg, ground_z = compute_horizon_from_grid(
            dem_m=dem_patch,
            pixel_size_m=pixel_size_m,
            obs_row=local_row,
            obs_col=local_col,
            observer_height_m=observer_height_m,
            azimuth_step_deg=azimuth_step_deg,
            max_radius_m=max_radius_m,
            nodata_val=src.nodata,
        )

    return HorizonProfile(
        azimuths_deg=az_deg,
        horizon_elevations_deg=hor_deg,
        observer_lat_deg=lat_deg,
        observer_lon_deg=lon_deg,
        observer_height_m=observer_height_m,
        ground_elevation_m=ground_z,
    )


if __name__ == "__main__":
    # Run a direct test on the real Site01 raster (Connecting Ridge), if available.
    site01_dem = Path("data/raw/site01/Site01_final_adj_5mpp_surf.tif")
    if site01_dem.exists():
        profile = compute_horizon_from_geotiff(
            dem_path=site01_dem,
            lat_deg=-89.5,
            lon_deg=225.0,
            observer_height_m=1.5,
            azimuth_step_deg=1.0,
            max_radius_m=10_000.0,
        )
        print(f"Site01 ground elevation (-89.5°, 225°): {profile.ground_elevation_m:.2f} m")
        print(f"Minimum horizon angle  : {np.min(profile.horizon_elevations_deg):.3f}°")
        print(f"Maximum horizon angle  : {np.max(profile.horizon_elevations_deg):.3f}°")
        print(f"Mean horizon angle     : {np.mean(profile.horizon_elevations_deg):.3f}°")
    else:
        print("Site01_final_adj_5mpp_surf.tif was not found in data/raw/site01/.")