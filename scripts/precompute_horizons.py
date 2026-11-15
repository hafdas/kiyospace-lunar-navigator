from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from backend.physics.horizon import compute_horizon_from_geotiff

SITE_PRESETS = {
    "site01": {
        "name": "Site01 — Connecting Ridge",
        "dem_path": "data/raw/site01/Site01_final_adj_5mpp_surf.tif",
        "lat_deg": -89.5000,
        "lon_deg": 225.0000,
        "observer_height_m": 1.5,
    },
    "site04": {
        "name": "Site04 — Shackleton Rim",
        "dem_path": "data/raw/site04/Site04_final_adj_5mpp_surf.tif",
        "lat_deg": -89.7800,
        "lon_deg": 195.0000,
        "observer_height_m": 1.5,
    },
}


def build_horizon_cache(output_path: str | Path = "data/cache/horizons.json") -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {}

    for site_id, meta in SITE_PRESETS.items():
        dem_file = Path(meta["dem_path"])
        if dem_file.exists():
            print(f"[INFO] Computing 360° horizon from GeoTIFF for {site_id} ({meta['name']})...")
            profile = compute_horizon_from_geotiff(
                dem_path=dem_file,
                lat_deg=meta["lat_deg"],
                lon_deg=meta["lon_deg"],
                observer_height_m=meta["observer_height_m"],
                azimuth_step_deg=1.0,
                max_radius_m=10_000.0,
            )
            cache_payload[site_id] = {
                "site_id": site_id,
                "name": meta["name"],
                "data_label": "SIMULATED (LOLA 5m DEM Ray-Traced)",
                "observer_lat_deg": profile.observer_lat_deg,
                "observer_lon_deg": profile.observer_lon_deg,
                "observer_height_m": profile.observer_height_m,
                "ground_elevation_m": profile.ground_elevation_m,
                "azimuths_deg": profile.azimuths_deg.tolist(),
                "horizon_elevations_deg": np.round(profile.horizon_elevations_deg, 4).tolist(),
            }
        else:
            # Fallback synthetic profile if raw GeoTIFF is not present on disk yet
            print(f"[WARN] {dem_file} not found. Generating labeled SYNTHETIC fallback for {site_id}...")
            az = np.arange(0.0, 360.0, 1.0)
            phase_shift = 0.0 if site_id == "site01" else 1.2
            hor = 0.35 * np.sin(np.radians(az * 2.0) + phase_shift) + 0.25 * np.cos(np.radians(az * 5.0))
            cache_payload[site_id] = {
                "site_id": site_id,
                "name": meta["name"],
                "data_label": "SYNTHETIC (Fallback Horizon)",
                "observer_lat_deg": meta["lat_deg"],
                "observer_lon_deg": meta["lon_deg"],
                "observer_height_m": meta["observer_height_m"],
                "ground_elevation_m": 1250.0 if site_id == "site01" else 2050.0,
                "azimuths_deg": az.tolist(),
                "horizon_elevations_deg": np.round(hor, 4).tolist(),
            }

    out_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved pre-computed horizon cache to {out_file}")


if __name__ == "__main__":
    build_horizon_cache()