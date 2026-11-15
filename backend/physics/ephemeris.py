from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List
import numpy as np
import spiceypy as spice

# Standard lunar datum radius (LOLA / IAU MOON_ME) in kilometers.
MOON_DATUM_RADIUS_KM = 1737.4

REQUIRED_KERNELS = [
    "naif0012.tls",
    "de421.bsp",
    "pck00011.tpc",
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
    "moon_assoc_me.tf",
]


@dataclass
class EphemerisPoint:
    utc: str
    et: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_distance_km: float
    sun_angular_radius_deg: float
    earth_azimuth_deg: float
    earth_elevation_deg: float
    earth_distance_km: float


class LunarEphemerisEngine:
    """
    Compute Sun and Earth geometry from the lunar surface
    using the MOON_ME / DE421 reference frame and the local East-North-Up (ENU) basis.
    """

    def __init__(self, kernel_dir: str | Path = "data/raw/spice") -> None:
        self.kernel_dir = Path(kernel_dir)
        self._load_kernels()

        # Read the Sun radius from the PCK kernel (fall back to 696000.0 km if unavailable).
        try:
            _, sun_radii = spice.bodvrd("SUN", "RADII", 3)
            self.sun_radius_km = float(sun_radii[0])
        except Exception:
            self.sun_radius_km = 696000.0

    def _load_kernels(self) -> None:
        """Load all required NAIF SPICE kernels into the kernel pool."""
        spice.kclear()
        for kernel_name in REQUIRED_KERNELS:
            kernel_path = self.kernel_dir / kernel_name
            if not kernel_path.exists():
                raise FileNotFoundError(f"SPICE kernel not found: {kernel_path}")
            spice.furnsh(str(kernel_path.resolve()))

    @staticmethod
    def _enu_basis_and_position(
        lat_deg: float, lon_deg: float, elev_m: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the observer position vector (km) and unit basis vectors (East, North, Up)
        in the body-fixed MOON_ME frame.
        """
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)

        # Local East-North-Up (ENU) basis.
        east_hat = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
        north_hat = np.array(
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float
        )
        up_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)

        # Observer position from the lunar center of mass (km).
        r_km = MOON_DATUM_RADIUS_KM + (elev_m / 1000.0)
        obs_pos_km = r_km * up_hat

        return obs_pos_km, east_hat, north_hat, up_hat

    @staticmethod
    def _vector_to_az_el(
        vec_me: np.ndarray,
        east_hat: np.ndarray,
        north_hat: np.ndarray,
        up_hat: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Project a target direction vector onto the local ENU basis to obtain:
        - Azimuth (0° = North, 90° = East, 180° = South, 270° = West)
        - Elevation (-90° to +90° above the local horizontal plane)
        - Distance (km)
        """
        dist_km = float(np.linalg.norm(vec_me))
        v_east = float(np.dot(vec_me, east_hat))
        v_north = float(np.dot(vec_me, north_hat))
        v_up = float(np.dot(vec_me, up_hat))

        azimuth_deg = float(np.degrees(np.arctan2(v_east, v_north)) % 360.0)
        elevation_deg = float(np.degrees(np.arcsin(np.clip(v_up / dist_km, -1.0, 1.0))))

        return azimuth_deg, elevation_deg, dist_km

    def evaluate_single(
        self,
        utc_str: str,
        lat_deg: float,
        lon_deg: float,
        elev_m: float = 0.0,
        abcorr: str = "LT+S",
    ) -> EphemerisPoint:
        """
        Compute the Sun and Earth azimuth and elevation angles for one UTC time.
        abcorr='LT+S' produces apparent positions consistent with JPL Horizons.
        """
        et = float(spice.str2et(utc_str))
        obs_pos_km, east_hat, north_hat, up_hat = self._enu_basis_and_position(
            lat_deg, lon_deg, elev_m
        )

        # Sun center position relative to the lunar center in the MOON_ME frame.
        sun_from_moon_km, _ = spice.spkpos("SUN", et, "MOON_ME", abcorr, "MOON")
        obs_to_sun_km = np.array(sun_from_moon_km, dtype=float) - obs_pos_km
        sun_az, sun_el, sun_dist = self._vector_to_az_el(
            obs_to_sun_km, east_hat, north_hat, up_hat
        )
        sun_ang_rad_deg = float(
            np.degrees(np.arcsin(np.clip(self.sun_radius_km / sun_dist, 0.0, 1.0)))
        )

        # Earth center position relative to the lunar center in the MOON_ME frame.
        earth_from_moon_km, _ = spice.spkpos("EARTH", et, "MOON_ME", abcorr, "MOON")
        obs_to_earth_km = np.array(earth_from_moon_km, dtype=float) - obs_pos_km
        earth_az, earth_el, earth_dist = self._vector_to_az_el(
            obs_to_earth_km, east_hat, north_hat, up_hat
        )

        return EphemerisPoint(
            utc=utc_str,
            et=et,
            sun_azimuth_deg=sun_az,
            sun_elevation_deg=sun_el,
            sun_distance_km=sun_dist,
            sun_angular_radius_deg=sun_ang_rad_deg,
            earth_azimuth_deg=earth_az,
            earth_elevation_deg=earth_el,
            earth_distance_km=earth_dist,
        )

    def evaluate_series(
        self,
        start_utc: str,
        duration_hours: float,
        step_minutes: float,
        lat_deg: float,
        lon_deg: float,
        elev_m: float = 0.0,
        abcorr: str = "LT+S",
    ) -> List[EphemerisPoint]:
        """Generate a time series of Sun and Earth positions over the mission duration."""
        clean_start = start_utc.replace("Z", "+00:00")
        t0 = datetime.fromisoformat(clean_start)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)

        total_steps = int(round((duration_hours * 60.0) / step_minutes)) + 1
        results: List[EphemerisPoint] = []

        for step_idx in range(total_steps):
            current_dt = t0 + timedelta(minutes=step_idx * step_minutes)
            utc_iso = current_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            results.append(
                self.evaluate_single(
                    utc_str=utc_iso,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    elev_m=elev_m,
                    abcorr=abcorr,
                )
            )
        return results


if __name__ == "__main__":
    # Quick test at the Site01 (Connecting Ridge) coordinates.
    engine = LunarEphemerisEngine()
    sample = engine.evaluate_single(
        utc_str="2026-11-15T00:00:00Z",
        lat_deg=-89.5,
        lon_deg=225.0,
        elev_m=1200.0,
    )
    print(f"UTC Time             : {sample.utc}")
    print(f"Sun Azimuth / Elev   : {sample.sun_azimuth_deg:.4f}° / {sample.sun_elevation_deg:.4f}°")
    print(f"Sun Angular Radius   : {sample.sun_angular_radius_deg:.4f}°")
    print(f"Earth Azimuth / Elev : {sample.earth_azimuth_deg:.4f}° / {sample.earth_elevation_deg:.4f}°")