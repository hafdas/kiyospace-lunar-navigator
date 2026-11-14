from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List
import numpy as np
import spiceypy as spice

# Radius datum standar Bulan (LOLA / IAU MOON_ME) dalam kilometer
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
    Mesin perhitungan geometri Matahari dan Bumi dari permukaan Bulan
    menggunakan kerangka referensi MOON_ME / DE421 dan basis lokal East-North-Up (ENU).
    """

    def __init__(self, kernel_dir: str | Path = "data/raw/spice") -> None:
        self.kernel_dir = Path(kernel_dir)
        self._load_kernels()

        # Ambil radius Matahari dari kernel PCK (fallback ke 696000.0 km jika belum dimuat)
        try:
            _, sun_radii = spice.bodvrd("SUN", "RADII", 3)
            self.sun_radius_km = float(sun_radii[0])
        except Exception:
            self.sun_radius_km = 696000.0

    def _load_kernels(self) -> None:
        """Memuat seluruh kernel NAIF SPICE yang dibutuhkan ke dalam kernel pool."""
        spice.kclear()
        for kernel_name in REQUIRED_KERNELS:
            kernel_path = self.kernel_dir / kernel_name
            if not kernel_path.exists():
                raise FileNotFoundError(f"Kernel SPICE tidak ditemukan: {kernel_path}")
            spice.furnsh(str(kernel_path.resolve()))

    @staticmethod
    def _enu_basis_and_position(
        lat_deg: float, lon_deg: float, elev_m: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Menghitung vektor posisi pengamat (km) dan vektor basis satuan (East, North, Up)
        dalam kerangka body-fixed MOON_ME.
        """
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)

        # Basis lokal East-North-Up (ENU)
        east_hat = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
        north_hat = np.array(
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float
        )
        up_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)

        # Posisi pengamat dari pusat massa Bulan (dalam km)
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
        Memproyeksikan vektor arah target ke basis lokal ENU menjadi:
        - Azimuth (0° = Utara, 90° = Timur, 180° = Selatan, 270° = Barat)
        - Elevation (-90° hingga +90° di atas bidang horizontal lokal)
        - Jarak (km)
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
        Menghitung sudut azimuth & elevasi Matahari dan Bumi untuk satu waktu UTC.
        abcorr='LT+S' menghasilkan posisi tampak (apparent) yang selaras dengan JPL Horizons.
        """
        et = float(spice.str2et(utc_str))
        obs_pos_km, east_hat, north_hat, up_hat = self._enu_basis_and_position(
            lat_deg, lon_deg, elev_m
        )

        # Posisi pusat Matahari relatif terhadap pusat Bulan dalam kerangka MOON_ME
        sun_from_moon_km, _ = spice.spkpos("SUN", et, "MOON_ME", abcorr, "MOON")
        obs_to_sun_km = np.array(sun_from_moon_km, dtype=float) - obs_pos_km
        sun_az, sun_el, sun_dist = self._vector_to_az_el(
            obs_to_sun_km, east_hat, north_hat, up_hat
        )
        sun_ang_rad_deg = float(
            np.degrees(np.arcsin(np.clip(self.sun_radius_km / sun_dist, 0.0, 1.0)))
        )

        # Posisi pusat Bumi relatif terhadap pusat Bulan dalam kerangka MOON_ME
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
        """Menghasilkan deret waktu posisi Matahari & Bumi selama durasi misi."""
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
    # Contoh uji cepat pada koordinat Site01 (Connecting Ridge)
    engine = LunarEphemerisEngine()
    sample = engine.evaluate_single(
        utc_str="2026-11-15T00:00:00Z",
        lat_deg=-89.5,
        lon_deg=225.0,
        elev_m=1200.0,
    )
    print(f"Waktu UTC            : {sample.utc}")
    print(f"Sun Azimuth / Elev   : {sample.sun_azimuth_deg:.4f}° / {sample.sun_elevation_deg:.4f}°")
    print(f"Sun Angular Radius   : {sample.sun_angular_radius_deg:.4f}°")
    print(f"Earth Azimuth / Elev : {sample.earth_azimuth_deg:.4f}° / {sample.earth_elevation_deg:.4f}°")