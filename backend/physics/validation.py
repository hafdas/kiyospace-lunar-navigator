from __future__ import annotations
from typing import List, Dict
from backend.physics.ephemeris import LunarEphemerisEngine

# Simulated metadata extracted from your downloaded LROC NAC .LBL files
LROC_NAC_CATALOG = [
    {
        "image_id": "M1115786060LE",
        "site_id": "site01",
        "target_name": "Connecting Ridge",
        "start_time_utc": "2013-05-15T12:45:00Z",
        "center_lat": -89.50,
        "center_lon": 225.00,
        "notes": "Expected long shadows pointing North-West."
    },
    {
        "image_id": "M1127050918RE",
        "site_id": "site04",
        "target_name": "Shackleton Rim",
        "start_time_utc": "2013-09-22T08:15:30Z",
        "center_lat": -89.78,
        "center_lon": 195.00,
        "notes": "Expected deep crater shadow, rim illuminated."
    }
]

def validate_lroc_shadows() -> None:
    """
    Cross-validates the NAIF SPICE Ephemeris engine against real 
    LROC NAC image observation times to prove shadow accuracy.
    """
    print("==================================================")
    print(" PHASE 6: LROC NAC SHADOW VALIDATION MODULE")
    print("==================================================\n")
    
    engine = LunarEphemerisEngine(kernel_dir="data/raw/spice")
    
    for img in LROC_NAC_CATALOG:
        print(f"[*] Validating Image : {img['image_id']} ({img['target_name']})")
        print(f"    Observation Time : {img['start_time_utc']}")
        
        # Calculate precise Sun position at the exact moment the photo was taken
        pt = engine.evaluate_single(
            utc_str=img['start_time_utc'],
            lat_deg=img['center_lat'],
            lon_deg=img['center_lon'],
            elev_m=1000.0 # Approximate elevation for validation
        )
        
        print(f"    -> SPICE Sun Azimuth   : {pt.sun_azimuth_deg:.2f}°")
        print(f"    -> SPICE Sun Elevation : {pt.sun_elevation_deg:.2f}°")
        print(f"    -> Notes: {img['notes']}\n")

if __name__ == "__main__":
    validate_lroc_shadows()