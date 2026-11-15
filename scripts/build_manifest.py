import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import rasterio

RAW_DIR = Path("data/raw")
OUTPUT_MANIFEST = Path("data/provenance_manifest.json")

# Official source mapping (URL/DOI) by dataset subfolder.
SOURCE_CATALOG = {
    "site01": {
        "dataset_name": "NASA LOLA PGDA 5m DEM (Site01 - Connecting Ridge)",
        "source_url": "https://pgda.gsfc.nasa.gov/products/78",
        "reference_frame": "MOON_ME / DE421",
    },
    "site04": {
        "dataset_name": "NASA LOLA PGDA 5m DEM (Site04 - Shackleton Rim)",
        "source_url": "https://pgda.gsfc.nasa.gov/products/78",
        "reference_frame": "MOON_ME / DE421",
    },
    "sdem": {
        "dataset_name": "NASA SDEM PGDA 5m Shape-from-Shading DEM",
        "source_url": "https://doi.org/10.5281/zenodo.17954508",
        "reference_frame": "MOON_ME / DE421",
    },
    "lupex": {
        "dataset_name": "JAXA LUPEX 1m DTM Benchmark (CR1)",
        "source_url": "https://archive.jlpeda.isas.jaxa.jp/pub/product/lupex-nac-dtm-orthomosaics/",
        "reference_frame": "MOON_ME / DE421",
    },
    "spice": {
        "dataset_name": "NASA NAIF SPICE Generic Kernels",
        "source_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
        "reference_frame": "MOON_ME / DE421",
    },
    "lroc_nac": {
        "dataset_name": "NASA LROC NAC Calibrated Optical Imagery",
        "source_url": "https://ode.rsl.wustl.edu/moon/",
        "reference_frame": "MOON_ME / DE421",
    },
}

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def extract_geotiff_info(filepath: Path) -> dict:
    try:
        with rasterio.open(filepath) as src:
            return {
                "crs": src.crs.to_string() if src.crs else "UNDEFINED",
                "resolution_m": [round(abs(src.res[0]), 3), round(abs(src.res[1]), 3)],
                "dimensions_px": [src.width, src.height],
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
            }
    except Exception as e:
        return {"raster_read_error": str(e)}

def main():
    if not RAW_DIR.exists():
        print(f"Folder {RAW_DIR} was not found.")
        return

    entries = []
    for file_path in sorted(RAW_DIR.rglob("*")):
        if not file_path.is_file():
            continue

        # Identify the category from the first subfolder name under data/raw/.
        rel_parts = file_path.relative_to(RAW_DIR).parts
        category = rel_parts[0].lower() if len(rel_parts) > 1 else "uncategorized"
        source_meta = SOURCE_CATALOG.get(category, {
            "dataset_name": "Custom / Uncategorized Lunar Dataset",
            "source_url": "LOCAL_INGEST",
            "reference_frame": "MOON_ME / DE421",
        })

        stat = file_path.stat()
        download_date = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        record = {
            "file_path": file_path.as_posix(),
            "file_name": file_path.name,
            "category": category,
            "dataset_name": source_meta["dataset_name"],
            "source_url": source_meta["source_url"],
            "reference_frame": source_meta["reference_frame"],
            "download_timestamp_utc": download_date,
            "size_bytes": stat.st_size,
            "sha256": compute_sha256(file_path),
        }

        if file_path.suffix.lower() in {".tif", ".tiff"}:
            record["spatial_metadata"] = extract_geotiff_info(file_path)

        entries.append(record)
        print(f"[TERCATAT] {file_path.as_posix()} (SHA-256: {record['sha256'][:12]}...)")

    manifest = {
        "project": "KiyoSpace Lunar Landing Site Navigator",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_files": len(entries),
        "files": entries,
    }

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSelesai! {len(entries)} file tercatat di {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()