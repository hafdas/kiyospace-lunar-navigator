"use client";

import React, { useRef, useState, useEffect } from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";

export interface RecommendedLandingSite {
  rank: number;
  region_name: string;
  lat_deg: number;
  lon_deg: number;
  elevation_m: number;
  slope_deg: number;
  predicted_min_soc_pct: number;
  predicted_feasibility: string;
  data_label: string;
}

interface LunarGlobe3DProps {
  selectedDateUtc: string;
  onCoordinateSelected: (lat: number, lon: number, predictedSoc?: number) => void;
}

interface PointPrediction {
  predicted_min_soc_pct: number;
  predicted_feasibility: string;
  elevation_m?: number;
  slope_deg?: number;
}

export default function LunarGlobe3D({
  selectedDateUtc,
  onCoordinateSelected,
}: LunarGlobe3DProps) {
  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 700, height: 420 });

  // State for manually entered or clicked coordinates.
  const [clickCoord, setClickCoord] = useState<{ lat: number; lng: number } | null>(null);
  const [manualLat, setManualLat] = useState<string>("-89.5000");
  const [manualLon, setManualLon] = useState<string>("225.0000");
  const [pointPrediction, setPointPrediction] = useState<PointPrediction | null>(null);
  const [isPredictingPoint, setIsPredictingPoint] = useState(false);

  // ML best-site recommendations for the selected date.
  const [searchDateUtc, setSearchDateUtc] = useState<string>(selectedDateUtc);
  const [recommendedSites, setRecommendedSites] = useState<RecommendedLandingSite[]>([]);
  const [isScanningSites, setIsScanningSites] = useState(false);
  const [scanSummary, setScanSummary] = useState<string | null>(null);

  useEffect(() => {
    void Promise.resolve().then(() => setSearchDateUtc(selectedDateUtc));
  }, [selectedDateUtc]);

  // Keep the globe sized to its container.
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: 420,
        });
      }
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Point the camera at the lunar south pole on initial load.
  useEffect(() => {
    if (globeRef.current) {
      globeRef.current.pointOfView({ lat: -88.5, lng: 210, altitude: 0.45 }, 1500);
    }
  }, []);

  const evaluatePointWithML = async (lat: number, lng: number) => {
    const normalizedLon = ((lng % 360) + 360) % 360;
    setClickCoord({ lat, lng: normalizedLon });
    setManualLat(lat.toFixed(4));
    setManualLon(normalizedLon.toFixed(4));
    setIsPredictingPoint(true);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/api/predict-viability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat_deg: lat,
          lon_deg: normalizedLon,
          start_utc: searchDateUtc,
        }),
      });
      const data = await res.json();
      setPointPrediction(data);
      onCoordinateSelected(lat, normalizedLon, data.predicted_min_soc_pct);
    } catch (err) {
      console.error("Failed to fetch ML viability prediction:", err);
      onCoordinateSelected(lat, normalizedLon);
    } finally {
      setIsPredictingPoint(false);
    }
  };

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const latNum = parseFloat(manualLat);
    const lonNum = parseFloat(manualLon);
    if (isNaN(latNum) || isNaN(lonNum) || latNum < -90 || latNum > 90) return;

    evaluatePointWithML(latNum, lonNum);
    if (globeRef.current) {
      globeRef.current.pointOfView({ lat: latNum, lng: lonNum, altitude: 0.35 }, 1000);
    }
  };

  const handleFindBestSites = async () => {
    setIsScanningSites(true);
    setScanSummary(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/api/recommend-sites`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_utc: searchDateUtc,
          top_k: 3,
          max_slope_deg: 12.0,
        }),
      });
      const data = await res.json();
      setRecommendedSites(data.recommendations || []);
      setScanSummary(`Scanned ${data.candidates_scanned} South Pole points via LightGBM`);

      if (data.recommendations && data.recommendations.length > 0 && globeRef.current) {
        const best = data.recommendations[0];
        globeRef.current.pointOfView({ lat: best.lat_deg, lng: best.lon_deg, altitude: 0.35 }, 1200);
      }
    } catch (err) {
      console.error("Failed to run ML site recommendation:", err);
    } finally {
      setIsScanningSites(false);
    }
  };

  // Combine the selected point marker with the top three ML recommendations.
  const markerData = [
    ...recommendedSites.map((site) => ({
      lat: site.lat_deg,
      lng: site.lon_deg,
      size: site.rank === 1 ? 0.14 : 0.1,
      color: site.rank === 1 ? "#10b981" : "#f59e0b",
      label: `#${site.rank} AI Best (${site.predicted_min_soc_pct.toFixed(1)}% SOC)`,
    })),
    ...(clickCoord
      ? [
          {
            lat: clickCoord.lat,
            lng: clickCoord.lng,
            size: 0.12,
            color: "#ef4444",
            label: "Selected Target",
          },
        ]
      : []),
  ];

  return (
    <div className="space-y-3">
      {/* BAR 1: AI BEST LANDING SITE FINDER FOR SELECTED DATE */}
      <div className="bg-slate-900 text-white p-3.5 rounded-xl border border-slate-700 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
            ✨ ML Site Finder
          </span>
          <input
            type="text"
            value={searchDateUtc}
            onChange={(e) => setSearchDateUtc(e.target.value)}
            placeholder="2026-11-29T00:00:00Z"
            className="bg-slate-800 border border-slate-600 rounded-lg px-2.5 py-1.5 text-xs font-mono text-white w-48 focus:border-emerald-400 focus:outline-none"
          />
          <button
            type="button"
            onClick={handleFindBestSites}
            disabled={isScanningSites}
            className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-600 text-white text-xs font-semibold transition cursor-pointer"
          >
            {isScanningSites ? "Scanning South Pole..." : "Find Best Sites for Date"}
          </button>
        </div>

        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-700">
          AI-ESTIMATED • LightGBM + LOLA 5m DEM
        </span>
      </div>

      {/* BAR 2: Top three ML-recommended landing sites */}
      {recommendedSites.length > 0 && (
        <div className="bg-emerald-50/80 border border-emerald-200 rounded-xl p-3 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-bold text-emerald-900">
              Top 3 AI-Recommended Landing Sites for {searchDateUtc}
            </p>
            {scanSummary && <span className="text-[11px] text-emerald-700">{scanSummary}</span>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {recommendedSites.map((site) => (
              <div
                key={site.rank}
                className="bg-white rounded-lg border border-emerald-200 p-2.5 shadow-xs flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold text-emerald-700">
                      #{site.rank} • {site.predicted_feasibility}
                    </span>
                    <span className="text-xs font-mono font-bold text-slate-900 bg-emerald-100 px-1.5 py-0.5 rounded">
                      {site.predicted_min_soc_pct.toFixed(1)}% SOC
                    </span>
                  </div>
                  <p className="text-xs font-semibold text-slate-800 mt-1 truncate">
                    {site.region_name}
                  </p>
                  <p className="text-[11px] font-mono text-slate-500 mt-0.5">
                    {site.lat_deg.toFixed(4)}°, {site.lon_deg.toFixed(4)}° • Elev {site.elevation_m}m • Slope {site.slope_deg}°
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => evaluatePointWithML(site.lat_deg, site.lon_deg)}
                  className="mt-2 w-full py-1 rounded bg-[#1e3a8a] hover:bg-blue-800 text-white text-[11px] font-semibold transition cursor-pointer"
                >
                  Apply to Active Scenario
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3D lunar globe */}
      <div
        ref={containerRef}
        className="w-full h-[420px] bg-[#090d16] rounded-xl overflow-hidden relative border border-slate-800"
      >
        {/* Top left: manual coordinate input */}
        <form
          onSubmit={handleManualSubmit}
          className="absolute top-3 left-3 z-10 bg-slate-900/90 p-3 rounded-lg border border-slate-700 backdrop-blur-sm space-y-2 w-56"
        >
          <p className="text-xs font-bold text-white flex items-center gap-1.5">
            🌕 3D Lunar South Pole
          </p>
          <p className="text-[10px] text-slate-300 leading-tight">
            Click anywhere on the Moon or enter exact coordinates:
          </p>
          <div className="grid grid-cols-2 gap-1.5">
            <div>
              <label className="block text-[9px] text-slate-400">Lat (-90..0)</label>
              <input
                type="text"
                value={manualLat}
                onChange={(e) => setManualLat(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded px-1.5 py-1 text-[11px] font-mono text-white"
              />
            </div>
            <div>
              <label className="block text-[9px] text-slate-400">Lon (0..360)</label>
              <input
                type="text"
                value={manualLon}
                onChange={(e) => setManualLon(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded px-1.5 py-1 text-[11px] font-mono text-white"
              />
            </div>
          </div>
          <button
            type="submit"
            className="w-full py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-semibold cursor-pointer"
          >
            Set & Predict Point
          </button>
        </form>

        {/* Top right: ML prediction for the selected point */}
        {clickCoord && (
          <div className="absolute top-3 right-3 z-10 bg-slate-900/90 p-3 rounded-lg border border-emerald-500/80 backdrop-blur-sm w-56">
            <p className="text-[10px] text-slate-400 font-semibold">Selected Target Coordinates</p>
            <p className="text-xs text-white font-mono mt-0.5">
              {clickCoord.lat.toFixed(4)}° / {clickCoord.lng.toFixed(4)}°
            </p>
            <div className="mt-2 pt-2 border-t border-slate-700">
              <p className="text-[10px] text-emerald-400 font-bold">
                ML Viability (LightGBM)
              </p>
              {isPredictingPoint ? (
                <p className="text-xs text-slate-300 animate-pulse mt-1">Predicting...</p>
              ) : pointPrediction ? (
                <div className="mt-1 space-y-0.5">
                  <div className="flex items-baseline justify-between">
                    <span className="text-lg font-mono font-bold text-white">
                      {pointPrediction.predicted_min_soc_pct?.toFixed(1)}%
                    </span>
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300">
                      {pointPrediction.predicted_feasibility}
                    </span>
                  </div>
                  {pointPrediction.elevation_m !== undefined &&
                    pointPrediction.slope_deg !== undefined && (
                      <p className="text-[10px] text-slate-300 font-mono">
                        Elev: {pointPrediction.elevation_m}m • Slope: {pointPrediction.slope_deg}°
                      </p>
                    )}
                  <p className="text-[9px] text-emerald-500 font-bold uppercase tracking-wider pt-0.5">
                    AI-ESTIMATED
                  </p>
                </div>
              ) : null}
            </div>
          </div>
        )}

        <Globe
          ref={globeRef}
          width={dimensions.width}
          height={dimensions.height}
          globeImageUrl="/moon_8k.jpg"
          bumpImageUrl="https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_bump_1024.jpg"
          backgroundColor="rgba(9,13,22,1)"
          onGlobeClick={(coord: { lat: number; lng: number }) =>
            evaluatePointWithML(coord.lat, coord.lng)
          }
          pointsData={markerData}
          pointAltitude={0.015}
          pointColor="color"
          pointRadius="size"
          pointLabel="label"
        />
      </div>
    </div>
  );
}