"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import ScenarioCharts, {
  type MissionEvaluationResponse,
} from "@/components/ScenarioCharts";
import AiDecisionPanel from "@/components/AiDecisionPanel";

const LunarGlobe3D = dynamic(() => import("../components/LunarMap3D"), {
  ssr: false,
  loading: () => (
    <div className="h-[420px] w-full rounded-xl bg-slate-900 flex items-center justify-center text-slate-400 text-sm">
      Initializing 3D Lunar South Pole Globe & ML Surrogate...
    </div>
  ),
});

export default function KiyoSpaceApexDashboard() {
  const [siteA, setSiteA] = useState<string>("site01");
  const [startUtcA, setStartUtcA] = useState<string>("2026-11-15T00:00:00Z");
  const [customCoordA, setCustomCoordA] = useState<{ lat: number; lon: number } | null>(null);

  const [siteB, setSiteB] = useState<string>("site04");
  const [startUtcB, setStartUtcB] = useState<string>("2026-11-29T00:00:00Z");
  const [customCoordB, setCustomCoordB] = useState<{ lat: number; lon: number } | null>(null);

  const [activeMapTarget, setActiveMapTarget] = useState<"A" | "B">("B");

  const [durationHours, setDurationHours] = useState<number>(120);
  const [pvAreaM2, setPvAreaM2] = useState<number>(4.0);
  const [batteryCapacityWh, setBatteryCapacityWh] = useState<number>(5000);
  const [nominalLoadW, setNominalLoadW] = useState<number>(250);

  const [resultA, setResultA] = useState<MissionEvaluationResponse | null>(null);
  const [resultB, setResultB] = useState<MissionEvaluationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleEvaluate = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const landerConfig = {
        pv_area_m2: Number(pvAreaM2),
        battery_capacity_wh: Number(batteryCapacityWh),
        nominal_load_w: Number(nominalLoadW),
      };

      const payloadA = {
        site_id: siteA,
        start_utc: startUtcA,
        duration_hours: Number(durationHours),
        custom_lat: siteA === "custom" ? customCoordA?.lat : null,
        custom_lon: siteA === "custom" ? customCoordA?.lon : null,
        lander_config: landerConfig,
      };

      const payloadB = {
        site_id: siteB,
        start_utc: startUtcB,
        duration_hours: Number(durationHours),
        custom_lat: siteB === "custom" ? customCoordB?.lat : null,
        custom_lon: siteB === "custom" ? customCoordB?.lon : null,
        lander_config: landerConfig,
      };

      const [resA, resB] = await Promise.all([
        fetch("http://127.0.0.1:8000/api/evaluate-mission", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payloadA),
        }),
        fetch("http://127.0.0.1:8000/api/evaluate-mission", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payloadB),
        }),
      ]);

      if (!resA.ok || !resB.ok) {
        throw new Error("Backend FastAPI returned an error. Ensure custom coordinates are set.");
      }

      const dataA: MissionEvaluationResponse = await resA.json();
      const dataB: MissionEvaluationResponse = await resB.json();
      setResultA(dataA);
      setResultB(dataB);
    } catch (err: unknown) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : "Failed to connect to the SPICE physics engine."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void Promise.resolve().then(handleEvaluate);
    // Load the default mission scenarios once when the dashboard mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      {/* LEFT SIDEBAR: APEX COBALT NAVY */}
      <aside className="w-64 bg-[#0f172a] text-white hidden md:flex flex-col justify-between shrink-0 border-r border-slate-800">
        <div>
          <div className="p-6 border-b border-slate-800/80">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-700 flex items-center justify-center font-bold text-white shadow-md">
                KS
              </div>
              <div>
                <h1 className="font-bold text-base tracking-tight text-white">KiyoSpace</h1>
                <p className="text-[11px] text-blue-300 font-medium">Lunar Site Navigator</p>
              </div>
            </div>
          </div>

          <nav className="p-4 space-y-1.5 text-sm">
            <a
              href="#mission-control"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg bg-[#1e3a8a] text-white font-semibold shadow-sm"
            >
              <span>🛰️</span> Mission Control
            </a>
            <a
              href="#surface-map"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-slate-300 hover:bg-slate-800/70 hover:text-white transition"
            >
              <span>🌕</span> 3D Globe & ML Site Finder
            </a>
            <a
              href="#scenario-comparison"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-slate-300 hover:bg-slate-800/70 hover:text-white transition"
            >
              <span>📊</span> Scenario A vs B
            </a>
            <a
              href="#ai-copilot"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-slate-300 hover:bg-slate-800/70 hover:text-white transition"
            >
              <span>✨</span> AI Decision Copilot
            </a>
          </nav>
        </div>

        <div className="p-4 m-4 rounded-xl bg-slate-900/90 border border-slate-800 text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Ephemeris</span>
            <span className="text-emerald-400 font-mono font-semibold">SPICE DE421</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Topography</span>
            <span className="text-blue-400 font-mono font-semibold">LOLA 5m DEM</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">ML Surrogate</span>
            <span className="text-amber-400 font-mono font-semibold">LightGBM Active</span>
          </div>
        </div>
      </aside>

      {/* MAIN WORKSPACE */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="bg-white border-b border-slate-200 px-6 py-4 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-20">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-slate-900">
                Artemis South Pole Landing & Replanning Dashboard
              </h2>
              <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-blue-50 text-[#1e3a8a] border border-blue-200">
                Apex Cobalt Edition
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Deterministic SPICE Solar/DTE Simulation + LightGBM Spatial Landing Site Predictor
            </p>
          </div>

          <button
            onClick={handleEvaluate}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-[#1e3a8a] hover:bg-blue-800 disabled:bg-slate-400 text-white text-sm font-semibold shadow-sm transition cursor-pointer flex items-center gap-2"
          >
            {loading ? "⏳ Computing SPICE Physics..." : "▶ Evaluate & Compare Scenarios"}
          </button>
        </header>

        <main className="p-6 space-y-6 max-w-[1600px] w-full mx-auto">
          {errorMsg && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm font-medium">
              ⚠️ {errorMsg}
            </div>
          )}

          {/* 1. MISSION PARAMETERS */}
          <section
            id="mission-control"
            className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm"
          >
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-4">
              1. Lander Hardware & Scenario Schedule Parameters
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Scenario A Site
                </label>
                <select
                  value={siteA}
                  onChange={(e) => setSiteA(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-medium"
                >
                  <option value="site01">Site01 (Connecting Ridge)</option>
                  <option value="site04">Site04 (Shackleton Rim)</option>
                  {customCoordA && (
                    <option value="custom">
                      Custom A ({customCoordA.lat.toFixed(2)}°, {customCoordA.lon.toFixed(2)}°)
                    </option>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Scenario A Start (UTC)
                </label>
                <input
                  type="text"
                  value={startUtcA}
                  onChange={(e) => setStartUtcA(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Scenario B Site
                </label>
                <select
                  value={siteB}
                  onChange={(e) => setSiteB(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-medium"
                >
                  <option value="site01">Site01 (Connecting Ridge)</option>
                  <option value="site04">Site04 (Shackleton Rim)</option>
                  {customCoordB && (
                    <option value="custom">
                      Custom B ({customCoordB.lat.toFixed(2)}°, {customCoordB.lon.toFixed(2)}°)
                    </option>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Scenario B Start (UTC)
                </label>
                <input
                  type="text"
                  value={startUtcB}
                  onChange={(e) => setStartUtcB(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Duration (Hours)
                </label>
                <input
                  type="number"
                  value={durationHours}
                  onChange={(e) => setDurationHours(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Solar PV Area (m²)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={pvAreaM2}
                  onChange={(e) => setPvAreaM2(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Battery (Wh)
                </label>
                <input
                  type="number"
                  step="100"
                  value={batteryCapacityWh}
                  onChange={(e) => setBatteryCapacityWh(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Lander Load (W)
                </label>
                <input
                  type="number"
                  step="10"
                  value={nominalLoadW}
                  onChange={(e) => setNominalLoadW(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-mono"
                />
              </div>
            </div>
          </section>

          {/* 2. MIDDLE ROW: 3D GLOBE + ML SITE FINDER (LEFT) & KPI CARDS (RIGHT) */}
          <section id="surface-map" className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    3D Lunar South Pole Globe & ML Landing Site Predictor
                  </h3>
                  <p className="text-xs text-slate-500">
                    Find Top 3 optimal landing sites for any date or click/enter custom coordinates
                  </p>
                </div>

                {/* Target Assignment Toggle */}
                <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg border border-slate-200">
                  <span className="text-[11px] text-slate-500 px-1.5 font-medium">
                    Assign selection to:
                  </span>
                  <button
                    type="button"
                    onClick={() => setActiveMapTarget("A")}
                    className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer transition ${
                      activeMapTarget === "A"
                        ? "bg-[#1e3a8a] text-white shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Scenario A
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveMapTarget("B")}
                    className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer transition ${
                      activeMapTarget === "B"
                        ? "bg-emerald-600 text-white shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Scenario B
                  </button>
                </div>
              </div>

              <LunarGlobe3D
                selectedDateUtc={activeMapTarget === "A" ? startUtcA : startUtcB}
                onCoordinateSelected={(lat, lon) => {
                  if (activeMapTarget === "A") {
                    setCustomCoordA({ lat, lon });
                    setSiteA("custom");
                  } else {
                    setCustomCoordB({ lat, lon });
                    setSiteB("custom");
                  }
                }}
              />
            </div>

            {/* Right KPI Cards */}
            <div className="lg:col-span-4 flex flex-col gap-5">
              <div className="rounded-xl bg-gradient-to-br from-[#0f172a] via-[#1e3a8a] to-[#1d4ed8] text-white p-6 shadow-md flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-wider text-blue-200 font-semibold">
                      Minimum Battery SOC Delta
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-white/15 text-white font-mono">
                      SIMULATED
                    </span>
                  </div>
                  <div className="mt-4 flex items-baseline gap-4">
                    <div>
                      <p className="text-[11px] text-blue-200">Scenario A</p>
                      <p className="text-3xl font-bold font-mono">
                        {resultA ? `${resultA.summary.min_battery_soc_pct.toFixed(1)}%` : "--"}
                      </p>
                    </div>
                    <span className="text-blue-300 text-xl">→</span>
                    <div>
                      <p className="text-[11px] text-emerald-300">Scenario B</p>
                      <p className="text-3xl font-bold font-mono text-emerald-300">
                        {resultB ? `${resultB.summary.min_battery_soc_pct.toFixed(1)}%` : "--"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-6 space-y-2">
                  <div className="flex justify-between text-xs text-blue-100">
                    <span>Status A: {resultA?.summary.feasibility_status || "PENDING"}</span>
                    <span>Status B: {resultB?.summary.feasibility_status || "PENDING"}</span>
                  </div>
                  <div className="w-full bg-slate-950/40 rounded-full h-2.5 overflow-hidden p-0.5">
                    <div
                      className="bg-amber-400 h-full rounded-full transition-all"
                      style={{ width: `${resultA?.summary.min_battery_soc_pct || 0}%` }}
                    />
                  </div>
                  <div className="w-full bg-slate-950/40 rounded-full h-2.5 overflow-hidden p-0.5">
                    <div
                      className="bg-emerald-400 h-full rounded-full transition-all"
                      style={{ width: `${resultB?.summary.min_battery_soc_pct || 0}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-xl bg-white border border-slate-200 p-6 shadow-sm flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-wider text-slate-500 font-bold">
                      Max DTE Comm Blackout
                    </span>
                    <span className="text-[11px] font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                      L1 Line-of-Sight
                    </span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                      <p className="text-[11px] text-slate-500">Scenario A Max</p>
                      <p className="text-xl font-bold font-mono text-slate-900 mt-0.5">
                        {resultA ? `${resultA.summary.max_dte_blackout_hours.toFixed(1)} h` : "--"}
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-emerald-50/60 border border-emerald-200">
                      <p className="text-[11px] text-emerald-700">Scenario B Max</p>
                      <p className="text-xl font-bold font-mono text-emerald-800 mt-0.5">
                        {resultB ? `${resultB.summary.max_dte_blackout_hours.toFixed(1)} h` : "--"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                  <span>Mean Sun Visibility (B):</span>
                  <span className="font-mono font-bold text-slate-800">
                    {resultB
                      ? `${(resultB.summary.mean_sun_visible_fraction * 100).toFixed(1)}%`
                      : "--"}
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* 3. BOTTOM ROW: SCENARIO CHARTS */}
          <section id="scenario-comparison" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {resultA && (
              <ScenarioCharts
                label="Scenario A (Baseline)"
                accentColor="#1e3a8a"
                data={resultA}
              />
            )}
            {resultB && (
              <ScenarioCharts
                label="Scenario B (Replanning / Delayed)"
                accentColor="#10b981"
                data={resultB}
              />
            )}
          </section>

          {/* 4. AI DECISION SUPPORT PANEL */}
          <section id="ai-copilot">
            <AiDecisionPanel
              baseStartUtc={startUtcA}
              durationHours={durationHours}
              pvAreaM2={pvAreaM2}
              batteryCapacityWh={batteryCapacityWh}
              nominalLoadW={nominalLoadW}
              scenarioAResult={resultA}
              scenarioBResult={resultB}
              onApplyToScenarioB={(recommendedSiteId, recommendedStartUtc) => {
                if (recommendedSiteId === "site01" || recommendedSiteId === "site04") {
                  setSiteB(recommendedSiteId);
                }
                setStartUtcB(recommendedStartUtc);
              }}
            />
          </section>
        </main>
      </div>
    </div>
  );
}