"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from "recharts";

export interface TimeSeriesStep {
  utc: string;
  sun_azimuth_deg: number;
  sun_elevation_deg: number;
  sun_horizon_deg: number;
  sun_visible_fraction: number;
  pv_power_w: number;
  load_power_w: number;
  battery_energy_wh: number;
  battery_soc: number;
  curtailment_wh: number;
  unmet_load_wh: number;
  earth_azimuth_deg: number;
  earth_elevation_deg: number;
  earth_horizon_deg: number;
  dte_margin_deg: number;
  dte_link_active: boolean;
}

export interface MissionEvaluationResponse {
  run_id: string;
  site: {
    site_id: "site01" | "site04" | "custom";
    name: string;
    data_label: string;
    lat_deg: number;
    lon_deg: number;
    ground_elevation_m: number;
    observer_height_m: number;
  };
  horizon_profile: {
    azimuths_deg: number[];
    horizon_elevations_deg: number[];
  };
  summary: {
    feasibility_status: "FEASIBLE" | "UNRESOLVED" | "INFEASIBLE";
    duration_hours: number;
    step_minutes: number;
    mean_sun_visible_fraction: number;
    total_pv_generation_wh: number;
    total_load_demand_wh: number;
    total_curtailment_wh: number;
    total_unmet_load_wh: number;
    min_battery_soc_pct: number;
    final_battery_soc_pct: number;
    energy_balance_residual_pct: number;
    dte_availability_pct: number;
    max_dte_blackout_hours: number;
    earthrise_timestamps: string[];
    earthset_timestamps: string[];
  };
  time_series: TimeSeriesStep[];
}

interface ScenarioChartsProps {
  label: string;
  accentColor: string;
  data: MissionEvaluationResponse;
}

export default function ScenarioCharts({
  label,
  accentColor,
  data,
}: ScenarioChartsProps) {
  const [stepIndex, setStepIndex] = useState<number>(0);
  const currentStep =
    data.time_series[Math.min(stepIndex, data.time_series.length - 1)];

  // Prepare 360° horizon profile data with current Sun & Earth markers
  const horizonChartData = data.horizon_profile.azimuths_deg.map((az, idx) => {
    const isSunAz = Math.abs(az - Math.round(currentStep.sun_azimuth_deg) % 360) <= 1;
    const isEarthAz = Math.abs(az - Math.round(currentStep.earth_azimuth_deg) % 360) <= 1;
    return {
      azimuth: az,
      terrain_horizon_deg: data.horizon_profile.horizon_elevations_deg[idx],
      sun_el_deg: isSunAz ? Number(currentStep.sun_elevation_deg.toFixed(3)) : null,
      earth_el_deg: isEarthAz ? Number(currentStep.earth_elevation_deg.toFixed(3)) : null,
    };
  });

  // Prepare Mission Timeline data
  const timelineData = data.time_series.map((pt, idx) => ({
    hour: idx,
    utcShort: pt.utc.slice(5, 16).replace("T", " "),
    soc_pct: Number((pt.battery_soc * 100.0).toFixed(1)),
    pv_power_w: Number(pt.pv_power_w.toFixed(1)),
    dte_blackout_band: pt.dte_link_active ? 0 : 100,
  }));

  const statusBadgeColor =
    data.summary.feasibility_status === "FEASIBLE"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
      : data.summary.feasibility_status === "UNRESOLVED"
      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
      : "bg-rose-500/20 text-rose-300 border-rose-500/40";

return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4 shadow-lg">
      {/* Scenario Header & Provenance Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full inline-block"
              style={{ backgroundColor: accentColor }}
            />
            <h3 className="text-base font-bold text-white">{label}</h3>
            <span className={`text-xs px-2 py-0.5 rounded border font-semibold ${statusBadgeColor}`}>
              {data.summary.feasibility_status}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {data.site.name} ({data.site.lat_deg.toFixed(3)}°, {data.site.lon_deg.toFixed(2)}°) • Elev:{" "}
            {data.site.ground_elevation_m.toFixed(1)} m
          </p>
        </div>

        {/* Provenance Badge & Phase 6 Export Buttons */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
            {data.site.data_label}
          </span>
          <a
            href={`http://127.0.0.1:8000/api/results/${data.run_id}/export/csv`}
            download={`KiyoSpace_${data.site.site_id}_Report.csv`}
            className="text-[11px] px-2 py-0.5 rounded bg-cyan-900/50 text-cyan-300 border border-cyan-800 hover:bg-cyan-800 cursor-pointer transition"
          >
            ↓ CSV
          </a>
          <a
            href={`http://127.0.0.1:8000/api/results/${data.run_id}/export/json`}
            download={`KiyoSpace_${data.site.site_id}_Full.json`}
            className="text-[11px] px-2 py-0.5 rounded bg-amber-900/50 text-amber-300 border border-amber-800 hover:bg-amber-800 cursor-pointer transition"
          >
            ↓ JSON
          </a>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div
          className={`p-3 rounded-lg border ${
            data.summary.min_battery_soc_pct < 20.0
              ? "bg-rose-950/40 border-rose-600 text-rose-200"
              : "bg-slate-950/60 border-slate-800 text-slate-100"
          }`}
        >
          <p className="text-[11px] uppercase tracking-wider text-slate-400">
            Minimum Battery SOC
          </p>
          <p className="text-xl font-bold mt-1">
            {data.summary.min_battery_soc_pct.toFixed(1)}%
          </p>
          <p className="text-[11px] text-slate-400">
            Final: {data.summary.final_battery_soc_pct.toFixed(1)}%
          </p>
        </div>

        <div className="p-3 rounded-lg border bg-slate-950/60 border-slate-800">
          <p className="text-[11px] uppercase tracking-wider text-slate-400">
            Total Solar PV
          </p>
          <p className="text-xl font-bold text-amber-400 mt-1">
            {(data.summary.total_pv_generation_wh / 1000).toFixed(2)} kWh
          </p>
          <p className="text-[11px] text-slate-400">
            Mean Sun Vis: {(data.summary.mean_sun_visible_fraction * 100).toFixed(1)}%
          </p>
        </div>

        <div className="p-3 rounded-lg border bg-slate-950/60 border-slate-800">
          <p className="text-[11px] uppercase tracking-wider text-slate-400">
            Max DTE Blackout
          </p>
          <p className="text-xl font-bold text-cyan-400 mt-1">
            {data.summary.max_dte_blackout_hours.toFixed(1)} h
          </p>
          <p className="text-[11px] text-slate-400">
            DTE Avail: {data.summary.dte_availability_pct.toFixed(1)}%
          </p>
        </div>

        <div className="p-3 rounded-lg border bg-slate-950/60 border-slate-800">
          <p className="text-[11px] uppercase tracking-wider text-slate-400">
            Energy Balance Audit
          </p>
          <p className="text-xl font-bold text-emerald-400 mt-1">
            {data.summary.energy_balance_residual_pct.toFixed(4)}%
          </p>
          <p className="text-[11px] text-slate-400">
            Unmet Load: {data.summary.total_unmet_load_wh.toFixed(0)} Wh
          </p>
        </div>
      </div>

      {/* Interactive Time Scrubber */}
      <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-slate-300">
            Active Time Scrubber:{" "}
            <span className="text-cyan-400 font-mono">{currentStep.utc}</span>
          </span>
          <span className="text-slate-400">
            Sun Az/El: {currentStep.sun_azimuth_deg.toFixed(1)}° /{" "}
            {currentStep.sun_elevation_deg.toFixed(2)}° | Vis:{" "}
            {(currentStep.sun_visible_fraction * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, data.time_series.length - 1)}
          value={stepIndex}
          onChange={(e) => setStepIndex(Number(e.target.value))}
          className="w-full accent-cyan-400 cursor-pointer"
        />
      </div>

      {/* Chart 1: 360° Terrain Horizon Profile */}
      <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
          1. 360° Terrain Horizon Profile vs. Instantaneous Sun & Earth Elevation
        </h4>
        <div className="h-52 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={horizonChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="azimuth"
                type="number"
                domain={[0, 360]}
                tickCount={9}
                stroke="#94a3b8"
                fontSize={11}
                unit="°"
              />
              <YAxis
                domain={[-5, 10]}
                stroke="#94a3b8"
                fontSize={11}
                unit="°"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
              <ReferenceLine
                x={Math.round(currentStep.sun_azimuth_deg)}
                stroke="#facc15"
                label={{ value: "SUN", fill: "#facc15", fontSize: 10, position: "top" }}
              />
              <ReferenceLine
                x={Math.round(currentStep.earth_azimuth_deg)}
                stroke="#38bdf8"
                label={{ value: "EARTH", fill: "#38bdf8", fontSize: 10, position: "top" }}
              />
              <Area
                type="monotone"
                dataKey="terrain_horizon_deg"
                name="Terrain Horizon (deg)"
                stroke={accentColor}
                fill={accentColor}
                fillOpacity={0.25}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Mission Timeline (Battery SOC %, Solar PV W, DTE Blackout Shading) */}
      <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
          2. Mission Timeline — Battery SOC (%), Solar PV Power (W) & DTE Blackout Windows
        </h4>
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="utcShort"
                stroke="#94a3b8"
                fontSize={10}
                minTickGap={24}
              />
              <YAxis
                yAxisId="left"
                domain={[0, 100]}
                stroke="#38bdf8"
                fontSize={11}
                unit="%"
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#facc15"
                fontSize={11}
                unit="W"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              {/* Shaded red background regions where DTE link is in blackout */}
              <Area
                yAxisId="left"
                type="stepAfter"
                dataKey="dte_blackout_band"
                name="DTE Blackout Window"
                fill="#f43f5e"
                stroke="none"
                fillOpacity={0.18}
              />
              <ReferenceLine
                yAxisId="left"
                y={20}
                stroke="#f43f5e"
                strokeDasharray="4 4"
                label={{ value: "Min Safe SOC (20%)", fill: "#f43f5e", fontSize: 10 }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="soc_pct"
                name="Battery SOC (%)"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="pv_power_w"
                name="Solar PV Generation (W)"
                stroke="#facc15"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}