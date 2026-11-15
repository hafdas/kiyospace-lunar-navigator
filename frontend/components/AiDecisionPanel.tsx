"use client";

import React, { useState } from "react";

interface OptimizationRecommendation {
  rank: number;
  site_id: string;
  site_name: string;
  lat_deg: number;
  lon_deg: number;
  recommended_start_utc: string;
  offset_hours_from_base: number;
  strict_status: string;
  optimization_score: number;
  min_battery_soc_pct: number;
  final_battery_soc_pct: number;
  mean_sun_visible_pct: number;
  total_pv_generation_wh: number;
  max_dte_blackout_hours: number;
  dte_availability_pct: number;
  data_label: string;
}

interface OptimizeWindowResponse {
  search_space: {
    base_start_utc: string;
    search_window_days: number;
    candidate_sites: string[];
    trials_executed: number;
    unique_windows_evaluated: number;
  };
  recommendations: OptimizationRecommendation[];
  provenance: {
    optimizer: string;
    physics_solver: string;
    compliance_note: string;
  };
}

interface ActionRecommendation {
  priority: number;
  action_title: string;
  parameter_adjustment: string;
  expected_impact: string;
}

interface AiAdvisorResponse {
  data_label: string;
  model_used: string;
  advice: {
    executive_verdict: string;
    root_cause_diagnosis: string[];
    actionable_recommendations: ActionRecommendation[];
    tradeoff_notes: string;
  };
}

export interface AiDecisionPanelProps {
  baseStartUtc: string;
  durationHours: number;
  pvAreaM2?: number;
  batteryCapacityWh?: number;
  nominalLoadW?: number;
  scenarioAResult: object | null;
  scenarioBResult?: object | null;
  onApplyToScenarioB?: (siteId: string, startUtc: string) => void;
}

export default function AiDecisionPanel({
  baseStartUtc,
  durationHours,
  pvAreaM2 = 4.0,
  batteryCapacityWh = 5000.0,
  nominalLoadW = 250.0,
  scenarioAResult,
  scenarioBResult = null,
  onApplyToScenarioB,
}: AiDecisionPanelProps) {
  // Option 1 State: Bayesian Window & Site Optimizer
  const [searchDays, setSearchDays] = useState<number>(30);
  const [targetSite, setTargetSite] = useState<string>("all");
  const [nTrials, setNTrials] = useState<number>(36);
  const [optLoading, setOptLoading] = useState<boolean>(false);
  const [optError, setOptError] = useState<string | null>(null);
  const [optData, setOptData] = useState<OptimizeWindowResponse | null>(null);

  // Option 2 State: AI Mission Replanning Advisor
  const [advLoading, setAdvLoading] = useState<boolean>(false);
  const [advError, setAdvError] = useState<string | null>(null);
  const [advData, setAdvData] = useState<AiAdvisorResponse | null>(null);

  const handleRunOptimizer = async () => {
    setOptLoading(true);
    setOptError(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/api/optimize-window", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_start_utc: baseStartUtc,
          search_window_days: searchDays,
          duration_hours: durationHours,
          step_minutes: 60.0,
          target_site_id: targetSite === "all" ? null : targetSite,
          n_trials: nTrials,
          top_k: 3,
          pv_area_m2: pvAreaM2,
          battery_capacity_wh: batteryCapacityWh,
          nominal_load_w: nominalLoadW,
          max_allowed_blackout_hours: 24.0,
        }),
      });

      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(errPayload.detail || `HTTP ${response.status}`);
      }

      const json: OptimizeWindowResponse = await response.json();
      setOptData(json);
    } catch (err: unknown) {
      setOptError(
        err instanceof Error ? err.message : "Failed to run Bayesian optimizer."
      );
    } finally {
      setOptLoading(false);
    }
  };

  const handleRunAdvisor = async () => {
    if (!scenarioAResult) {
      setAdvError(
        "Please run 'Evaluate & Compare Scenarios' first so the AI Advisor has physics telemetry to diagnose."
      );
      return;
    }

    setAdvLoading(true);
    setAdvError(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/api/ai-advisor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_a: scenarioAResult,
          scenario_b: scenarioBResult,
          mission_constraints: { max_dte_blackout_hours: 24.0 },
        }),
      });

      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(errPayload.detail || `HTTP ${response.status}`);
      }

      const json: AiAdvisorResponse = await response.json();
      setAdvData(json);
    } catch (err: unknown) {
      setAdvError(
        err instanceof Error ? err.message : "Failed to generate AI mission advice."
      );
    } finally {
      setAdvLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    if (status === "FEASIBLE") {
      return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    }
    if (status === "UNRESOLVED") {
      return "bg-amber-500/20 text-amber-300 border-amber-500/40";
    }
    return "bg-rose-500/20 text-rose-300 border-rose-500/40";
  };

  return (
    <section className="bg-slate-900/90 border border-indigo-500/30 rounded-xl p-5 space-y-5 shadow-xl">
      {/* Section Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
              AI Decision Support
            </span>
            <h2 className="text-lg font-bold text-white">
              Autonomous Launch Optimizer & Flight Dynamics Copilot
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Physics-grounded Bayesian search (Optuna TPE) and AI root-cause replanning advisor for South Pole missions.
          </p>
        </div>
        <span className="text-[11px] px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
          Hybrid Physics + AI Architecture
        </span>
      </div>

      {/* Two-Column AI Tools Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ============================================================== */}
        {/* OPTION 1: BAYESIAN LAUNCH WINDOW & SITE OPTIMIZER              */}
        {/* ============================================================== */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-bold text-cyan-300">
                1. Bayesian Launch Window & Site Optimizer
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                SIMULATED (AI-Guided)
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Automatically searches across dates and candidate sites using NAIF SPICE + LOLA 5m DEM to maximize minimum battery SOC and DTE visibility.
            </p>

            {/* Optimizer Controls */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Search Window
                </label>
                <select
                  value={searchDays}
                  onChange={(e) => setSearchDays(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value={15}>+15 Days</option>
                  <option value={30}>+30 Days (1 Lunar Cycle)</option>
                  <option value={45}>+45 Days</option>
                  <option value={60}>+60 Days</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Target Site
                </label>
                <select
                  value={targetSite}
                  onChange={(e) => setTargetSite(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Sites (Auto-Select)</option>
                  <option value="site01">Site01 (Connecting Ridge)</option>
                  <option value="site04">Site04 (Shackleton Rim)</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">
                  Bayesian Trials
                </label>
                <select
                  value={nTrials}
                  onChange={(e) => setNTrials(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value={24}>24 Trials (Fast)</option>
                  <option value={36}>36 Trials (Balanced)</option>
                  <option value={60}>60 Trials (Deep)</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleRunOptimizer}
              disabled={optLoading}
              className="w-full py-2 px-4 rounded-lg text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-400 text-white transition cursor-pointer shadow"
            >
              {optLoading
                ? "Running Bayesian Optimization over SPICE..."
                : "Find Top 3 Optimal Launch Windows"}
            </button>

            {optError && (
              <div className="p-2.5 rounded bg-rose-950/60 border border-rose-800 text-xs text-rose-300">
                {optError}
              </div>
            )}

            {/* Optimizer Results */}
            {optData && (
              <div className="space-y-2.5 pt-2">
                <div className="flex items-center justify-between text-[11px] text-slate-400">
                  <span>
                    Evaluated {optData.search_space.unique_windows_evaluated} unique windows
                  </span>
                  <span>Base: {optData.search_space.base_start_utc.slice(0, 10)}</span>
                </div>

                {optData.recommendations.map((rec) => (
                  <div
                    key={`${rec.site_id}-${rec.recommended_start_utc}`}
                    className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-300 text-xs font-bold flex items-center justify-center border border-cyan-500/40">
                          #{rec.rank}
                        </span>
                        <span className="text-xs font-bold text-white">
                          {rec.site_name}
                        </span>
                      </div>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded border font-semibold ${getStatusBadge(
                          rec.strict_status
                        )}`}
                      >
                        {rec.strict_status}
                      </span>
                    </div>

                    <div className="text-xs text-cyan-300 font-mono">
                      Start: {rec.recommended_start_utc} (+{rec.offset_hours_from_base}h)
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-[11px] bg-slate-950/80 p-2 rounded border border-slate-800/80">
                      <div>
                        <span className="text-slate-400 block">Min SOC</span>
                        <span className="font-semibold text-white">
                          {rec.min_battery_soc_pct}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">Mean Sun</span>
                        <span className="font-semibold text-amber-300">
                          {rec.mean_sun_visible_pct}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">Max Blackout</span>
                        <span className="font-semibold text-white">
                          {rec.max_dte_blackout_hours}h
                        </span>
                      </div>
                    </div>

                    {onApplyToScenarioB && (
                      <button
                        onClick={() =>
                          onApplyToScenarioB(rec.site_id, rec.recommended_start_utc)
                        }
                        className="w-full py-1.5 px-3 rounded text-[11px] font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 transition cursor-pointer"
                      >
                        Apply Window to Scenario B &rarr;
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ============================================================== */}
        {/* OPTION 2: AI MISSION REPLANNING ADVISOR (DECISION COPILOT)     */}
        {/* ============================================================== */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-bold text-indigo-300">
                2. AI Mission Replanning Advisor
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                AI-ESTIMATED
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Diagnoses energy deficits, crater shadowing, and DTE link blackouts from your active simulation results and generates ranked engineering fixes.
            </p>

            <button
              onClick={handleRunAdvisor}
              disabled={advLoading}
              className="w-full py-2 px-4 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-400 text-white transition cursor-pointer shadow"
            >
              {advLoading
                ? "Analyzing Simulation Telemetry..."
                : "Diagnose & Generate Replanning Strategy"}
            </button>

            {advError && (
              <div className="p-2.5 rounded bg-rose-950/60 border border-rose-800 text-xs text-rose-300">
                {advError}
              </div>
            )}

            {/* Advisor Structured Output */}
            {advData && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between text-[10px] text-indigo-300 bg-indigo-950/40 px-2.5 py-1 rounded border border-indigo-800/50">
                  <span>{advData.data_label}</span>
                  <span className="font-mono">{advData.model_used}</span>
                </div>

                {/* Executive Verdict */}
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-indigo-400 block mb-1">
                    Executive Verdict
                  </span>
                  <p className="text-xs text-slate-200 leading-relaxed">
                    {advData.advice.executive_verdict}
                  </p>
                </div>

                {/* Root Cause Diagnosis */}
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1.5">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-amber-400 block">
                    Root-Cause Bottlenecks
                  </span>
                  <ul className="list-disc list-inside space-y-1 text-xs text-slate-300">
                    {advData.advice.root_cause_diagnosis.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                {/* Ranked Actionable Recommendations */}
                <div className="space-y-2">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-emerald-400 block">
                    Recommended Engineering Actions
                  </span>
                  {advData.advice.actionable_recommendations.map((act) => (
                    <div
                      key={act.priority}
                      className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1"
                    >
                      <div className="flex items-center gap-2">
                        <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                          P{act.priority}
                        </span>
                        <span className="text-xs font-bold text-white">
                          {act.action_title}
                        </span>
                      </div>
                      <p className="text-[11px] font-mono text-cyan-300">
                        {act.parameter_adjustment}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        Impact: {act.expected_impact}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Tradeoff Notes */}
                <p className="text-[11px] text-slate-400 italic border-t border-slate-800 pt-2">
                  Trade-off Note: {advData.advice.tradeoff_notes}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}