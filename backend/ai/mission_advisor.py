from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class ActionRecommendation(BaseModel):
    priority: int = Field(description="Priority order starting from 1")
    action_title: str = Field(description="Short action headline")
    parameter_adjustment: str = Field(description="Specific parameter change, e.g., Shift launch +14 days or Increase battery to 7500 Wh")
    expected_impact: str = Field(description="Quantitative impact on SOC, illumination, or DTE blackout")


class AdvisorStructuredOutput(BaseModel):
    executive_verdict: str = Field(description="2-sentence executive summary of mission viability")
    root_cause_diagnosis: List[str] = Field(description="Specific physical bottlenecks identified from simulation metrics")
    actionable_recommendations: List[ActionRecommendation] = Field(description="Concrete replanning steps ranked by impact")
    tradeoff_notes: str = Field(description="Key engineering trade-offs or operational constraints")


SYSTEM_INSTRUCTION = """
You are a senior NASA Flight Dynamics & Lunar Surface Systems Engineer advising on South Pole lunar lander missions (Artemis-class).
Analyze the provided deterministic physics simulation metrics (from NAIF SPICE DE421/MOON_ME and NASA LOLA 5m DEM ray-traced horizons).
Rules:
1. Ground every diagnosis strictly in the numbers provided (Battery SOC %, Unmet Load Wh, Sun Visible Fraction %, DTE Blackout hours).
2. Remember the lunar synodic day-night cycle is ~29.5 Earth days (~708 hours). If solar illumination is near 0% for multi-day durations, the site is in local lunar night; shifting launch by ~14 days (~336–354 hours) rotates the Sun ~180 degrees into local daytime.
3. Provide concrete, quantitative engineering recommendations (e.g., exact battery capacity increase needed to bridge unmet load, schedule shift delta-t, or site switch).
4. Never fabricate telemetry not present in the input. Keep language concise, technical, and decisive.
"""


def _build_offline_fallback_advice(
    scenario_a: Dict[str, Any],
    scenario_b: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic engineering copilot fallback if GEMINI_API_KEY is not configured."""
    summary_a = scenario_a.get("summary", scenario_a)
    site_a = scenario_a.get("site", {})
    site_name = site_a.get("name", site_a.get("site_id", "Primary Site"))

    min_soc = float(summary_a.get("min_battery_soc_pct", 20.0))
    unmet_wh = float(summary_a.get("total_unmet_load_wh", 0.0))
    sun_pct = float(summary_a.get("mean_sun_visible_pct", summary_a.get("mean_sun_visible_fraction", 0.0) * 100.0))
    blackout_h = float(summary_a.get("max_dte_blackout_hours", 0.0))

    diagnoses: List[str] = []
    actions: List[ActionRecommendation] = []
    prio = 1

    if sun_pct < 15.0:
        diagnoses.append(
            f"Severe solar shadowing at {site_name} (mean solar visibility {sun_pct:.1f}%), indicating local lunar night or high crater rim obstruction."
        )
        actions.append(
            ActionRecommendation(
                priority=prio,
                action_title="Shift Launch Window by Half Synodic Cycle (+14 Days)",
                parameter_adjustment="Advance start_utc by +336 hours (+14 days) or run Bayesian Window Optimizer",
                expected_impact="Rotates solar azimuth by ~180° to place the landing site in sustained polar illumination.",
            )
        )
        prio += 1

    if unmet_wh > 0.0 or min_soc <= 20.5:
        extra_wh = round(unmet_wh * 1.25 + 1000.0, -2)
        diagnoses.append(
            f"Energy deficit detected: battery reached minimum threshold ({min_soc:.1f}% SOC) with {unmet_wh:.1f} Wh of unmet load."
        )
        actions.append(
            ActionRecommendation(
                priority=prio,
                action_title="Upgrade Energy Storage or Shed Non-Essential Load",
                parameter_adjustment=f"Increase battery_capacity_wh by +{extra_wh:.0f} Wh or reduce nominal_load_w by 20%",
                expected_impact="Prevents deep-discharge cutoff during eclipse intervals.",
            )
        )
        prio += 1

    if blackout_h > 24.0:
        diagnoses.append(
            f"Extended Direct-to-Earth (DTE) communication blackout of {blackout_h:.1f} hours caused by Earth elevation dipping below local topography."
        )
        actions.append(
            ActionRecommendation(
                priority=prio,
                action_title="Evaluate High-Ridge Alternative Site or Lunar Relay",
                parameter_adjustment="Compare against Connecting Ridge (Site01) / Shackleton Rim (Site04) or schedule store-and-forward telemetry",
                expected_impact="Reduces continuous DTE blackout windows below 24 hours.",
            )
        )
        prio += 1

    if not diagnoses:
        diagnoses.append(
            f"Nominal power and DTE link margins satisfied at {site_name} (min SOC {min_soc:.1f}%, max DTE blackout {blackout_h:.1f} h)."
        )
        actions.append(
            ActionRecommendation(
                priority=1,
                action_title="Proceed to Ensemble Terrain-Clone Verification",
                parameter_adjustment="Run /api/evaluate-ensemble across 10 LOLA DEM clones",
                expected_impact="Confirms mission remains FEASIBLE under ±Z topographic elevation uncertainty.",
            )
        )

    verdict = (
        f"Scenario A at {site_name} shows {min_soc:.1f}% minimum SOC and {blackout_h:.1f}h max DTE blackout. "
        + ("Replanning is required due to energy/illumination constraints." if unmet_wh > 0 or sun_pct < 15.0 else "Baseline trajectory meets primary power feasibility thresholds.")
    )

    if scenario_b:
        summary_b = scenario_b.get("summary", scenario_b)
        soc_b = float(summary_b.get("min_battery_soc_pct", 0.0))
        verdict += f" Comparison Scenario B achieves {soc_b:.1f}% minimum SOC."

    return {
        "data_label": "AI-ESTIMATED (Flight Dynamics Heuristic Copilot - Set GEMINI_API_KEY for Live LLM)",
        "model_used": "deterministic-flight-dynamics-fallback",
        "advice": {
            "executive_verdict": verdict,
            "root_cause_diagnosis": diagnoses,
            "actionable_recommendations": [a.model_dump() for a in actions],
            "tradeoff_notes": "Increasing battery mass reduces payload capacity; shifting launch date affects Earth-Moon transfer trajectory windows.",
        },
    }


def generate_mission_advice(
    scenario_a: Dict[str, Any],
    scenario_b: Optional[Dict[str, Any]] = None,
    mission_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates AI engineering replanning recommendations using Gemini API,
    with automatic fallback to deterministic flight-dynamics heuristics.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not GENAI_AVAILABLE or not api_key:
        return _build_offline_fallback_advice(scenario_a, scenario_b)

    payload = {
        "scenario_a_baseline": scenario_a,
        "scenario_b_comparison": scenario_b,
        "mission_constraints": mission_constraints or {"max_dte_blackout_hours": 24.0},
    }

    prompt = (
        "Analyze the following lunar South Pole mission simulation results and provide "
        "concrete flight-dynamics replanning recommendations:\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=AdvisorStructuredOutput,
                temperature=0.2,
            ),
        )
        parsed = json.loads(response.text)
        return {
            "data_label": "AI-ESTIMATED (Gemini 2.5 Flash Mission Decision Copilot)",
            "model_used": "gemini-2.5-flash",
            "advice": parsed,
        }
    except Exception as exc:
        fallback = _build_offline_fallback_advice(scenario_a, scenario_b)
        fallback["llm_error_note"] = f"Fallback triggered: {str(exc)}"
        return fallback