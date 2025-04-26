# scenario_runner.py  ─────────────────────────────────────────────────────────
"""
GDP Baseline + Scenario Runner (Executive Payload Edition)
----------------------------------------------------------
Produces five artifacts in ./outputs :

  • gdp_impact_summary.csv   – national Δ 2025-30
  • sector_summary.csv       – sector GDP shares 2025-30
  • topline_gdp.csv          – region GDP 2025-30
  • trace.json               – prompt + raw agent JSON
  • frontend.json            – executive-level payload for Streamlit

All analytics are restricted to years 2025-2030.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import agent                    # local module – Streams Gemini JSON
import baseline_model as bm     # local module – Cobb-Douglas helpers

# ────────────────────────────────────────────────────────────────── constants
YR_START, YR_END = 2025, 2030          # Dashboard window
N_TOP = 3                              # #regional winners / losers to surface

# ───────────────────────────────────────────────────────── helper functions
def capture_agent_json(prompt: str) -> Dict[str, Any]:
    """Call agent.generate(prompt) and return the exact JSON it streams."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.generate(prompt)
    raw = buf.getvalue().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        sys.exit(f"[ERROR] Could not parse agent JSON:\n{raw}\n{err}")

    required = {"rules", "reasoning"}
    if not required.issubset(data.keys()):
        sys.exit(f"[ERROR] Agent JSON missing keys {required - data.keys()}")
    return data


def apply_rules(kpi_df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a KPI-wide dataframe with percentage shocks applied."""
    df = kpi_df.copy()
    for rule in rules:
        region, sector, kpi_key = rule["region"], rule["sector"], rule["kpi"]
        years = {int(y): v for y, v in rule["years"].items()}

        mask = pd.Series(True, index=df.index)
        if region.lower() != "all":
            mask &= df["regioncode"] == region
        if sector.lower() != "all":
            mask &= df["SectorCode"] == sector

        targets = ["Productivity", "Investments", "Workforce"] \
            if kpi_key.lower() == "all" else [kpi_key]

        for year, pct in years.items():
            yr_mask = mask & (df["year"] == year)
            if not yr_mask.any():
                continue
            for col in targets:
                df.loc[yr_mask, col] *= 1.0 + pct / 100.0
    return df


def compare_nat(base_nat: pd.DataFrame,
                scn_nat: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline & scenario national GDP and compute deltas."""
    merged = base_nat.merge(
        scn_nat,
        on="year",
        suffixes=("_baseline", "_scenario"),
    )
    merged["delta_abs"] = merged["Saudi_GDP_SAR_scenario"] \
                        - merged["Saudi_GDP_SAR_baseline"]
    merged["delta_pct"] = 100 * merged["delta_abs"] \
                        / merged["Saudi_GDP_SAR_baseline"]
    return merged.query("@YR_START <= year <= @YR_END")


def make_sector_summary(base_sec: pd.DataFrame,
                        scn_sec: pd.DataFrame,
                        base_nat: pd.DataFrame,
                        scn_nat: pd.DataFrame) -> pd.DataFrame:
    """Sector GDP shares baseline vs scenario, 2025-30."""
    base_agg = (base_sec.groupby(["SectorCode", "year"], as_index=False)
                        .GDP_SAR.sum()
                        .rename(columns={"GDP_SAR": "GDP_SAR_baseline"}))
    scn_agg  = (scn_sec.groupby(["SectorCode", "year"], as_index=False)
                        .GDP_SAR.sum()
                        .rename(columns={"GDP_SAR": "GDP_SAR_scenario"}))

    df = base_agg.merge(scn_agg, on=["SectorCode", "year"])

    nat_base = dict(zip(base_nat["year"], base_nat["Saudi_GDP_SAR"]))
    nat_scn  = dict(zip(scn_nat["year"],  scn_nat["Saudi_GDP_SAR"]))

    df["baseline_pct"] = 100 * df["GDP_SAR_baseline"] / df["year"].map(nat_base)
    df["scenario_pct"] = 100 * df["GDP_SAR_scenario"] / df["year"].map(nat_scn)
    df["Δ_pct_pts"]    = df["scenario_pct"] - df["baseline_pct"]

    return df.query("@YR_START <= year <= @YR_END")


def make_topline_gdp(base_reg: pd.DataFrame,
                     scn_reg: pd.DataFrame) -> pd.DataFrame:
    """Region GDP baseline vs scenario, 2025-30 (+ TOTAL rows)."""
    reg = base_reg.rename(columns={"Region_GDP_SAR": "GDP_SAR_baseline"}) \
            .merge(scn_reg.rename(columns={"Region_GDP_SAR":
                                           "GDP_SAR_scenario"}),
                   on=["regioncode", "year"])
    return reg.query("@YR_START <= year <= @YR_END")


# ───────────────────────────────── executive-payload constructor
def build_frontend_payload(meta: Dict[str, Any],
                           nat_yearly: pd.DataFrame,
                           sector_summary: pd.DataFrame,
                           topline_gdp: pd.DataFrame,
                           trace: Dict[str, Any]) -> Dict[str, Any]:
    # -------- headline KPIs ------------------------------------------
    g25 = nat_yearly[nat_yearly["year"] == 2025].iloc[0]
    g30 = nat_yearly[nat_yearly["year"] == 2030].iloc[0]

    def cagr(v0, v1, years=5):
        return (float(v1) / float(v0))**(1 / years) - 1.0

    headline = {
        "baseline_2030":   int(g30["Saudi_GDP_SAR_baseline"]),
        "scenario_2030":   int(g30["Saudi_GDP_SAR_scenario"]),
        "delta_abs_2030":  int(g30["delta_abs"]),
        "delta_pct_2030":  round(float(g30["delta_pct"]), 2),
        "cagr_baseline":   round(cagr(g25["Saudi_GDP_SAR_baseline"],
                                      g30["Saudi_GDP_SAR_baseline"])*100, 2),
        "cagr_scenario":   round(cagr(g25["Saudi_GDP_SAR_scenario"],
                                      g30["Saudi_GDP_SAR_scenario"])*100, 2),
        "cumulative_delta_25_30": int(nat_yearly["delta_abs"].sum()),
        "worst_year": {
            "year": int(nat_yearly.loc[nat_yearly['delta_pct'].idxmin(), "year"]),
            "delta_pct": round(float(nat_yearly["delta_pct"].min()), 2)
        }
    }

    # -------- sector mix in 2030 -------------------------------------
    sector_30 = sector_summary[sector_summary["year"] == 2030] \
                   .rename(columns={"SectorCode": "sector",
                                    "baseline_pct": "baseline_pct",
                                    "scenario_pct": "scenario_pct",
                                    "Δ_pct_pts": "delta_pp"}) \
                   .round(2) \
                   .sort_values("delta_pp", ascending=True) \
                   .to_dict("records")

    # -------- regional winners / losers in 2030 ----------------------
    reg30 = topline_gdp[topline_gdp["year"] == 2030].copy()
    reg30 = reg30[reg30["regioncode"] != "TOTAL"]
    reg30["delta_abs"] = reg30["GDP_SAR_scenario"] - reg30["GDP_SAR_baseline"]
    reg30["delta_pct"] = np.where(reg30["GDP_SAR_baseline"] == 0,
                                  np.nan,
                                  100 * reg30["delta_abs"] / reg30["GDP_SAR_baseline"])

    losers = reg30.nsmallest(N_TOP, "delta_abs") \
                  .rename(columns={"regioncode": "region"}) \
                  .round({"delta_pct": 2}) \
                  .to_dict("records")
    winners = reg30.nlargest(N_TOP, "delta_abs") \
                   .rename(columns={"regioncode": "region"}) \
                   .round({"delta_pct": 2}) \
                   .to_dict("records")

    # -------- yearly national series ---------------------------------
    yearly = nat_yearly.rename(
                columns={"Saudi_GDP_SAR_baseline": "baseline",
                         "Saudi_GDP_SAR_scenario": "scenario"}) \
              .round({"delta_pct": 2}) \
              .assign(baseline=lambda d: d["baseline"].astype(int),
                      scenario=lambda d: d["scenario"].astype(int),
                      delta_abs=lambda d: d["delta_abs"].astype(int)) \
              .to_dict("records")

    return {
        "meta": meta,
        "national_summary": headline,
        "sector_breakdown_2030": sector_30,
        "regional_impacts_2030": {
            "top_negative": losers,
            "top_positive": winners
        },
        "yearly_national": yearly,
        "trace": trace
    }

# ──────────────────────────────────────────────────────────── main routine
def main(input_dir: str = "inputs") -> None:
    prompt = input("\nEnter your policy scenario → ").strip()
    if not prompt:
        sys.exit("Empty prompt. Aborting.")

    print("\n· Generating impact-rules with Gemini …")
    agent_json = capture_agent_json(prompt)
    rules, reasoning = agent_json["rules"], agent_json["reasoning"]

    kpi_csv = os.path.join(input_dir, "input_data.csv")
    if not os.path.exists(kpi_csv):
        sys.exit(f"[ERROR] KPI file not found: {kpi_csv}")

    # ---------------- Baseline
    print("· Computing baseline GDP …")
    kpis_wide = bm.load_kpis(kpi_csv)
    sec_base  = bm.compute_sector_gdp(kpis_wide)
    reg_base  = bm.aggregate_region(sec_base)
    nat_base  = bm.aggregate_country(reg_base)

    # ---------------- Scenario
    print("· Applying rules & computing scenario GDP …")
    kpis_scn  = apply_rules(kpis_wide, rules)
    sec_scn   = bm.compute_sector_gdp(kpis_scn)
    reg_scn   = bm.aggregate_region(sec_scn)
    nat_scn   = bm.aggregate_country(reg_scn)

    # ---------------- Analytics
    nat_yearly     = compare_nat(nat_base, nat_scn)
    sector_summary = make_sector_summary(sec_base, sec_scn, nat_base, nat_scn)
    topline_gdp    = make_topline_gdp(reg_base, reg_scn)

    # ---------------- Console quick view
    pd.options.display.float_format = "{:,.0f}".format
    print("\n────────────────────────────────────────────────────────────")
    print("EXECUTIVE SUMMARY (SAR, 2025-2030)")
    print(nat_yearly.to_string(index=False))
    print("────────────────────────────────────────────────────────────")
    print("Agent rationale:")
    print(reasoning.strip())
    print("────────────────────────────────────────────────────────────")

    # ---------------- Save CSVs + trace
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    nat_yearly.to_csv   (f"{out_dir}/gdp_impact_summary.csv", index=False)
    sector_summary.to_csv(f"{out_dir}/sector_summary.csv", index=False)
    topline_gdp.to_csv   (f"{out_dir}/topline_gdp.csv",  index=False)
    with open(f"{out_dir}/trace.json", "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "agent_output": agent_json},
                  f, indent=2, ensure_ascii=False)

    # ---------------- Build & save frontend payload
    meta = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "model": "gemini-2.0-flash",
        "alpha": bm.ALPHA,
        "hours_per_worker": bm.HOURS_PER_WORKER
    }
    exec_payload = build_frontend_payload(meta,
                                          nat_yearly,
                                          sector_summary,
                                          topline_gdp,
                                          agent_json)
    with open(f"{out_dir}/frontend.json", "w", encoding="utf-8") as f:
        json.dump(exec_payload, f, indent=2, ensure_ascii=False)

    # ---------------- Done
    print("\n✔  Files saved to 'outputs':")
    for fn in ("gdp_impact_summary.csv", "sector_summary.csv",
               "topline_gdp.csv", "trace.json", "frontend.json"):
        print("   •", fn)


def run_scenario(prompt: str, input_dir: str = "inputs", output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Runs the GDP scenario based on the given prompt.
    Generates output files (CSV, trace.json) and returns the executive payload.
    Raises FileNotFoundError if input data is missing.
    Raises Exception for other errors during execution.
    """
    print(f"\n· Running scenario for prompt: '{prompt}'")
    print("· Generating impact-rules with Gemini …")
    try:
        agent_json = capture_agent_json(prompt) # Assuming capture_agent_json is defined elsewhere
    except Exception as e:
        print(f"[ERROR] Agent generation failed: {e}", file=sys.stderr)
        raise RuntimeError(f"Agent generation failed: {e}") from e # Raise specific error

    rules, reasoning = agent_json.get("rules", []), agent_json.get("reasoning", "No reasoning provided.") # Handle potential missing keys

    kpi_csv = os.path.join(input_dir, "input_data.csv")
    if not os.path.exists(kpi_csv):
        print(f"[ERROR] KPI file not found: {kpi_csv}", file=sys.stderr)
        raise FileNotFoundError(f"Input data file not found: {kpi_csv}. Please ensure 'inputs/input_data.csv' exists.")

    try:
        # ---------------- Baseline
        print("· Computing baseline GDP …")
        kpis_wide = bm.load_kpis(kpi_csv) # Assuming bm is baseline_model
        sec_base  = bm.compute_sector_gdp(kpis_wide)
        reg_base  = bm.aggregate_region(sec_base)
        nat_base  = bm.aggregate_country(reg_base)

        # ---------------- Scenario
        print("· Applying rules & computing scenario GDP …")
        kpis_scn  = apply_rules(kpis_wide, rules) # Assuming apply_rules is defined elsewhere
        sec_scn   = bm.compute_sector_gdp(kpis_scn)
        reg_scn   = bm.aggregate_region(sec_scn) # Corrected
        nat_scn   = bm.aggregate_country(reg_scn)

        # ---------------- Analytics
        print("· Computing analytics …")
        nat_yearly     = compare_nat(nat_base, nat_scn) # Assuming compare_nat is defined elsewhere
        sector_summary = make_sector_summary(sec_base, sec_scn, nat_base, nat_scn) # Assuming make_sector_summary is defined elsewhere
        topline_gdp    = make_topline_gdp(reg_base, reg_scn) # Assuming make_topline_gdp is defined elsewhere

        # ---------------- Save CSVs + trace
        print(f"· Saving output files to '{output_dir}' …")
        os.makedirs(output_dir, exist_ok=True)
        nat_yearly.to_csv   (f"{output_dir}/gdp_impact_summary.csv", index=False)
        sector_summary.to_csv(f"{output_dir}/sector_summary.csv", index=False)
        topline_gdp.to_csv   (f"{output_dir}/topline_gdp.csv",  index=False)
        with open(f"{output_dir}/trace.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "agent_output": agent_json},
                      f, indent=2, ensure_ascii=False)

        # ---------------- Build & return frontend payload
        print("· Building frontend payload …")
        meta = {
            "run_id": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "model": "gemini-2.0-flash", # Or whatever model you are using
            "alpha": bm.ALPHA, # Assuming ALPHA is a constant in baseline_model
            "hours_per_worker": bm.HOURS_PER_WORKER # Assuming HOURS_PER_WORKER is a constant
        }
        exec_payload = build_frontend_payload(meta,
                                              nat_yearly,
                                              sector_summary,
                                              topline_gdp,
                                              agent_json)

        print("✔ Scenario run complete.")
        return exec_payload

    except Exception as e:
        print(f"[ERROR] An error occurred during scenario computation: {e}", file=sys.stderr)
        raise RuntimeError(f"Scenario computation failed: {e}") from e # Catch other errors

