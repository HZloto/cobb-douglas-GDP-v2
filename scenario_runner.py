# scenario_runner.py  ──────────────────────────────────────────────────────────
"""
GDP Baseline + Scenario Runner with Observability
-------------------------------------------------
Outputs:
  • gdp_impact_summary.csv   (national totals Δ)
  • sector_summary.csv       (sector share Δ)
  • topline_gdp.csv          (region GDP baseline & scenario)
  • trace.json               (prompt + agent JSON)
  • frontend.json            (single payload for the Streamlit UI)

Filters everything to years 2025-2030, per product requirements.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd

import agent                              # agent.generate()
import baseline_model as bm               # Cobb–Douglas helpers

# ────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────
YR_START, YR_END = 2025, 2030  # <- UI must only see these years


def _capture_agent_json(prompt: str) -> Dict[str, Any]:
    """Run agent.generate(prompt) and return the JSON it streams."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.generate(prompt)
    raw = buf.getvalue().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        sys.exit(f"[ERROR] Could not parse agent JSON:\n{raw}\n{err}")
    if not {"rules", "reasoning"} <= data.keys():
        sys.exit("[ERROR] Agent JSON missing keys 'rules' or 'reasoning'.")
    return data


def _apply_rules(kpi_df: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return KPI-wide dataframe with rules applied."""
    df = kpi_df.copy()
    for rule in rules:
        region, sector, kpi_key = rule["region"], rule["sector"], rule["kpi"]
        years = {int(y): v for y, v in rule["years"].items()}

        mask = pd.Series(True, index=df.index)
        if region.lower() != "all":
            mask &= df["regioncode"] == region
        if sector.lower() != "all":
            mask &= df["SectorCode"] == sector

        targets = (
            ["Productivity", "Investments", "Workforce"]
            if kpi_key.lower() == "all" else [kpi_key]
        )

        for yr, pct in years.items():
            ym = mask & (df["year"] == yr)
            if not ym.any():
                continue
            for col in targets:
                df.loc[ym, col] *= 1.0 + pct / 100.0
    return df


def _compare_nat(base_nat: pd.DataFrame,
                 scn_nat: pd.DataFrame) -> pd.DataFrame:
    merged = base_nat.merge(
        scn_nat, on="year", suffixes=("_baseline", "_scenario")
    )
    merged["Δ_SAR"] = merged["Saudi_GDP_SAR_scenario"] - merged["Saudi_GDP_SAR_baseline"]
    merged["%_diff"] = 100 * merged["Δ_SAR"] / merged["Saudi_GDP_SAR_baseline"]
    return merged


# ────────────────────────────────────────────────────────────────────────────
# Extra observability helpers (unchanged but year-filter added)
# ────────────────────────────────────────────────────────────────────────────
def _make_sector_summary(base_sec: pd.DataFrame, scn_sec: pd.DataFrame,
                         base_nat: pd.DataFrame, scn_nat: pd.DataFrame) -> pd.DataFrame:
    base_agg = (base_sec.groupby(["SectorCode", "year"], as_index=False)
                        .agg(GDP_SAR_baseline=("GDP_SAR", "sum")))
    scn_agg  = (scn_sec.groupby(["SectorCode", "year"], as_index=False)
                       .agg(GDP_SAR_scenario=("GDP_SAR", "sum")))

    df = base_agg.merge(scn_agg, on=["SectorCode", "year"]).fillna(0)

    nat_base_map = dict(zip(base_nat["year"], base_nat["Saudi_GDP_SAR"]))
    nat_scn_map  = dict(zip(scn_nat["year"],  scn_nat["Saudi_GDP_SAR"]))

    df["baseline_pct"] = 100 * df["GDP_SAR_baseline"] / df["year"].map(nat_base_map)
    df["scenario_pct"] = 100 * df["GDP_SAR_scenario"] / df["year"].map(nat_scn_map)
    df["Δ_pct_pts"]    = df["scenario_pct"] - df["baseline_pct"]

    return df.query("@YR_START <= year <= @YR_END").sort_values(["year", "SectorCode"])


def _make_topline_gdp(base_reg: pd.DataFrame, scn_reg: pd.DataFrame,
                      base_nat: pd.DataFrame, scn_nat: pd.DataFrame) -> pd.DataFrame:
    rb = base_reg.rename(columns={"Region_GDP_SAR": "GDP_SAR_baseline"})
    rs = scn_reg.rename(columns={"Region_GDP_SAR": "GDP_SAR_scenario"})
    merged = rb.merge(rs, on=["regioncode", "year"]).fillna(0)

    nat_b = (base_nat.rename(columns={"Saudi_GDP_SAR": "GDP_SAR_baseline"})
                      .assign(regioncode="TOTAL"))
    nat_s = (scn_nat.rename(columns={"Saudi_GDP_SAR": "GDP_SAR_scenario"})
                     .assign(regioncode="TOTAL"))
    merged = pd.concat([merged, nat_b.merge(nat_s, on=["regioncode", "year"])],
                       ignore_index=True)

    return merged.query("@YR_START <= year <= @YR_END").sort_values(["regioncode", "year"])


def _save_trace(prompt: str, agent_json: Dict[str, Any], out_dir: str) -> None:
    with open(os.path.join(out_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "agent_output": agent_json}, f, indent=2, ensure_ascii=False)


def _build_frontend_payload(meta: Dict[str, Any],
                            nat_df: pd.DataFrame,
                            sector_df: pd.DataFrame,
                            region_df: pd.DataFrame,
                            trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict ready to dump to frontend.json"""
    nat_rec    = nat_df[["year", "baseline", "scenario", "delta_abs", "delta_pct"]].to_dict("records")
    sector_rec = sector_df.rename(columns={"SectorCode": "sector",
                                           "baseline_pct": "baseline_pct",
                                           "scenario_pct": "scenario_pct",
                                           "Δ_pct_pts": "delta_pp"})\
                          .to_dict("records")
    region_rec = region_df.rename(columns={"regioncode": "region",
                                           "GDP_SAR_baseline": "baseline",
                                           "GDP_SAR_scenario": "scenario"})\
                          .assign(delta_abs=lambda d: d["scenario"] - d["baseline"],
                                  delta_pct=lambda d: 100 * (d["scenario"] - d["baseline"]) / d["baseline"].replace(0, pd.NA))\
                          .to_dict("records")

    return {
        "meta": meta,
        "national_gdp": nat_rec,
        "sector_summary": sector_rec,
        "region_gdp": region_rec,
        "trace": trace
    }


# ────────────────────────────────────────────────────────────────────────────
def main(input_dir: str = "inputs") -> None:
    # ------------------------------------------------------------------ user prompt
    prompt = input("\nEnter your policy scenario → ").strip()
    if not prompt:
        sys.exit("Empty prompt. Aborting.")

    print("\n· Generating impact-rules with Gemini …")
    agent_json = _capture_agent_json(prompt)
    rules, reasoning = agent_json["rules"], agent_json["reasoning"]

    # ------------------------------------------------------------------ load KPI
    kpi_csv = os.path.join(input_dir, "input_data.csv")
    if not os.path.exists(kpi_csv):
        sys.exit(f"[ERROR] KPI file not found: {kpi_csv}")

    # Baseline ---------------------------------------------------------
    print("· Computing baseline GDP …")
    kpi_wide      = bm.load_kpis(kpi_csv)
    sector_base   = bm.compute_sector_gdp(kpi_wide)
    region_base   = bm.aggregate_region(sector_base)
    national_base = bm.aggregate_country(region_base)

    # Scenario ---------------------------------------------------------
    print("· Applying rules & computing scenario GDP …")
    kpi_scn      = _apply_rules(kpi_wide, rules)
    sector_scn   = bm.compute_sector_gdp(kpi_scn)
    region_scn   = bm.aggregate_region(sector_scn)
    national_scn = bm.aggregate_country(region_scn)

    impact_nat = _compare_nat(national_base, national_scn) \
                   .query("@YR_START <= year <= @YR_END")

    # Display summary --------------------------------------------------
    pd.options.display.float_format = "{:,.0f}".format
    print("\n────────────────────────────────────────────────────────────")
    print("SCENARIO IMPACT VS. BASELINE (SAR, 2025-2030)")
    print(impact_nat.to_string(index=False))
    print("────────────────────────────────────────────────────────────")
    print("Agent rationale:")
    print(reasoning.strip())
    print("────────────────────────────────────────────────────────────")

    # Observability tables (year filtered) -----------------------------
    sector_summary = _make_sector_summary(sector_base, sector_scn,
                                          national_base, national_scn)
    topline_gdp    = _make_topline_gdp(region_base, region_scn,
                                       national_base, national_scn)

    # Aliases for frontend payload -------------------------------------
    nat_front = impact_nat.rename(columns={
        "Saudi_GDP_SAR_baseline": "baseline",
        "Saudi_GDP_SAR_scenario": "scenario",
        "Δ_SAR": "delta_abs",
        "%_diff": "delta_pct"
    })

    # ------------------------------------------------------------------ write files
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    impact_nat.to_csv   (os.path.join(out_dir, "gdp_impact_summary.csv"), index=False)
    sector_summary.to_csv(os.path.join(out_dir, "sector_summary.csv"),     index=False)
    topline_gdp.to_csv   (os.path.join(out_dir, "topline_gdp.csv"),        index=False)
    _save_trace(prompt, agent_json, out_dir)

    # frontend.json ----------------------------------------------------
    meta_section = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "model": "gemini-2.0-flash",
        "alpha": bm.ALPHA,
        "hours_per_worker": bm.HOURS_PER_WORKER
    }
    front_payload = _build_frontend_payload(
        meta_section,
        nat_front,
        sector_summary,
        topline_gdp,
        agent_json
    )
    with open(os.path.join(out_dir, "frontend.json"), "w", encoding="utf-8") as f:
        json.dump(front_payload, f, indent=2, ensure_ascii=False)

    # Done -------------------------------------------------------------
    print(f"\n✔  Files saved to '{out_dir}':")
    for fn in ("gdp_impact_summary.csv", "sector_summary.csv", "topline_gdp.csv",
               "trace.json", "frontend.json"):
        print("   •", fn)


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GDP baseline + scenario impact analysis.")
    parser.add_argument("-i", "--input_dir", default="inputs",
                        help="Folder containing input_data.csv (default: inputs/)")
    main(parser.parse_args().input_dir)
