# app.py – interactive CLI for baseline vs scenario GDP simulations
# -----------------------------------------------------------------------------
"""Run ad‑hoc policy scenarios and compare them with the baseline
(including spillovers) generated from Cobb‑Douglas sector data.
"""

import argparse
import io
import json
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import agent           # LLM interface → JSON rules
import baseline_model  # computes raw Cobb‑Douglas baseline
import legacy.run_scenario as run_scenario    # adds spillovers & runs scenarios

# ── Paths --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs"      # baseline + scenarios
TEMP_DIR = ROOT_DIR / "temp"           # scratch
TEMP_DIR.mkdir(exist_ok=True)

# ── Utils --------------------------------------------------------------------

def _slugify(text: str, max_len: int = 40) -> str:
    import re, unicodedata
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:max_len] or "scenario"


def _standardise(df: pd.DataFrame, candidates: List[str], new: str) -> pd.DataFrame:
    """Rename first matching column to *new*."""
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: new})
    raise KeyError(f"None of {candidates} found in DataFrame")


def _capture_rules(prompt: str) -> List[Dict[str, Any]]:
    """Call LLM agent and return the rule list."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        agent.generate(prompt)
    raw = buf.getvalue().strip()
    data = json.loads(raw)  # will raise JSONDecodeError if malformed
    if "rules" not in data:
        raise ValueError("Agent output missing 'rules' key")
    return data["rules"]

# ── Baseline (spill‑inclusive) ------------------------------------------------

def _ensure_baseline() -> None:
    """Compute baseline with spillovers once and cache in outputs/."""
    reg_file = OUTPUT_DIR / "gdp_region_2023-2030.csv"
    if reg_file.exists():
        return

    print("⇢ Computing baseline GDP (with spillovers)…")

    # 1. raw Cobb‑Douglas
    baseline_model.main(str(INPUT_DIR))

    # 2. empty scenario to attach spillovers
    empty_json = TEMP_DIR / "empty_rules.json"
    empty_json.write_text("[]", encoding="utf-8")
    tmp_out = TEMP_DIR / "baseline_full"
    tmp_out.mkdir(exist_ok=True)
    run_scenario.main(str(INPUT_DIR), str(empty_json), str(tmp_out))

    # 3. move / rename
    for f in ["gdp_region.csv", "gdp_sector.csv", "gdp_saudi.csv"]:
        (tmp_out / f).replace(OUTPUT_DIR / f.replace(".csv", "_2023-2030.csv"))
    print("✔  Baseline stored in", OUTPUT_DIR)

# ── Comparison helpers --------------------------------------------------------

def _compare_region(base: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(scen, on=["regioncode", "year"], how="inner")
    df["abs_diff"] = df["Region_GDP_SAR_scenario"] - df["Region_GDP_SAR_baseline"]
    df["pct_diff"] = df["abs_diff"] / df["Region_GDP_SAR_baseline"] * 100
    return df


def _most_impacted_sector(base: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(scen, on=["regioncode", "SectorCode", "year"], how="inner",
                    suffixes=("_baseline", "_scenario"))
    df["abs_diff"] = df["GDP_SAR_spilled_scenario"] - df["GDP_SAR_spilled_baseline"]
    df["pct_diff"] = df["abs_diff"] / df["GDP_SAR_spilled_baseline"].replace(0, pd.NA) * 100
    df = df.dropna(subset=["pct_diff"])
    if df.empty:
        return df
    idx = df.groupby(["regioncode", "year"])["pct_diff"].apply(lambda s: s.abs().idxmax())
    return df.loc[idx].reset_index(drop=True)

# ── Orchestration ------------------------------------------------------------

def run_pipeline(prompt: str) -> None:
    slug = _slugify(prompt)
    scen_dir = OUTPUT_DIR / slug
    results_dir = scen_dir / "results"
    comp_dir = scen_dir / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(exist_ok=True)

    # 1. LLM ➜ rules
    rules = _capture_rules(prompt)
    scen_json = scen_dir / "scenario.json"
    scen_json.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print("✔  Scenario JSON saved →", scen_json)

    # 2. baseline
    _ensure_baseline()

    # 3. scenario run
    run_scenario.main(str(INPUT_DIR), str(scen_json), str(results_dir))

    # 4. comparisons ---------------------------------------------------
    base_reg = _standardise(
        pd.read_csv(OUTPUT_DIR / "gdp_region_2023-2030.csv"),
        ["Region_GDP_SAR_spilled", "Region_GDP_SAR"],
        "Region_GDP_SAR_baseline",
    )
    scen_reg = _standardise(
        pd.read_csv(results_dir / "gdp_region.csv"),
        ["Region_GDP_SAR_spilled", "Region_GDP_SAR"],
        "Region_GDP_SAR_scenario",
    )
    _compare_region(base_reg, scen_reg).to_csv(comp_dir / "region_comparison.csv", index=False)

    base_nat = _standardise(
        pd.read_csv(OUTPUT_DIR / "gdp_saudi_total_2023-2030.csv"),
        ["Saudi_GDP_SAR_spilled", "Saudi_GDP_SAR"],
        "Saudi_GDP_SAR_baseline",
    )
    scen_nat = _standardise(
        pd.read_csv(results_dir / "gdp_saudi.csv"),
        ["Saudi_GDP_SAR_spilled", "Saudi_GDP_SAR"],
        "Saudi_GDP_SAR_scenario",
    )
    nat = base_nat.merge(scen_nat, on="year")
    nat["abs_diff"] = nat["Saudi_GDP_SAR_scenario"] - nat["Saudi_GDP_SAR_baseline"]
    nat["pct_diff"] = nat["abs_diff"] / nat["Saudi_GDP_SAR_baseline"] * 100
    nat.to_csv(comp_dir / "national_comparison.csv", index=False)

    base_sec = _standardise(
        pd.read_csv(OUTPUT_DIR / "gdp_sector_2023-2030.csv"),
        ["GDP_SAR_spilled", "GDP_SAR"],
        "GDP_SAR_spilled_baseline",
    )
    scen_sec = _standardise(
        pd.read_csv(results_dir / "gdp_sector.csv"),
        ["GDP_SAR_spilled", "GDP_SAR"],
        "GDP_SAR_spilled_scenario",
    )
    _most_impacted_sector(base_sec, scen_sec).to_csv(comp_dir / "top_sector_impact.csv", index=False)

    # 5. console summary ----------------------------------------------
    print("\n──────── RESULTS ────────")
    for _, r in nat.iterrows():
        print(f"{int(r['year'])}: baseline {r['Saudi_GDP_SAR_baseline']:.0f} → scenario {r['Saudi_GDP_SAR_scenario']:.0f}  (Δ {r['pct_diff']:+.2f} %)")
    print("────────────────────────")
    print("Artefacts →", scen_dir)

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="GDP Scenario Simulator")
    parser.add_argument("-p", "--prompt", help="Scenario description")
    args = parser.parse_args()

    prompt_text = args.prompt or input("Enter policy scenario description:\n> ")
    run_pipeline(prompt_text)
