"""
Run alternative scenarios on the Cobb-Douglas GDP model
======================================================

Usage
-----
$ python gdp_scenario_runner.py \
      --input_dir inputs \
      --scenario   scenarios/pay_raise_ict_makkah.json \
      --output_dir outputs/pay_raise_ict_makkah
"""
import os, json, argparse
import pandas as pd
import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────
ALPHA               = 0.30          # capital share
HOURS_PER_WORKER    = 8 * 250       # 2 000 h / worker / year
YEAR_START, YEAR_END = 2023, 2030

# ── 1. KPI loader (same as baseline) ───────────────────────────────────
def load_kpis(path: str) -> pd.DataFrame:
    df = (
        pd.read_csv(path)
          .query("@YEAR_START <= year <= @YEAR_END")
          .groupby(["regioncode", "SectorCode", "year", "kpi"], as_index=False)
          .agg(value=("value", "mean"))
    )
    wide = (
        df.pivot_table(index=["regioncode", "SectorCode", "year"],
                       columns="kpi",
                       values="value",
                       aggfunc="first")
          .reset_index()
          .rename_axis(None, axis=1)
    )
    return wide

# ── 2. Apply scenario impacts ──────────────────────────────────────────
def apply_impacts(wide_df: pd.DataFrame, scenario_json: str) -> pd.DataFrame:
    with open(scenario_json, "r", encoding="utf-8") as f:
        rules = json.load(f)

    df = wide_df.copy()

    for rule in rules:
        region  = rule["region"]
        sector  = rule["sector"]
        kpi_sel = rule["kpi"]
        years   = {int(y): pct for y, pct in rule["years"].items()}

        # Build boolean mask
        mask = (df["year"].isin(years.keys()) &
               ((df["regioncode"] == region) if region != "all" else True) &
               ((df["SectorCode"]  == sector) if sector != "all" else True))

        # Figure out which KPI columns to touch
        kpi_cols = df.columns.intersection(["Productivity", "Workforce", "Investments"])
        if kpi_sel != "all":
            kpi_cols = [kpi_sel]

        # Apply multiplicative impacts year-by-year
        for yr, pct in years.items():
            year_mask = mask & (df["year"] == yr)
            factor = 1 + pct / 100.0
            df.loc[year_mask, kpi_cols] *= factor

    return df

# ── 3. Cobb-Douglas GDP at sector level ────────────────────────────────
def compute_sector_gdp(wide_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    req = {"Productivity", "Investments", "Workforce"}
    missing = req - set(wide_df.columns)
    if missing:
        raise KeyError(f"Missing KPI columns: {missing}")

    df = wide_df.copy()
    df["GDP_SAR"] = (
        df["Productivity"]
        * (df["Workforce"] * HOURS_PER_WORKER) ** (1 - alpha)
        * (df["Investments"] * 1_000_000) ** alpha
    )
    return df[["regioncode", "SectorCode", "year", "GDP_SAR"]]

# ── 4. Sector spillovers (square matrix) ───────────────────────────────
def apply_sector_spill(sector_df: pd.DataFrame,
                       spill_path: str) -> pd.DataFrame:
    """
    Add intra-regional sector spillovers.

    • sector_df:  regioncode, year, SectorCode, GDP_SAR
    • spill_path: csv where *first* column = sector letter,
                  remaining cols = spillover % to target sector.
    """
    # 1️⃣  pivot sector GDP wide
    wide = (sector_df
            .pivot(index=["regioncode", "year"],
                   columns="SectorCode",
                   values="GDP_SAR")
            .fillna(0))

    # 2️⃣  read spill matrix
    raw = pd.read_csv(spill_path, header=None)
    codes = raw.iloc[1:, 0].str.strip()          # sector letters
    mtx   = (raw.iloc[1:, 1:]                    # numeric body
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .values / 100.0)

    # 3️⃣  keep only codes present in *both* datasets, preserve order of wide
    common = [c for c in wide.columns if c in set(codes)]
    if not common:                               # nothing to spill
        sector_df = sector_df.rename(columns={"GDP_SAR": "GDP_SAR_spilled"})
        return sector_df

    # reorder matrix rows/cols to match `common`
    idx = [codes.tolist().index(c) for c in common]
    mtx = mtx[:, idx][idx, :]      # first select columns, then rows

    wide_c = wide[common]

    # 4️⃣  apply spillovers:  Y + Y·M
    spilled_vals = wide_c.values + wide_c.values.dot(mtx)
    spilled_df   = pd.DataFrame(spilled_vals,
                                index=wide_c.index,
                                columns=common)

    # 5️⃣  back to long
    long = (spilled_df
            .reset_index()
            .melt(id_vars=["regioncode", "year"],
                  var_name="SectorCode",
                  value_name="GDP_SAR_spilled"))
    return long


# ── 5. Regional spillovers ─────────────────────────────────────────────
def apply_regional_spill(region_df: pd.DataFrame, spill_path: str) -> pd.DataFrame:
    spill = pd.read_csv(spill_path)
    years = region_df["year"].unique()

    # Cross-join spill matrix with years
    spill_y = (
        pd.DataFrame(np.repeat(spill.values, len(years), axis=0),
                     columns=spill.columns)
          .assign(year=np.tile(years, len(spill)))
    )

    merged = (
        spill_y.merge(region_df,
                      left_on=["region_from", "year"],
                      right_on=["regioncode", "year"],
                      how="left")
               .rename(columns={"Region_GDP_SAR": "origin_gdp"})
               .fillna({"origin_gdp": 0})
    )

    merged["contrib"] = merged["origin_gdp"] * merged["spillover_pct"] / 100.0

    dest = (merged.groupby(["region_to", "year"], as_index=False)["contrib"]
                  .sum()
                  .rename(columns={"region_to": "regioncode"}))

    final = (
        region_df.merge(dest, on=["regioncode", "year"], how="left")
                 .fillna({"contrib": 0})
    )
    final["Region_GDP_SAR_spilled"] = final["Region_GDP_SAR"] + final["contrib"]
    return final[["regioncode", "year", "Region_GDP_SAR_spilled"]]

# ── 6. Aggregations ────────────────────────────────────────────────────
def aggregate_region(sector_df: pd.DataFrame) -> pd.DataFrame:
    return (sector_df.groupby(["regioncode", "year"], as_index=False)
                     .agg(Region_GDP_SAR=("GDP_SAR_spilled", "sum")))

def aggregate_country(region_df: pd.DataFrame) -> pd.DataFrame:
    return (region_df.groupby("year", as_index=False)
                     .agg(Saudi_GDP_SAR=("Region_GDP_SAR_spilled", "sum")))

# ── 7. Orchestrator ────────────────────────────────────────────────────
def main(input_dir: str,
         scenario_path: str,
         output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ① load KPIs + scenario overrides
    kpi_path = os.path.join(input_dir, "input_data.csv")
    kpis     = load_kpis(kpi_path)
    kpis_adj = apply_impacts(kpis, scenario_path)

    # ② GDP at sector level + sector spillovers
    sector_base  = compute_sector_gdp(kpis_adj)
    sector_final = apply_sector_spill(
        sector_base,
        os.path.join(input_dir, "sector_spillovers.csv")
    )

    # ③ Aggregate → region totals + regional spillovers
    region_base  = aggregate_region(sector_final)
    region_final = apply_regional_spill(
        region_base,
        os.path.join(input_dir, "regional_spillovers.csv")
    )

    # ④ Saudi total
    national = aggregate_country(region_final)

    # ⑤ Save
    sector_final.to_csv(os.path.join(output_dir, "gdp_sector.csv"),  index=False)
    region_final.to_csv(os.path.join(output_dir, "gdp_region.csv"),  index=False)
    national.to_csv    (os.path.join(output_dir, "gdp_saudi.csv"),   index=False)

    print(f"✔  Scenario results written to →  {output_dir}")

# ── 8. CLI plumbing ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="inputs")
    parser.add_argument("--scenario",   required=True,
                        help="Path to scenario JSON file")
    parser.add_argument("--output_dir", default="outputs/scenario")
    args = parser.parse_args()

    main(args.input_dir, args.scenario, args.output_dir)
