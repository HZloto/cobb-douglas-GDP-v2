# gdp_cobb_douglas.py  ──────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────
ALPHA = 0.30                 # Capital share in Cobb-Douglas
HOURS_PER_WORKER = 8 * 250   # 8 h × 250 working days ≈ 2 000 h / yr
YEAR_START, YEAR_END = 2023, 2030

# ── 1. Load & reshape KPI data ────────────────────────────────────────
def load_kpis(path: str) -> pd.DataFrame:
    """
    Return a wide dataframe keyed by regioncode, sector, year with columns:
    Productivity, Investments, Workforce
    """
    df = (
        pd.read_csv(path)
          # keep analysis window
          .query("@YEAR_START <= year <= @YEAR_END")
          # average duplicates if any
          .groupby(["regioncode", "SectorCode", "year", "kpi"], as_index=False)
          .agg(value=("value", "mean"))
    )

    wide = (
        df.pivot_table(
            index=["regioncode", "SectorCode", "year"],
            columns="kpi",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return wide


# ── 2. Cobb-Douglas at sector level ────────────────────────────────────
def compute_sector_gdp(wide_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    required = {"Productivity", "Investments", "Workforce"}
    if not required.issubset(wide_df.columns):
        missing = required - set(wide_df.columns)
        raise KeyError(f"Missing KPI columns: {missing}")

    df = wide_df.copy()

    # units: Productivity [SAR/hour] ; Workforce [workers] ; Investments [mn SAR]
    df["GDP_SAR"] = (
        df["Productivity"]
        * (df["Workforce"] * HOURS_PER_WORKER) ** (1 - alpha)
        * (df["Investments"] * 1_000_000) ** alpha
    )

    return df[["regioncode", "SectorCode", "year", "GDP_SAR"]]


# ── 3. Aggregations ────────────────────────────────────────────────────
def aggregate_region(sector_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sector_df.groupby(["regioncode", "year"], as_index=False)["GDP_SAR"]
        .sum()
        .rename(columns={"GDP_SAR": "Region_GDP_SAR"})
    )


def aggregate_country(region_df: pd.DataFrame) -> pd.DataFrame:
    return (
        region_df.groupby("year", as_index=False)["Region_GDP_SAR"]
        .sum()
        .rename(columns={"Region_GDP_SAR": "Saudi_GDP_SAR"})
    )


# ── 4. Main driver ─────────────────────────────────────────────────────
def main(input_dir: str = "inputs") -> None:
    csv_path = os.path.join(input_dir, "input_data.csv")
    kpis = load_kpis(csv_path)
    sector_gdp = compute_sector_gdp(kpis)

    region_gdp = aggregate_region(sector_gdp)
    national_gdp = aggregate_country(region_gdp)
    
    # ── Save outputs ───────────────────────────────────────────────────
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    sector_gdp.to_csv(os.path.join(output_dir, "gdp_sector_2023-2030.csv"), index=False)
    region_gdp.to_csv(os.path.join(output_dir, "gdp_region_2023-2030.csv"), index=False)
    national_gdp.to_csv(os.path.join(output_dir, "gdp_saudi_total_2023-2030.csv"), index=False)

    print("✔  Saved sector, region, and national GDP files.")

if __name__ == "__main__":
    main()
