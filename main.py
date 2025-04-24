import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass, field

# --- Configuration ---
@dataclass(frozen=True)
class Settings:
    """Holds configuration settings for the analysis."""
    # Required KPIs for base GDP calculation
    kpis_base_gdp: tuple = ("population", "investments", "productivity")
    kpi_population: str = "population"
    kpi_investments: str = "investments"
    kpi_productivity: str = "productivity"

    # Required columns for input files
    req_cols: dict = field(default_factory=lambda: {
        "regional": ["region", "year", "kpi", "value"],
        "sectors": ["region", "year", "sector", "pct_gdp"],
        "regional_spill": ["region_from", "region_to", "spillover_pct"],
        "sector_spill": []  # Validated by index/column check later
    })

    # Data types for efficient processing and validation
    dtypes: dict = field(default_factory=lambda: {
        "region": "category", "sector": "category", "kpi": "category",
        "region_from": "category", "region_to": "category",
        "year": "int16",
        "value": "float64", "pct_gdp": "float64", "spillover_pct": "float64"
    })

    # Default Cobb-Douglas exponents
    worker_exponent: float = 0.7
    capital_exponent: float = 0.3

    # Input/Output directory paths
    input_dir: str = "inputs"
    output_dir: str = "outputs"

    # File names (relative to input_dir)
    input_csv: str = "input_data.csv"
    sectors_csv: str = "sectors_data.csv"
    regional_spillover_csv: str = "regional_spillovers.csv"
    sector_spillover_csv: str = "sector_spillovers.csv"
    final_consolidated_csv: str = "consolidated_gdp_final.csv"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- Validation Function ---
def validate_dataframe(df: pd.DataFrame, name: str, cfg: Settings) -> pd.DataFrame:
    """Validates required columns and applies specified dtypes."""
    logging.debug(f"Validating DataFrame: {name}")
    required = cfg.req_cols.get(name, [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}")

    for col in df.columns:
        if col in cfg.dtypes:
            dtype = cfg.dtypes[col]
            try:
                # Updated categorical check
                if dtype == 'category' or isinstance(dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logging.error(f"Failed to cast column '{col}' in {name} to {dtype}: {e}")
                raise
    logging.debug(f"Validation successful for {name}")
    return df

# --- Data Loading (Using Validation & Config) ---
def load_data(cfg: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data from CSV files defined in config."""
    logging.info("Starting data loading...")
    start_time = time.time()
    try:
        input_path = lambda f: os.path.join(cfg.input_dir, f)
        
        # --- Load Regional Data ---
        regional_df = pd.read_csv(input_path(cfg.input_csv))
        regional_df = validate_dataframe(regional_df, "regional", cfg)
        regional_df['value'] = pd.to_numeric(regional_df['value'], errors='coerce')
        regional_df.dropna(subset=['value'], inplace=True)
        kpis_in_data = set(regional_df['kpi'].unique())
        missing_kpis = set(cfg.kpis_base_gdp) - kpis_in_data
        if missing_kpis:
            raise ValueError(f"Regional data missing required KPIs: {missing_kpis}")

        # --- Load Sectors Data ---
        sectors_df = pd.read_csv(input_path(cfg.sectors_csv))
        sectors_df = validate_dataframe(sectors_df, "sectors", cfg)
        sectors_df['pct_gdp'] = pd.to_numeric(sectors_df['pct_gdp'], errors='coerce') / 100.0
        sectors_df.dropna(subset=['pct_gdp'], inplace=True)

        # --- Load Regional Spillovers ---
        regional_spill_df = pd.read_csv(input_path(cfg.regional_spillover_csv))
        regional_spill_df = validate_dataframe(regional_spill_df, "regional_spill", cfg)
        regional_spill_df['spillover_pct'] = pd.to_numeric(
            regional_spill_df['spillover_pct'], errors='coerce'
        ) / 100.0
        regional_spill_df.dropna(subset=['spillover_pct'], inplace=True)

        # --- Load Sector Spillovers ---
        sector_spill_df = pd.read_csv(input_path(cfg.sector_spillover_csv), index_col=0)
        if not sector_spill_df.index.equals(sector_spill_df.columns):
            logging.warning("Sector spillover matrix index and columns differ. Ensure they represent sectors consistently.")
        sector_spill_df.index.name = 'sector_from'
        sector_spill_df.columns.name = 'sector_to'
        for col in sector_spill_df.columns:
            sector_spill_df[col] = pd.to_numeric(sector_spill_df[col], errors='coerce')
        sector_spill_df = sector_spill_df / 100.0
        sector_spill_df = sector_spill_df.fillna(0)

        logging.info(f"Data loading and preprocessing completed in {time.time() - start_time:.2f} seconds.")
        return regional_df, sectors_df, regional_spill_df, sector_spill_df

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Check if files exist in the {cfg.input_dir} directory.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data loading or validation: {e}")
        raise

# --- Directory Handling ---
def ensure_directories_exist(cfg: Settings) -> None:
    """Ensure both input and output directories exist."""
    for directory in [cfg.input_dir, cfg.output_dir]:
        if not os.path.exists(directory):
            logging.info(f"Creating directory: {directory}")
            os.makedirs(directory)

# --- Core Calculations ---
def calculate_base_regional_gdp_tidy(
    regional_df: pd.DataFrame,
    cfg: Settings
) -> pd.DataFrame:
    logging.info("Step 1: Calculating Base Regional GDP (Tidy)...")
    start_time = time.time()
    L = regional_df[regional_df['kpi'] == cfg.kpi_population]
    K = regional_df[regional_df['kpi'] == cfg.kpi_investments]
    A = regional_df[regional_df['kpi'] == cfg.kpi_productivity]

    gdp_long = pd.merge(
        A[['region','year','value']],
        L[['region','year','value']],
        on=["region","year"], suffixes=("_A","_L")
    ).merge(
        K[['region','year','value']],
        on=["region","year"]
    ).rename(columns={'value':'value_K'})

    gdp_long['gdp'] = (
        gdp_long['value_A'] *
        (gdp_long['value_L'] ** cfg.worker_exponent) *
        (gdp_long['value_K'] ** cfg.capital_exponent)
    )

    gdp_long['gdp'] = gdp_long['gdp'].replace([np.inf, -np.inf], np.nan).fillna(0)
    result_df = gdp_long[['region','year','gdp']].copy()
    result_df['step'] = 'base_gdp'
    logging.info(f"Step 1: Base Regional GDP calculated in {time.time() - start_time:.2f} seconds.")
    return result_df


def apply_regional_spillovers_tidy(
    base_gdp_df_long: pd.DataFrame,
    regional_spill_df: pd.DataFrame
) -> pd.DataFrame:
    logging.info("Step 2: Applying Regional Spillovers (Tidy)...")
    start_time = time.time()
    gdp_long = base_gdp_df_long.rename(columns={'gdp':'base_gdp'}).copy()

    regions_idx = gdp_long['region'].unique()
    spill_filtered = regional_spill_df[    
        regional_spill_df['region_from'].isin(regions_idx) &
        regional_spill_df['region_to'].isin(regions_idx)
    ]
    if spill_filtered.empty:
        logging.warning("No applicable regional spillovers found. Returning base GDP.")
        gdp_long['step'] = 'after_regional_spillovers'
        return gdp_long.rename(columns={'base_gdp':'gdp'})

    merged = pd.merge(
        spill_filtered,
        gdp_long[['region','year','base_gdp']],
        left_on='region_from', right_on='region', how='inner'
    ).drop(columns='region')

    merged['spillover_amount'] = merged['base_gdp'] * merged['spillover_pct']
    total_received = merged.groupby(
        ['region_to','year'], observed=False
    )['spillover_amount'].sum().reset_index()
    total_received.rename(
        columns={'region_to':'region','spillover_amount':'total_spillover_received'},
        inplace=True
    )

    adj = pd.merge(
        gdp_long, total_received, on=['region','year'], how='left'
    )
    adj['total_spillover_received'] = adj['total_spillover_received'].fillna(0)
    adj['gdp'] = adj['base_gdp'] + adj['total_spillover_received']
    adj['step'] = 'after_regional_spillovers'

    result_df = adj[['region','year','gdp','step']].copy()
    logging.info(f"Step 2: Regional Spillovers applied in {time.time() - start_time:.2f} seconds.")
    return result_df


def breakdown_gdp_by_sector_tidy(
    adj_gdp_df_long: pd.DataFrame,
    sectors_df: pd.DataFrame
) -> pd.DataFrame:
    logging.info("Step 3: Breaking Down GDP by Sector (Tidy)...")
    start_time = time.time()
    adj = adj_gdp_df_long.rename(columns={'gdp':'adj_regional_gdp'})
    merged = pd.merge(
        adj[['region','year','adj_regional_gdp']],
        sectors_df[['region','year','sector','pct_gdp']],
        on=['region','year'], how='inner'
    )
    pct_sum = merged.groupby(
        ['region','year'], observed=False
    )['pct_gdp'].sum()
    if not np.allclose(pct_sum, 1.0, atol=0.01):
        logging.warning("Sector percentages may not sum to 1.0 for some region/year")

    merged['gdp'] = merged['adj_regional_gdp'] * merged['pct_gdp']
    result_df = merged[['region','year','sector','gdp']].copy()
    result_df['step'] = 'before_sector_spillovers'
    logging.info(f"Step 3: Sector breakdown completed in {time.time() - start_time:.2f} seconds.")
    return result_df


def apply_sector_spillovers_einsum(
    initial_sector_gdp_df: pd.DataFrame,
    sector_spill_df: pd.DataFrame
) -> pd.DataFrame:
    logging.info("Step 4: Applying Sector Spillovers (Einsum)...")
    start_time = time.time()

    # Prepare indices
    regions_idx = pd.Index(initial_sector_gdp_df['region'].astype('category').cat.categories, name='region')
    years_idx = pd.Index(sorted(initial_sector_gdp_df['year'].unique()), name='year')
    sectors_idx = pd.Index(initial_sector_gdp_df['sector'].astype('category').cat.categories, name='sector')

    # Pivot to cube
    gdp_pivot = initial_sector_gdp_df.pivot_table(
        index='region', columns=['year','sector'], values='gdp',
        fill_value=0.0, observed=False
    )
    multi_idx = pd.MultiIndex.from_product(
        [years_idx, sectors_idx], names=['year','sector']
    )
    gdp_cube_df = gdp_pivot.reindex(index=regions_idx, columns=multi_idx, fill_value=0.0)
    gdp_cube_np = gdp_cube_df.values.reshape(
        (len(regions_idx), len(years_idx), len(sectors_idx))
    )

    S = sector_spill_df.reindex(index=sectors_idx, columns=sectors_idx, fill_value=0.0).to_numpy()
    inflow = np.einsum("rys,st->ryt", gdp_cube_np, S, optimize='greedy')
    final_cube = gdp_cube_np + inflow

    final_index = pd.MultiIndex.from_product(
        [regions_idx, years_idx, sectors_idx], names=['region','year','sector']
    )
    final_df = pd.DataFrame(
        final_cube.ravel(), index=final_index, columns=['gdp']
    ).reset_index()
    final_df['step'] = 'after_sector_spillovers'
    final_df['region'] = final_df['region'].astype('category')
    final_df['sector'] = final_df['sector'].astype('category')
    final_df['year'] = final_df['year'].astype('int16')

    logging.info(f"Step 4: Sector spillovers (Einsum) applied in {time.time() - start_time:.2f} seconds.")
    return final_df


def create_consolidated_output_tidy(sector_gdp_df_final: pd.DataFrame) -> pd.DataFrame:
    logging.info("Creating consolidated output from tidy data...")
    start_time = time.time()
    df = sector_gdp_df_final[sector_gdp_df_final['step']=='after_sector_spillovers'].copy()
    df['row_type'] = 'detail'
    parts = [df]

    # sector totals
    sector_totals = df.groupby(['sector','year'], observed=False, as_index=False)['gdp'].sum()
    sector_totals['region'] = 'ALL REGIONS'
    sector_totals['row_type'] = 'sector_total'
    parts.append(sector_totals)

    # region totals
    region_tots = df.groupby(['region','year'], observed=False, as_index=False)['gdp'].sum()
    region_tots['sector'] = 'ALL SECTORS'
    region_tots['row_type'] = 'region_total'
    parts.append(region_tots)

    # grand totals
    grand = region_tots.groupby('year', as_index=False)['gdp'].sum()
    grand['region'] = 'ALL REGIONS'
    grand['sector'] = 'ALL SECTORS'
    grand['row_type'] = 'grand_total'
    parts.append(grand)

    consolidated = pd.concat(parts, ignore_index=True)
    consolidated['step'] = 'final_consolidated'
    cols = ['year','region','sector','gdp','step','row_type']
    consolidated = consolidated[cols].sort_values(by=cols[:3]).reset_index(drop=True)
    logging.info(f"Consolidated output created in {time.time() - start_time:.2f} seconds.")
    return consolidated


def track_gdp_calculation_stages_tidy(
    base_gdp: Optional[pd.DataFrame],
    adj_gdp: Optional[pd.DataFrame],
    initial_sector_gdp: Optional[pd.DataFrame],
    final_sector_gdp: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    logging.info("Constructing tracking DataFrame from tidy stages...")
    parts = []
    if base_gdp is not None:
        df = base_gdp.copy()
        df['sector'] = 'ALL SECTORS'
        df['row_type'] = 'region_total'
        parts.append(df)
    if adj_gdp is not None:
        df = adj_gdp.copy()
        df['sector'] = 'ALL SECTORS'
        df['row_type'] = 'region_total'
        parts.append(df)
    if initial_sector_gdp is not None:
        df = initial_sector_gdp.copy()
        df['row_type'] = 'detail'
        parts.append(df)
    if final_sector_gdp is not None:
        df = final_sector_gdp.copy()
        df['row_type'] = 'detail'
        parts.append(df)
    if not parts:
        logging.warning("No data for tracking stages.")
        return None
    track = pd.concat(parts, ignore_index=True)
    cols = ['year','region','sector','gdp','step','row_type']
    track = track[cols].sort_values(by=cols).reset_index(drop=True)
    logging.info("Tracking DataFrame constructed.")
    return track


def run_regional_analysis(
    cfg: Settings,
    save_intermediate: bool = False,
    generate_tracking: bool = False
) -> Optional[pd.DataFrame]:
    logging.info("=== Starting Regional GDP Analysis (Tidy Workflow) ===")
    start_all = time.time()
    final_df = None
    try:
        ensure_directories_exist(cfg)
        regional_df, sectors_df, regional_spill_df, sector_spill_df = load_data(cfg)

        base_gdp_df = calculate_base_regional_gdp_tidy(regional_df, cfg)
        adj_gdp_df = apply_regional_spillovers_tidy(base_gdp_df, regional_spill_df)
        init_sec_df = breakdown_gdp_by_sector_tidy(adj_gdp_df, sectors_df)
        final_sec_df = apply_sector_spillovers_einsum(init_sec_df, sector_spill_df)
        final_df = create_consolidated_output_tidy(final_sec_df)

        if save_intermediate:
            base_gdp_df.to_csv(os.path.join(cfg.output_dir,"base_gdp_tidy.csv"), index=False)
            adj_gdp_df.to_csv(os.path.join(cfg.output_dir,"adjusted_gdp_tidy.csv"), index=False)
            final_sec_df.to_csv(os.path.join(cfg.output_dir,"final_sector_gdp_tidy.csv"), index=False)
        if generate_tracking:
            track = track_gdp_calculation_stages_tidy(
                base_gdp_df, adj_gdp_df, init_sec_df, final_sec_df
            )
            if track is not None:
                track.to_csv(os.path.join(cfg.output_dir,"gdp_calculation_stages_tidy.csv"), index=False)

        if final_df is not None and not final_df.empty:
            final_df.to_csv(os.path.join(cfg.output_dir, cfg.final_consolidated_csv), index=False)
            logging.info(f"Final consolidated results saved to {cfg.output_dir}/{cfg.final_consolidated_csv}")
        logging.info(f"=== Regional GDP Analysis Completed in {time.time() - start_all:.2f} seconds ===")
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return None
    return final_df


if __name__ == "__main__":
    cfg = Settings()

    def create_dummy_data(cfg: Settings):
        ensure_directories_exist(cfg.input_dir)
        input_path = lambda f: os.path.join(cfg.input_dir, f)
        
        if not os.path.exists(input_path(cfg.input_csv)):
            pd.DataFrame([
                {'region':'Riyadh Region','year':2020,'kpi':'population','value':8500000},
                {'region':'Riyadh Region','year':2020,'kpi':'investments','value':120000000000},
                {'region':'Riyadh Region','year':2020,'kpi':'productivity','value':50000},
                {'region':'Riyadh Region','year':2021,'kpi':'population','value':8700000},
                {'region':'Riyadh Region','year':2021,'kpi':'investments','value':130000000000},
                {'region':'Riyadh Region','year':2021,'kpi':'productivity','value':52000},
                {'region':'Makkah Region','year':2020,'kpi':'population','value':9000000},
                {'region':'Makkah Region','year':2020,'kpi':'investments','value':80000000000},
                {'region':'Makkah Region','year':2020,'kpi':'productivity','value':45000},
                {'region':'Makkah Region','year':2021,'kpi':'population','value':9200000},
                {'region':'Makkah Region','year':2021,'kpi':'investments','value':85000000000},
                {'region':'Makkah Region','year':2021,'kpi':'productivity','value':46000}
            ]).to_csv(input_path(cfg.input_csv), index=False)
        if not os.path.exists(input_path(cfg.sectors_csv)):
            pd.DataFrame([
                {'region':'Riyadh Region','year':2020,'sector':'A','pct_gdp':50.0},
                {'region':'Riyadh Region','year':2020,'sector':'B','pct_gdp':30.0},
                {'region':'Riyadh Region','year':2020,'sector':'C','pct_gdp':20.0},
                {'region':'Riyadh Region','year':2021,'sector':'A','pct_gdp':52.0},
                {'region':'Riyadh Region','year':2021,'sector':'B','pct_gdp':28.0},
                {'region':'Riyadh Region','year':2021,'sector':'C','pct_gdp':20.0},
                {'region':'Makkah Region','year':2020,'sector':'A','pct_gdp':45.0},
                {'region':'Makkah Region','year':2020,'sector':'B','pct_gdp':35.0},
                {'region':'Makkah Region','year':2020,'sector':'C','pct_gdp':20.0},
                {'region':'Makkah Region','year':2021,'sector':'A','pct_gdp':47.0},
                {'region':'Makkah Region','year':2021,'sector':'B','pct_gdp':33.0},
                {'region':'Makkah Region','year':2021,'sector':'C','pct_gdp':20.0}
            ]).to_csv(input_path(cfg.sectors_csv), index=False)
        if not os.path.exists(input_path(cfg.regional_spillover_csv)):
            pd.DataFrame({
                'region_from':['Riyadh Region','Makkah Region'],
                'region_to':['Makkah Region','Riyadh Region'],
                'spillover_pct':[25.0,27.0]
            }).to_csv(input_path(cfg.regional_spillover_csv), index=False)
        if not os.path.exists(input_path(cfg.sector_spillover_csv)):
            pd.DataFrame({
                'A':[np.nan,10.0,5.0],
                'B':[8.0,np.nan,7.0],
                'C':[6.0,9.0,np.nan]
            }, index=pd.Index(['A','B','C'], name='')).to_csv(input_path(cfg.sector_spillover_csv), index_label='')

    create_dummy_data(cfg)
    final = run_regional_analysis(cfg, save_intermediate=False, generate_tracking=False)
    if final is not None:
        print(final.head(20))
