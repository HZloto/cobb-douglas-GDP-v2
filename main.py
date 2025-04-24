"""
Regional GDP Analysis Tool using Cobb-Douglas Production Function

This module provides functionality for calculating regional GDP
using the Cobb-Douglas production function, based on regional data for
population (workforce), investments, productivity, and sector breakdown.
It also incorporates regional and sectoral spillover effects.

(Refactored for Tidy Data, Einsum Optimization, and Configuration)
"""
import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
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
        "sector_spill": [] # Validated by index/column check later
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

    # File names (assuming they are in the same directory as the script)
    input_csv: str = "input_data.csv"
    sectors_csv: str = "sectors_data.csv"
    regional_spillover_csv: str = "regional_spillovers.csv"
    sector_spillover_csv: str = "sector_spillovers.csv"
    output_dir: str = "outputs"
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

    # Apply specified dtypes
    dtype_map = {
        c: cfg.dtypes.get(c) for c in df.columns if cfg.dtypes.get(c)
    }
    # Handle potential errors during casting
    for col, dtype in dtype_map.items():
         try:
             if pd.api.types.is_categorical_dtype(dtype):
                  df[col] = df[col].astype('category')
             # Handle potential conversion issues for numeric types
             elif pd.api.types.is_numeric_dtype(dtype):
                  df[col] = pd.to_numeric(df[col], errors='coerce')
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
        # --- Load Regional Data ---
        regional_df = pd.read_csv(cfg.input_csv)
        regional_df = validate_dataframe(regional_df, "regional", cfg)
        # Convert numeric value after validation
        regional_df['value'] = pd.to_numeric(regional_df['value'], errors='coerce')
        regional_df.dropna(subset=['value'], inplace=True)
        # Ensure required base KPIs are present
        kpis_in_data = set(regional_df['kpi'].unique())
        missing_kpis = set(cfg.kpis_base_gdp) - kpis_in_data
        if missing_kpis:
            raise ValueError(f"Regional data missing required KPIs: {missing_kpis}")

        # --- Load Sectors Data ---
        sectors_df = pd.read_csv(cfg.sectors_csv)
        sectors_df = validate_dataframe(sectors_df, "sectors", cfg)
        # Convert pct_gdp to decimal
        sectors_df['pct_gdp'] = pd.to_numeric(sectors_df['pct_gdp'], errors='coerce') / 100.0
        sectors_df.dropna(subset=['pct_gdp'], inplace=True)


        # --- Load Regional Spillovers ---
        regional_spillover_df = pd.read_csv(cfg.regional_spillover_csv)
        regional_spillover_df = validate_dataframe(regional_spillover_df, "regional_spill", cfg)
        # Convert spillover_pct to decimal
        regional_spillover_df['spillover_pct'] = pd.to_numeric(regional_spillover_df['spillover_pct'], errors='coerce') / 100.0
        regional_spillover_df.dropna(subset=['spillover_pct'], inplace=True)

        # --- Load Sector Spillovers ---
        sector_spillover_df = pd.read_csv(cfg.sector_spillover_csv, index_col=0)
        # Validate index/column names match (basic check)
        if not sector_spillover_df.index.equals(sector_spillover_df.columns):
             logging.warning("Sector spillover matrix index and columns differ. Ensure they represent sectors consistently.")
        sector_spillover_df.index.name = 'sector_from'
        sector_spillover_df.columns.name = 'sector_to'
        # Convert values (percentages) to numeric, then to decimal
        for col in sector_spillover_df.columns:
            sector_spillover_df[col] = pd.to_numeric(sector_spillover_df[col], errors='coerce')
        sector_spillover_df = sector_spillover_df / 100.0
        sector_spillover_df.fillna(0, inplace=True)


        logging.info(f"Data loading and preprocessing completed in {time.time() - start_time:.2f} seconds.")
        return regional_df, sectors_df, regional_spillover_df, sector_spillover_df

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Check if file exists in the script directory.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data loading or validation: {e}")
        raise


# --- Directory Handling ---
def ensure_output_directory_exists(output_dir: str) -> None:
    """Ensure that the output directory exists."""
    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)


# --- Core Calculations (Refactored for Tidy Data) ---

# REMOVED: transform_to_wide_format - No longer needed

def calculate_base_regional_gdp_tidy(
    regional_df: pd.DataFrame,
    cfg: Settings
) -> pd.DataFrame:
    """
    Calculate base regional GDP using Cobb-Douglas from tidy data.
    Returns a tidy DataFrame: [region, year, gdp].
    """
    logging.info("Step 1: Calculating Base Regional GDP (Tidy)...")
    start_time = time.time()
    try:
        # Filter KPIs using boolean masks (efficient on categorical data)
        L = regional_df[regional_df['kpi'] == cfg.kpi_population]
        K = regional_df[regional_df['kpi'] == cfg.kpi_investments]
        A = regional_df[regional_df['kpi'] == cfg.kpi_productivity]

        # Merge the long-format dataframes
        gdp_long = pd.merge(
            A[['region', 'year', 'value']],
            L[['region', 'year', 'value']],
            on=["region", "year"], suffixes=("_A", "_L")
        ).merge(
            K[['region', 'year', 'value']],
            on=["region", "year"]
        ).rename(columns={'value': 'value_K'}) # Rename last value col explicitly

        if gdp_long.empty:
             raise ValueError("Merging A, L, K resulted in empty DataFrame. Check input data for complete region-year combinations.")

        # Vectorized Cobb-Douglas calculation
        gdp_long["gdp"] = (
            gdp_long["value_A"] *
            (gdp_long["value_L"] ** cfg.worker_exponent) *
            (gdp_long["value_K"] ** cfg.capital_exponent)
        )

        # Handle potential NaNs/Infs from calculation (e.g., 0^negative_exponent)
        gdp_long['gdp'] = gdp_long['gdp'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Select and return the relevant columns in tidy format
        result_df = gdp_long[['region', 'year', 'gdp']].copy()
        result_df['step'] = 'base_gdp' # Add step tracking

        logging.info(f"Step 1: Base Regional GDP calculated in {time.time() - start_time:.2f} seconds.")
        return result_df

    except Exception as e:
        logging.error(f"Error calculating base GDP: {e}")
        raise


def apply_regional_spillovers_tidy(
    base_gdp_df_long: pd.DataFrame, # Expects [region, year, gdp]
    regional_spillover_df: pd.DataFrame
) -> pd.DataFrame:
    """Apply regional spillover effects to tidy GDP data."""
    logging.info("Step 2: Applying Regional Spillovers (Tidy)...")
    start_time = time.time()
    try:
        # Input is already long ('base_gdp_df_long')
        gdp_long = base_gdp_df_long.copy()
        gdp_long.rename(columns={'gdp': 'base_gdp'}, inplace=True) # Rename for clarity

        if gdp_long.empty:
            logging.warning("Base GDP DataFrame is empty. Skipping regional spillovers.")
            gdp_long['step'] = 'after_regional_spillovers'
            return gdp_long.rename(columns={'base_gdp':'gdp'}) # Return with correct col name


        # --- Same logic as before, but starting and ending with long format ---
        regions_in_gdp = gdp_long['region'].unique()
        regional_spillover_filtered = regional_spillover_df[
            regional_spillover_df['region_from'].isin(regions_in_gdp) &
            regional_spillover_df['region_to'].isin(regions_in_gdp)
        ].copy()

        if regional_spillover_filtered.empty:
            logging.warning("No applicable regional spillovers found. Returning base GDP.")
            gdp_long['step'] = 'after_regional_spillovers'
            return gdp_long.rename(columns={'base_gdp':'gdp'})

        merged_spillovers = pd.merge(
            regional_spillover_filtered,
            gdp_long[['region', 'year', 'base_gdp']],
            left_on='region_from',
            right_on='region',
            how='inner'
        ).drop(columns='region')

        merged_spillovers['spillover_amount'] = merged_spillovers['base_gdp'] * merged_spillovers['spillover_pct']

        total_spillovers_received = merged_spillovers.groupby(['region_to', 'year'])['spillover_amount'].sum().reset_index()
        total_spillovers_received.rename(columns={'region_to': 'region', 'spillover_amount':'total_spillover_received'}, inplace=True)

        adj_gdp_long = pd.merge(
            gdp_long,
            total_spillovers_received,
            on=['region', 'year'],
            how='left'
        )

        adj_gdp_long['total_spillover_received'].fillna(0, inplace=True)
        adj_gdp_long['gdp'] = adj_gdp_long['base_gdp'] + adj_gdp_long['total_spillover_received']
        adj_gdp_long['step'] = 'after_regional_spillovers'

        # Return the adjusted GDP in tidy format
        result_df = adj_gdp_long[['region', 'year', 'gdp', 'step']].copy()
        logging.info(f"Step 2: Regional Spillovers applied in {time.time() - start_time:.2f} seconds.")
        return result_df

    except Exception as e:
        logging.error(f"Error applying regional spillovers: {e}")
        raise


def breakdown_gdp_by_sector_tidy(
    adj_gdp_df_long: pd.DataFrame, # Expects [region, year, gdp]
    sectors_df: pd.DataFrame
) -> pd.DataFrame:
    """Break down adjusted tidy GDP by sector."""
    logging.info("Step 3: Breaking Down GDP by Sector (Tidy)...")
    start_time = time.time()
    try:
        # Input adj_gdp_df_long is already tidy
        if adj_gdp_df_long.empty:
            logging.warning("Adjusted GDP DataFrame is empty. Cannot perform sector breakdown.")
            return pd.DataFrame(columns=['region', 'year', 'sector', 'gdp', 'step'])

        adj_gdp_long = adj_gdp_df_long.rename(columns={'gdp':'adj_regional_gdp'})

        # Merge adjusted GDP with sector percentages
        sector_gdp_df = pd.merge(
            adj_gdp_long[['region', 'year', 'adj_regional_gdp']], # Select necessary columns
            sectors_df[['region', 'year', 'sector', 'pct_gdp']],
            on=['region', 'year'],
            how='inner'
        )

        if sector_gdp_df.empty:
            logging.warning("Merge between adjusted GDP and sectors data resulted in an empty DataFrame.")
            return pd.DataFrame(columns=['region', 'year', 'sector', 'gdp', 'step'])

        # Check percentage sums
        pct_sum = sector_gdp_df.groupby(['region', 'year'])['pct_gdp'].sum()
        if not np.allclose(pct_sum, 1.0, atol=0.01):
            logging.warning(f"Sector percentages may not sum to 1.0 (100%) for some region/year combinations.")

        # Calculate GDP for each sector
        sector_gdp_df['gdp'] = sector_gdp_df['adj_regional_gdp'] * sector_gdp_df['pct_gdp']

        # Keep necessary columns and mark step
        result_df = sector_gdp_df[['region', 'year', 'sector', 'gdp']].copy()
        result_df['step'] = 'before_sector_spillovers'

        logging.info(f"Step 3: Sector breakdown completed in {time.time() - start_time:.2f} seconds.")
        return result_df
    except Exception as e:
        logging.error(f"Error breaking down GDP by sector: {e}")
        raise


# --- OPTIMIZED Sector Spillovers using einsum ---
def apply_sector_spillovers_einsum(
    initial_sector_gdp_df: pd.DataFrame, # Expects [region, year, sector, gdp]
    sector_spillover_df: pd.DataFrame    # Expects index=from, columns=to
) -> pd.DataFrame:
    """Apply sector spillover effects using numpy einsum."""
    logging.info("Step 4: Applying Sector Spillovers (Einsum)...")
    start_time = time.time()
    try:
        if initial_sector_gdp_df.empty:
            logging.warning("Initial sector GDP DataFrame is empty. Skipping sector spillovers.")
            final_df = initial_sector_gdp_df.copy()
            final_df['step'] = 'after_sector_spillovers'
            return final_df

        # Prepare unique sorted indices for cube dimensions
        regions_idx = pd.Index(initial_sector_gdp_df['region'].cat.categories, name='region')
        years_idx = pd.Index(sorted(initial_sector_gdp_df['year'].unique()), name='year')
        sectors_idx = pd.Index(initial_sector_gdp_df['sector'].cat.categories, name='sector')


        # Pivot to create the 3D cube: (regions, years, sectors)
        # Use MultiIndex for columns before unstacking or use pivot_table
        gdp_pivot = initial_sector_gdp_df.pivot_table(
            index='region', columns=['year', 'sector'], values='gdp',
            fill_value=0.0 # Ensure missing combinations are zero
        )
        # Reindex to ensure all combinations are present in the correct order
        multi_idx = pd.MultiIndex.from_product([years_idx, sectors_idx], names=['year', 'sector'])
        gdp_cube_df = gdp_pivot.reindex(index=regions_idx, columns=multi_idx, fill_value=0.0)

        # Reshape DataFrame into a 3D NumPy array (regions, years, sectors)
        # The shape needs to match the order of indices
        gdp_cube_np = gdp_cube_df.values.reshape((len(regions_idx), len(years_idx), len(sectors_idx)))


        # Prepare the (sectors, sectors) spillover matrix S (from -> rows, to -> cols)
        # Ensure it's aligned with sectors_idx
        S_matrix = sector_spillover_df.reindex(index=sectors_idx, columns=sectors_idx, fill_value=0.0).to_numpy()

        # Apply spillovers using einsum: "rys,st->ryt"
        # Sum over 's' (source sector): Multiply GDP of each source sector 's' in region 'r' year 'y'
        # by the spillover rate from 's' to target sector 't'. Result is total inflow to sector 't'.
        # gdp_cube_np shape: (n_regions, n_years, n_sectors)
        # S_matrix shape:    (n_sectors, n_sectors)
        # inflow shape:      (n_regions, n_years, n_sectors)
        inflow = np.einsum("rys,st->ryt", gdp_cube_np, S_matrix, optimize='greedy') # Use optimize flag

        # Add the calculated inflow back to the original cube values
        final_gdp_cube_np = gdp_cube_np + inflow

        # --- Reshape the final 3D NumPy array back into a tidy DataFrame ---
        # Create DataFrame shell using MultiIndex product
        final_tidy_index = pd.MultiIndex.from_product(
            [regions_idx, years_idx, sectors_idx],
            names=['region', 'year', 'sector']
        )
        final_df = pd.DataFrame(
            final_gdp_cube_np.ravel(), # Flatten array in C-order (last axis varies fastest)
            index=final_tidy_index,
            columns=['gdp']
        ).reset_index()


        # Assign step and ensure correct dtypes
        final_df['step'] = 'after_sector_spillovers'
        # Cast back to original dtypes where appropriate (especially categoricals)
        final_df['region'] = final_df['region'].astype('category')
        final_df['sector'] = final_df['sector'].astype('category')
        final_df['year'] = final_df['year'].astype('int16') # Or original int type

        logging.info(f"Step 4: Sector spillovers (Einsum) applied in {time.time() - start_time:.2f} seconds.")
        return final_df

    except Exception as e:
        logging.error(f"Error applying sector spillovers using einsum: {e}")
        raise


# --- Consolidated Output (Adapted for Tidy Input) ---
def create_consolidated_output_tidy(sector_gdp_df_final: pd.DataFrame) -> pd.DataFrame:
    """Create a consolidated output from final tidy sector GDP."""
    logging.info("Creating consolidated output from tidy data...")
    start_time = time.time()
    try:
        # Input 'sector_gdp_df_final' is already the final tidy data
        df = sector_gdp_df_final[sector_gdp_df_final['step'] == 'after_sector_spillovers'].copy()
        if df.empty:
             logging.warning("No data found for step 'after_sector_spillovers'. Consolidated output will be empty.")
             return pd.DataFrame(columns=['year', 'region', 'sector', 'gdp', 'step', 'row_type'])

        df['gdp'] = pd.to_numeric(df['gdp'], errors='coerce').fillna(0)
        df['row_type'] = 'detail'
        consolidated_parts = [df]

        # --- Aggregations remain the same ---
        if not df.empty:
            sector_totals = df.groupby(['sector', 'year'], observed=False, as_index=False)['gdp'].sum()
            sector_totals['region'] = 'ALL REGIONS'
            sector_totals['row_type'] = 'sector_total'
            consolidated_parts.append(sector_totals)

            region_totals = df.groupby(['region', 'year'], observed=False, as_index=False)['gdp'].sum()
            region_totals['sector'] = 'ALL SECTORS'
            region_totals['row_type'] = 'region_total'
            consolidated_parts.append(region_totals)

            if not region_totals.empty:
                year_totals = region_totals.groupby('year', as_index=False)['gdp'].sum()
                year_totals['region'] = 'ALL REGIONS'
                year_totals['sector'] = 'ALL SECTORS'
                year_totals['row_type'] = 'grand_total'
                consolidated_parts.append(year_totals)

            consolidated_df = pd.concat(consolidated_parts, ignore_index=True)
            consolidated_df['step'] = 'final_consolidated'
        else:
             consolidated_df = df # Return empty structure
             consolidated_df['step'] = 'final_consolidated' # Ensure step column exists


        # Reorder and sort
        cols_order = ['year', 'region', 'sector', 'gdp', 'step', 'row_type']
        # Ensure categorical types are handled if needed before final output
        consolidated_df['region'] = consolidated_df['region'].astype(str) # Convert category to str for output if needed
        consolidated_df['sector'] = consolidated_df['sector'].astype(str)


        consolidated_df = consolidated_df.reindex(columns=cols_order)
        consolidated_df = consolidated_df.sort_values(by=['year', 'region', 'sector']).reset_index(drop=True)

        logging.info(f"Consolidated output created in {time.time() - start_time:.2f} seconds.")
        return consolidated_df
    except Exception as e:
        logging.error(f"Error creating consolidated output: {e}")
        raise


# --- Tracking (Adapted for Tidy) ---
def track_gdp_calculation_stages_tidy(
    base_gdp: Optional[pd.DataFrame],        # region, year, gdp, step='base_gdp'
    adj_gdp: Optional[pd.DataFrame],         # region, year, gdp, step='after_regional_spillovers'
    initial_sector_gdp: Optional[pd.DataFrame], # region, year, sector, gdp, step='before_sector_spillovers'
    final_sector_gdp: Optional[pd.DataFrame]    # region, year, sector, gdp, step='after_sector_spillovers'
) -> Optional[pd.DataFrame]:
    """Track stages of GDP calculation from tidy DataFrames."""
    logging.info("Constructing tracking DataFrame from tidy stages (if requested)...")
    start_time = time.time()
    tracking_parts = []
    try:
        # Base GDP needs 'sector' and 'row_type' added
        if base_gdp is not None and not base_gdp.empty:
            base_track = base_gdp.copy()
            base_track['sector'] = 'ALL SECTORS'
            base_track['row_type'] = 'region_total'
            tracking_parts.append(base_track)

        # Adjusted GDP needs 'sector' and 'row_type' added
        if adj_gdp is not None and not adj_gdp.empty:
            adj_track = adj_gdp.copy()
            adj_track['sector'] = 'ALL SECTORS'
            adj_track['row_type'] = 'region_total'
            tracking_parts.append(adj_track)

        # Initial Sector GDP needs 'row_type' added
        if initial_sector_gdp is not None and not initial_sector_gdp.empty:
            init_sec_track = initial_sector_gdp.copy()
            init_sec_track['row_type'] = 'detail'
            tracking_parts.append(init_sec_track)

        # Final Sector GDP needs 'row_type' added
        if final_sector_gdp is not None and not final_sector_gdp.empty:
            final_sec_track = final_sector_gdp.copy()
            final_sec_track['row_type'] = 'detail'
            tracking_parts.append(final_sec_track)

        if not tracking_parts:
            logging.warning("No data available for any tracking stage.")
            return None

        tracking_df = pd.concat(tracking_parts, ignore_index=True)

        cols_order = ['year', 'region', 'sector', 'gdp', 'step', 'row_type']
        tracking_df = tracking_df.reindex(columns=cols_order)
        # Ensure correct types before sorting if categoricals were lost
        tracking_df['region'] = tracking_df['region'].astype('category')
        tracking_df['sector'] = tracking_df['sector'].astype('category')
        tracking_df['year'] = tracking_df['year'].astype('int16')
        tracking_df = tracking_df.sort_values(by=['year', 'region', 'sector', 'step']).reset_index(drop=True)
        # Convert categoricals back to string for final output if desired
        tracking_df['region'] = tracking_df['region'].astype(str)
        tracking_df['sector'] = tracking_df['sector'].astype(str)


        logging.info(f"Tracking DataFrame constructed in {time.time() - start_time:.2f} seconds.")
        return tracking_df

    except Exception as e:
        logging.error(f"Error constructing tracking DataFrame: {e}")
        return None


# --- Main Execution Function (Refactored) ---
def run_regional_analysis(
    cfg: Settings, # Pass Settings object
    save_intermediate: bool = False,
    generate_tracking: bool = False
) -> Optional[pd.DataFrame]:
    """Run the complete regional GDP analysis using tidy data approach."""
    logging.info("=== Starting Regional GDP Analysis (Tidy Workflow) ===")
    overall_start_time = time.time()
    final_consolidated_df = None

    try:
        ensure_output_directory_exists(cfg.output_dir)

        regional_df, sectors_df, regional_spillover_df, sector_spillover_df = load_data(cfg)

        base_gdp_df = None
        adj_gdp_df = None
        initial_sector_gdp_df = None
        final_sector_gdp_df = None
        tracking_df = None

        # Step 1: Base GDP (Tidy)
        base_gdp_df = calculate_base_regional_gdp_tidy(regional_df, cfg)

        # Step 2: Regional Spillovers (Tidy)
        adj_gdp_df = apply_regional_spillovers_tidy(base_gdp_df, regional_spillover_df)

        # Step 3: Sector Breakdown (Tidy)
        initial_sector_gdp_df = breakdown_gdp_by_sector_tidy(adj_gdp_df, sectors_df)

        # Step 4: Sector Spillovers (Einsum)
        final_sector_gdp_df = apply_sector_spillovers_einsum(initial_sector_gdp_df, sector_spillover_df)

        # Step 5: Consolidated Output (Tidy)
        final_consolidated_df = create_consolidated_output_tidy(final_sector_gdp_df)

        # --- Conditional Saving and Tracking ---
        # (Saving tidy intermediate results if requested)
        if generate_tracking:
            tracking_df = track_gdp_calculation_stages_tidy(
                base_gdp_df, adj_gdp_df, initial_sector_gdp_df, final_sector_gdp_df
            )
            if tracking_df is not None and save_intermediate:
                tracking_path = os.path.join(cfg.output_dir, "gdp_calculation_stages_tidy.csv")
                tracking_df.to_csv(tracking_path, index=False)
                logging.info(f"Tracking data saved to {tracking_path}")

        if save_intermediate:
            # Save intermediate results (now mostly tidy)
            if base_gdp_df is not None:
                base_gdp_path = os.path.join(cfg.output_dir, "base_gdp_tidy.csv")
                base_gdp_df.to_csv(base_gdp_path, index=False)
            if adj_gdp_df is not None:
                adj_gdp_path = os.path.join(cfg.output_dir, "adjusted_gdp_tidy.csv")
                adj_gdp_df.to_csv(adj_gdp_path, index=False)
            if final_sector_gdp_df is not None:
                sector_gdp_path = os.path.join(cfg.output_dir, "final_sector_gdp_tidy.csv")
                final_sector_gdp_df.to_csv(sector_gdp_path, index=False)
            logging.info(f"Intermediate tidy files saved to {cfg.output_dir} (if generated)")

        # Save final consolidated output
        if final_consolidated_df is not None and not final_consolidated_df.empty:
            consolidated_path = os.path.join(cfg.output_dir, cfg.final_consolidated_csv)
            final_consolidated_df.to_csv(consolidated_path, index=False)
            logging.info(f"Final consolidated results saved to {consolidated_path}")
        else:
             logging.warning("Final consolidated DataFrame is empty or None. File not saved.")


        overall_time = time.time() - overall_start_time
        logging.info(f"=== Regional GDP Analysis Completed in {overall_time:.2f} seconds ===")

    except Exception as e:
        logging.error(f"An error occurred during the main analysis workflow: {e}", exc_info=True)
        return None

    return final_consolidated_df


if __name__ == "__main__":
    # Use the Settings dataclass for configuration
    analysis_settings = Settings()

    # --- Create Dummy Data (Optional - if files don't exist) ---
    def create_dummy_data(cfg: Settings):
        logging.info("Checking for dummy data files...")
        # Use filenames from cfg
        if not os.path.exists(cfg.input_csv):
            logging.info(f"Creating dummy {cfg.input_csv}")
            pd.DataFrame([
                {'region': 'Riyadh Region', 'year': 2020, 'kpi': 'population', 'value': 8500000},
                {'region': 'Riyadh Region', 'year': 2020, 'kpi': 'investments', 'value': 120000000000},
                {'region': 'Riyadh Region', 'year': 2020, 'kpi': 'productivity', 'value': 50000},
                {'region': 'Riyadh Region', 'year': 2021, 'kpi': 'population', 'value': 8700000},
                {'region': 'Riyadh Region', 'year': 2021, 'kpi': 'investments', 'value': 130000000000},
                {'region': 'Riyadh Region', 'year': 2021, 'kpi': 'productivity', 'value': 52000},
                {'region': 'Makkah Region', 'year': 2020, 'kpi': 'population', 'value': 9000000},
                {'region': 'Makkah Region', 'year': 2020, 'kpi': 'investments', 'value': 80000000000},
                {'region': 'Makkah Region', 'year': 2020, 'kpi': 'productivity', 'value': 45000},
                {'region': 'Makkah Region', 'year': 2021, 'kpi': 'population', 'value': 9200000},
                {'region': 'Makkah Region', 'year': 2021, 'kpi': 'investments', 'value': 85000000000},
                {'region': 'Makkah Region', 'year': 2021, 'kpi': 'productivity', 'value': 46000},
            ]).to_csv(cfg.input_csv, index=False)

        if not os.path.exists(cfg.sectors_csv):
            logging.info(f"Creating dummy {cfg.sectors_csv}")
            pd.DataFrame([
                {'region': 'Riyadh Region', 'year': 2020, 'sector': 'A', 'pct_gdp': 50.0},
                {'region': 'Riyadh Region', 'year': 2020, 'sector': 'B', 'pct_gdp': 30.0},
                {'region': 'Riyadh Region', 'year': 2020, 'sector': 'C', 'pct_gdp': 20.0},
                {'region': 'Riyadh Region', 'year': 2021, 'sector': 'A', 'pct_gdp': 52.0},
                {'region': 'Riyadh Region', 'year': 2021, 'sector': 'B', 'pct_gdp': 28.0},
                {'region': 'Riyadh Region', 'year': 2021, 'sector': 'C', 'pct_gdp': 20.0},
                {'region': 'Makkah Region', 'year': 2020, 'sector': 'A', 'pct_gdp': 45.0},
                {'region': 'Makkah Region', 'year': 2020, 'sector': 'B', 'pct_gdp': 35.0},
                {'region': 'Makkah Region', 'year': 2020, 'sector': 'C', 'pct_gdp': 20.0},
                {'region': 'Makkah Region', 'year': 2021, 'sector': 'A', 'pct_gdp': 47.0},
                {'region': 'Makkah Region', 'year': 2021, 'sector': 'B', 'pct_gdp': 33.0},
                {'region': 'Makkah Region', 'year': 2021, 'sector': 'C', 'pct_gdp': 20.0},
            ]).to_csv(cfg.sectors_csv, index=False)

        if not os.path.exists(cfg.regional_spillover_csv):
             logging.info(f"Creating dummy {cfg.regional_spillover_csv}")
             pd.DataFrame({
                 'region_from': ['Riyadh Region', 'Makkah Region'],
                 'region_to': ['Makkah Region', 'Riyadh Region'],
                 'spillover_pct': [25.0, 27.0] # Assuming percentages
             }).to_csv(cfg.regional_spillover_csv, index=False)

        if not os.path.exists(cfg.sector_spillover_csv):
             logging.info(f"Creating dummy {cfg.sector_spillover_csv}")
             pd.DataFrame({
                 'A': [np.nan, 10.0, 5.0],
                 'B': [8.0, np.nan, 7.0],
                 'C': [6.0, 9.0, np.nan]
             }, index=pd.Index(['A', 'B', 'C'], name='sector_from')
             ).to_csv(cfg.sector_spillover_csv, index_label='') # Match example structure
        logging.info("Dummy data check complete.")

    create_dummy_data(analysis_settings)

    # --- Run Analysis ---
    final_gdp_data = run_regional_analysis(
        analysis_settings,
        save_intermediate=False, # Control saving intermediate tidy files
        generate_tracking=False  # Control generating tracking file
    )

    # --- Display Results ---
    if final_gdp_data is not None:
        print("=== Final Consolidated GDP Results (Sample) ===")
        print(final_gdp_data.head(20))

        # Display summaries extracted from the final consolidated data
        print("Total GDP by Region and Year:")
        region_summary = final_gdp_data[final_gdp_data['row_type'] == 'region_total']
        if not region_summary.empty:
            # Need to handle potential non-numeric if conversion failed upstream
            try:
                region_pivot = region_summary.pivot(index='region', columns='year', values='gdp')
                print(region_pivot)
            except Exception as e:
                print(f"Could not pivot region summary: {e}")
                print(region_summary) # Print raw data if pivot fails
        else:
            print("No region totals found.")

        print("Total GDP by Sector and Year:")
        sector_summary = final_gdp_data[final_gdp_data['row_type'] == 'sector_total']
        if not sector_summary.empty:
             try:
                 sector_pivot = sector_summary.pivot(index='sector', columns='year', values='gdp')
                 print(sector_pivot)
             except Exception as e:
                 print(f"Could not pivot sector summary: {e}")
                 print(sector_summary)
        else:
             print("No sector totals found.")

        print(f"Final consolidated output saved to: {os.path.join(analysis_settings.output_dir, analysis_settings.final_consolidated_csv)}")
        if not os.path.exists(os.path.join(analysis_settings.output_dir, "base_gdp_tidy.csv")):
             print("Intermediate files were not saved (as configured).")

    else:
        print("GDP Analysis failed to complete. Please check the logs.")