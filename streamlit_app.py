# streamlit_app.py

import json
# import subprocess # <-- REMOVED
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np # <-- Keep numpy as it's used by pandas and potentially baseline_model/apply_rules

# Import the main function from your scenario_runner script
# Make sure scenario_runner.py, agent.py, and baseline_model.py are in the same directory
try:
    from scenario_runner import run_scenario
    # Assuming agent and baseline_model are needed elsewhere or imported by run_scenario
    # import agent
    # import baseline_model as bm
except ImportError as e:
    st.error(f"Error importing scenario runner components: {e}. Please ensure scenario_runner.py, agent.py, and baseline_model.py are in the same directory as the Streamlit app.")
    st.stop() # Stop execution if core modules cannot be imported

# ─────────────────────────── paths & mapping dicts
OUT = Path("outputs")
# FRONT_JSON is no longer read directly, but keep OUT for saving other files
# FRONT_JSON = OUT / "frontend.json" # <-- REMOVED direct use

# These paths are for the download buttons, assuming run_scenario writes these files
CSV_FILES = {
    "National impact": OUT / "gdp_impact_summary.csv",
    "Sector summary":  OUT / "sector_summary.csv",
    "Regional topline": OUT / "topline_gdp.csv",
}

SECTOR_MAP = {
    "A": "Agriculture",  "B": "Mining & Oil",     "C": "Manufacturing",
    "D": "Utilities",    "E": "Water",            "F": "Construction",
    "G": "Wholesale/Retail", "H": "Transport",    "I": "Accommodation & Food",
    "J": "ICT",          "K": "Finance",          "L": "Real Estate",
    "M": "Sci-Tech",     "N": "Admin Services",   "O": "Public Admin",
    "P": "Education",    "Q": "Health",           "R": "Culture & Sports",
    "S": "Other Services","T": "Households",      "U": "Extraterritorial",
    "V": "Unclassified"
}
REGION_MAP = {
    "ARI":"Riyadh","MAK":"Makkah","ASH":"Eastern",
    "AHU":"Al-Hudud","ABA":"Abha","NAJ":"Najran",
    # extend as needed
}

st.set_page_config(page_title="Saudi GDP Scenario Simulator",
                   layout="wide", page_icon="📈")
st.title("Saudi GDP Scenario Simulator")

# ───────── prompt & run
prompt = st.text_area("Describe the policy you want to test:",
                      value=st.session_state.get("prompt",""),
                      height=110,
                      placeholder="e.g. ICT workforce doubles in Riyadh by 2030")
run_btn = st.button("🚀 Run Scenario", use_container_width=True,
                    disabled=not prompt.strip())

# ═════ run model once ═════
if run_btn:
    # Store prompt in session state
    st.session_state["prompt"] = prompt

    # Clear previous results if any
    if "payload" in st.session_state:
         del st.session_state["payload"]

    # Remove old output files before generating new ones (optional but good practice)
    for path in CSV_FILES.values():
        path.unlink(missing_ok=True)
    trace_path = OUT / "trace.json"
    trace_path.unlink(missing_ok=True)
    # You could also remove frontend.json if run_scenario saves it, but the main logic won't rely on it now

    with st.spinner("Running model…"):
        try:
            # --- Call the scenario runner function directly ---
            # Assumes run_scenario function exists in scenario_runner.py
            # and returns the payload dictionary.
            # Assumes run_scenario handles writing the output CSVs and trace.json
            payload = run_scenario(prompt, input_dir="inputs", output_dir="outputs") # Pass paths explicitly
            # --- Store the results in session state ---
            st.session_state["payload"] = payload
            st.success("Scenario completed ✅")

        except FileNotFoundError as e:
             st.error(f"Input data error: {e}. Please ensure 'inputs/input_data.csv' exists.")
             st.stop() # Stop execution on this specific error
        except RuntimeError as e:
             # Catch errors raised by run_scenario for agent or computation issues
             st.error(f"Scenario execution failed: {e}")
             # The print statements inside run_scenario will appear in logs
             st.stop()
        except Exception as e:
            # Catch any other unexpected errors
            st.error(f"An unexpected error occurred during scenario execution: {e}")
            # Optional: import traceback and show it for debugging
            # import traceback
            # st.code(traceback.format_exc())
            st.stop()


# ═════ display results ═════
# Check if the payload is in session state before attempting to display
if "payload" in st.session_state:
    p = st.session_state["payload"]

    # Check if the payload has the expected structure
    if not isinstance(p, dict) or "national_summary" not in p:
        st.error("Invalid scenario results received.")
        # Clear invalid payload to prevent repeated errors
        del st.session_state["payload"]
        st.stop()

    nat = p["national_summary"]

    # helper to millions
    # Ensure nat values are numeric before formatting, or handle potential errors
    try:
        as_mn = lambda v: f"{v/1_000_000:,.0f} M SAR"
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Baseline GDP 2030", as_mn(nat.get("baseline_2030", 0))) # Use .get with default 0
        k2.metric("Scenario GDP 2030", as_mn(nat.get("scenario_2030", 0)),
                  delta=f"{nat.get('delta_abs_2030', 0)/1_000_000:+,.0f} M SAR") # Use .get
        k3.metric("Δ 2030 (%)", f"{nat.get('delta_pct_2030', 0):+.2f}%") # Use .get
        k4.metric("CAGR Base 25-30", f"{nat.get('cagr_baseline', 0):+.2f}%") # Use .get
        k5.metric("CAGR Scen 25-30", f"{nat.get('cagr_scenario', 0):+.2f}%") # Use .get
    except Exception as e:
        st.warning(f"Could not display key metrics. Data format might be incorrect: {e}")
        # You might want to inspect the payload here if this happens
        # st.json(p)

    tab_ov, tab_reg, tab_dl = st.tabs(
        ["📊 Overview", "📍 Regional Impact", "📤 Downloads & Trace"]
    )

    # ─── OVERVIEW ────────────────────────────────────────────────────
    with tab_ov:
        # Access trace data directly from payload
        reasoning = p.get("trace", {}).get("reasoning", "No reasoning provided.")
        if reasoning:
             st.info(reasoning)
        else:
             st.warning("Agent reasoning not available in the results.")

        # national line
        yearly_data = p.get("yearly_national", [])
        if yearly_data:
            yr = pd.DataFrame(yearly_data)
            if not yr.empty and all(col in yr.columns for col in ["year", "baseline", "scenario"]):
                 yr["year"] = yr["year"].astype(str)
                 line = (
                     alt.Chart(yr.melt("year", ["baseline","scenario"],
                                       var_name="Series", value_name="Value"))
                     .mark_line(point=True)
                     .encode(
                         x=alt.X("year:N", title="Year"), # Added title
                         y=alt.Y("Value:Q", title="GDP (SAR)"),
                         color=alt.Color("Series:N",
                                         scale=alt.Scale(range=["#1f77b4","#ff9900"]),
                                         legend=alt.Legend(title="")),
                         tooltip=["Series","year",alt.Tooltip("Value:Q",format=",")]
                     )
                     .properties(height=300, title="National GDP: Baseline vs Scenario") # Added title
                 )
                 st.altair_chart(line, use_container_width=True)
            else:
                 st.warning("Yearly national data not in expected format.")
                 # st.json(yearly_data) # Uncomment to debug data structure
        else:
            st.warning("Yearly national data not available in the results.")


        # sector Δ bar
        sector_breakdown_data = p.get("sector_breakdown_2030", [])
        if sector_breakdown_data:
            sec = pd.DataFrame(sector_breakdown_data).copy()
            if not sec.empty and all(col in sec.columns for col in ["sector", "scenario_pct", "baseline_pct"]):
                sec["Sector"] = sec["sector"].apply(lambda c: SECTOR_MAP.get(c, c))
                sec["Δ ppt"] = sec["scenario_pct"] - sec["baseline_pct"]
                sec = sec.sort_values("Δ ppt")
                # Handle case where delta_pp might be all zeros or NaN
                if not sec["Δ ppt"].abs().sum() == 0:
                    max_abs = sec["Δ ppt"].abs().max()
                    domain = [-max_abs, max_abs]
                else:
                     max_abs = 0.1 # Default domain if no change
                     domain = [-max_abs, max_abs]

                bar = (
                    alt.Chart(sec)
                    .mark_bar()
                    .encode(
                        y=alt.Y("Sector:N", sort=sec["Sector"].tolist(), title="Sector"), # Added title
                        x=alt.X("Δ ppt:Q", title="Change in share (ppt)",
                                scale=alt.Scale(domain=domain)), # Use calculated domain
                        color=alt.condition("datum['Δ ppt']>0",
                                            alt.value("#2ca02c"), alt.value("#d62728")),
                        tooltip=[
                            "Sector",
                            alt.Tooltip("baseline_pct:Q", format=".2f", title="Baseline %"),
                            alt.Tooltip("scenario_pct:Q", format=".2f", title="Scenario %"),
                            alt.Tooltip("Δ ppt:Q", format="+.2f")
                        ]
                    )
                    .properties(height=max(320, 18*len(sec)),
                                title="Sector Share Change, 2030")
                )
                st.altair_chart(bar, use_container_width=True)
            else:
                st.warning("Sector breakdown data not in expected format.")
                # st.json(sector_breakdown_data) # Uncomment to debug
        else:
             st.warning("Sector breakdown data not available in the results.")


    # ─── REGIONAL IMPACT ─────────────────────────────────────────────
    with tab_reg:
        # Use the CSV file generated by run_scenario for robustness as originally intended
        # This ensures the download buttons and this section use the same data source
        # Alternative: use regional_impacts_2030 from payload (top_negative/top_positive)
        # but reading the full CSV is more flexible for displaying top movers
        top_df_path = CSV_FILES["Regional topline"]

        if top_df_path.exists():
            try:
                top_df = pd.read_csv(top_df_path)
                df = top_df.query("year == 2030 & regioncode != 'TOTAL'").copy()
                if not df.empty:
                    # Ensure calculations handle potential zero/NA baseline GDP
                    df["Δ SAR"] = df["GDP_SAR_scenario"] - df["GDP_SAR_baseline"]
                    df["Δ %"] = 100*df["Δ SAR"] / df["GDP_SAR_baseline"].replace(0, np.nan) # Use np.nan for proper NA handling
                    df["Region"] = df["regioncode"].map(REGION_MAP).fillna(df["regioncode"])

                    # decide split logic
                    gains = df[df["Δ SAR"]>0].sort_values("Δ SAR", ascending=False)
                    losses = df[df["Δ SAR"]<0].sort_values("Δ SAR")

                    if gains.empty and losses.empty:
                        st.info("No regional GDP changes detected in 2030.")
                    elif gains.empty or losses.empty:
                        st.write("### Regional Movers")
                        movers = gains if losses.empty else losses
                        st.dataframe(
                            movers[["Region","Δ SAR","Δ %"]]
                            .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}%"}), # Added '%' format
                            use_container_width=True
                        )
                        if not movers.empty:
                             st.altair_chart(
                                 alt.Chart(movers).mark_bar().encode(
                                     y=alt.Y("Region:N", sort='-x', title="Region"), # Added title
                                     x=alt.X("Δ SAR:Q", title="Δ SAR (SAR)"), # Added title
                                     color=alt.value("#2ca02c" if gains.empty else "#d62728")
                                 ).properties(title="Regional GDP Change, 2030"), # Added title
                                 use_container_width=True
                             )
                    else:
                        c1,c2 = st.columns(2)
                        with c1:
                            st.write("### Biggest Declines (Top 3)")
                            show_losses = losses.head(3)
                            st.dataframe(show_losses[["Region","Δ SAR","Δ %"]]
                                         .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}%"})) # Added '%' format
                            if not show_losses.empty:
                                 st.altair_chart(
                                     alt.Chart(show_losses).mark_bar(color="#d62728").encode(
                                         y=alt.Y("Region:N", sort='-x', title="Region"), # Added title
                                         x=alt.X("Δ SAR:Q", title="Δ SAR (SAR)")), # Added title
                                     use_container_width=True
                                 )

                        with c2:
                            st.write("### Biggest Gains (Top 3)")
                            show_gains = gains.head(3)
                            st.dataframe(show_gains[["Region","Δ SAR","Δ %"]]
                                         .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}%"})) # Added '%' format
                            if not show_gains.empty:
                                 st.altair_chart(
                                     alt.Chart(show_gains).mark_bar(color="#2ca02c").encode(
                                         y=alt.Y("Region:N", sort='-x', title="Region"), # Added title
                                         x=alt.X("Δ SAR:Q", title="Δ SAR (SAR)")), # Added title
                                     use_container_width=True
                                 )
                else:
                     st.warning("Regional topline data for 2030 is empty.")

            except Exception as e:
                 st.error(f"Error processing regional impact data: {e}")
                 # Optional: Inspect the dataframe if load works but processing fails
                 # st.dataframe(top_df)
        else:
             st.warning("Regional topline CSV file not found. Run a scenario first.")


    # ─── DOWNLOADS & TRACE ──────────────────────────────────────────
    with tab_dl:
        # Download frontend payload directly from session state
        # You don't need to read a file anymore
        st.download_button("frontend.json (Generated)",
                           data=json.dumps(p, indent=2),
                           file_name="frontend.json",
                           mime="application/json",
                           use_container_width=True)

        # Download CSV files - these should have been saved by run_scenario
        for lbl, path in CSV_FILES.items():
            if path.exists():
                try:
                    with open(path, "rb") as f: # Open in binary mode for download button
                         st.download_button(f"Download {lbl} CSV",
                                            data=f.read(),
                                            file_name=path.name,
                                            mime="text/csv",
                                            use_container_width=True)
                except Exception as e:
                     st.error(f"Error reading {path.name} for download: {e}")
            else:
                 st.info(f"{path.name} not found. Run a scenario first.")


        st.write("---")
        st.markdown("### Impact rules (from Trace)")
        # Access rules data directly from the payload
        rules_data = p.get("trace", {}).get("rules", []) # Get rules list safely
        if rules_data:
            try:
                rules_df = pd.DataFrame(rules_data)
                # Ensure 'sector' column exists before mapping
                if 'sector' in rules_df.columns:
                     rules_df["sector"] = rules_df["sector"].map(SECTOR_MAP).fillna(rules_df["sector"])
                st.dataframe(rules_df, use_container_width=True)
            except Exception as e:
                 st.warning(f"Could not display impact rules: {e}. Rules data structure might be unexpected.")
                 # Optional: st.json(rules_data) to inspect the data
        else:
            st.info("No detailed impact rules available in the trace data.")