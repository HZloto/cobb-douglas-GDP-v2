import json
import subprocess
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ─────────────────────────── paths & mapping dicts
OUT = Path("outputs")
FRONT_JSON = OUT / "frontend.json"
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
    "AHU":"Al-Hudud","ABA":"Abha","NAJ":"Najran"
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
    st.session_state["prompt"] = prompt
    FRONT_JSON.unlink(missing_ok=True)
    with st.spinner("Running model…"):
        res = subprocess.run(
            ["python", "scenario_runner.py"],
            input=f"{prompt}\n",
            capture_output=True,
            text=True
        )
    if res.returncode or not FRONT_JSON.exists():
        st.error("Scenario runner failed"); st.code(res.stderr or res.stdout); st.stop()
    st.session_state["payload"] = json.loads(FRONT_JSON.read_text())
    st.success("Scenario completed ✅")

# ═════ display results ═════
if "payload" in st.session_state:
    p = st.session_state["payload"]
    nat = p["national_summary"]

    # helper to millions
    as_mn = lambda v: f"{v/1_000_000:,.0f} M SAR"

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Baseline GDP 2030", as_mn(nat["baseline_2030"]))
    k2.metric("Scenario GDP 2030", as_mn(nat["scenario_2030"]),
              delta=f"{nat['delta_abs_2030']/1_000_000:+,.0f} M SAR")
    k3.metric("Δ 2030 (%)", f"{nat['delta_pct_2030']:+.2f}%")
    k4.metric("CAGR Base 25-30", f"{nat['cagr_baseline']:+.2f}%")
    k5.metric("CAGR Scen 25-30", f"{nat['cagr_scenario']:+.2f}%")

    tab_ov, tab_reg, tab_dl = st.tabs(
        ["📊 Overview", "📍 Regional Impact", "📤 Downloads & Trace"]
    )

    # ─── OVERVIEW ────────────────────────────────────────────────────
    with tab_ov:
        st.info(p["trace"]["reasoning"])
        # national line
        yr = pd.DataFrame(p["yearly_national"])
        yr["year"] = yr["year"].astype(str)
        line = (
            alt.Chart(yr.melt("year", ["baseline","scenario"],
                              var_name="Series", value_name="Value"))
            .mark_line(point=True)
            .encode(
                x="year:N", y=alt.Y("Value:Q", title="GDP (SAR)"),
                color=alt.Color("Series:N",
                                scale=alt.Scale(range=["#1f77b4","#ff9900"]),
                                legend=alt.Legend(title="")),
                tooltip=["Series","year",alt.Tooltip("Value:Q",format=",")]
            )
            .properties(height=300)
        )
        st.altair_chart(line, use_container_width=True)

        # sector Δ bar
        sec = pd.DataFrame(p["sector_breakdown_2030"]).copy()
        sec["Sector"] = sec["sector"].apply(lambda c: SECTOR_MAP.get(c, c))
        sec["Δ ppt"] = sec["scenario_pct"] - sec["baseline_pct"]
        sec = sec.sort_values("Δ ppt")
        max_abs = sec["Δ ppt"].abs().max()
        bar = (
            alt.Chart(sec)
            .mark_bar()
            .encode(
                y=alt.Y("Sector:N", sort=sec["Sector"].tolist()),
                x=alt.X("Δ ppt:Q", title="Change in share (ppt)",
                        scale=alt.Scale(domain=[-max_abs, max_abs])),
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

   
    # ─── REGIONAL IMPACT ─────────────────────────────────────────────
    with tab_reg:
        # rebuild full region list from CSV for robustness
        top_df = pd.read_csv(CSV_FILES["Regional topline"])
        df = top_df.query("year == 2030 & regioncode != 'TOTAL'").copy()
        df["Δ SAR"] = df["GDP_SAR_scenario"] - df["GDP_SAR_baseline"]
        df["Δ %"] = 100*df["Δ SAR"] / df["GDP_SAR_baseline"].replace(0, pd.NA)
        df["Region"] = df["regioncode"].map(REGION_MAP).fillna(df["regioncode"])

        # decide split logic
        gains = df[df["Δ SAR"]>0].sort_values("Δ SAR", ascending=False)
        losses = df[df["Δ SAR"]<0].sort_values("Δ SAR")

        if gains.empty or losses.empty:
            st.write("### Top Movers (all regions moved in same direction)")
            movers = gains if losses.empty else losses
            st.dataframe(
                movers[["Region","Δ SAR","Δ %"]]
                .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}"}),
                use_container_width=True
            )
            st.altair_chart(
                alt.Chart(movers).mark_bar().encode(
                    y=alt.Y("Region:N", sort='-x'),
                    x=alt.X("Δ SAR:Q", title="Δ SAR"),
                    color=alt.value("#2ca02c" if gains.empty else "#d62728")
                ),
                use_container_width=True
            )
        else:
            c1,c2 = st.columns(2)
            with c1:
                st.write("### Biggest Declines")
                show = losses.head(3)
                st.dataframe(show[["Region","Δ SAR","Δ %"]]
                             .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}"}))
                st.altair_chart(
                    alt.Chart(show).mark_bar(color="#d62728").encode(
                        y=alt.Y("Region:N", sort='-x'),
                        x=alt.X("Δ SAR:Q", title="Δ SAR")),
                    use_container_width=True
                )
            with c2:
                st.write("### Biggest Gains")
                show = gains.head(3)
                st.dataframe(show[["Region","Δ SAR","Δ %"]]
                             .style.format({"Δ SAR":"{:,}","Δ %":"{:+.2f}"}))
                st.altair_chart(
                    alt.Chart(show).mark_bar(color="#2ca02c").encode(
                        y=alt.Y("Region:N", sort='-x'),
                        x=alt.X("Δ SAR:Q", title="Δ SAR")),
                    use_container_width=True
                )

    # ─── DOWNLOADS & TRACE ──────────────────────────────────────────
    with tab_dl:
        st.download_button("frontend.json",
                           data=json.dumps(p, indent=2),
                           file_name="frontend.json",
                           mime="application/json",
                           use_container_width=True)
        for lbl, path in CSV_FILES.items():
            if path.exists():
                st.download_button(lbl, data=path.read_bytes(),
                                   file_name=path.name, mime="text/csv",
                                   use_container_width=True)

        st.write("---")
        st.markdown("### Impact rules")
        rules = pd.DataFrame(p["trace"]["rules"])
        rules["sector"] = rules["sector"].map(SECTOR_MAP).fillna(rules["sector"])
        st.dataframe(rules, use_container_width=True)
