import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import altair as alt

st.set_page_config(page_title="DCF + Reverse DCF (Fixed-step)", layout="wide")
st.title("DCF + Reverse DCF (Fixed-step solver + visuals)")

st.caption(
    "Patched to avoid 'running forever' by gating heavy visuals behind a button, "
    "adding progress, safer defaults, and early-stopping in the fixed-step solver."
)

# -----------------------------
# Cached Yahoo pulls (serializable)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_yf_data(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info or {}
    bs = t.balancesheet
    cf = t.cash_flow
    return info, bs, cf

# -----------------------------
# Helpers
# -----------------------------
def safe_get(info: Dict[str, Any], key: str, default=None):
    v = info.get(key, default)
    return default if v is None else v

def safe_row_value(df: pd.DataFrame, row_name: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    if row_name not in df.index:
        return None
    vals = df.loc[row_name].dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[0])

def to_number(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def summarize_money(x: float) -> str:
    if x is None or np.isnan(x):
        return "NaN"
    absx = abs(x)
    if absx >= 1e12:
        return f"{x/1e12:,.2f} T"
    if absx >= 1e9:
        return f"{x/1e9:,.2f} B"
    if absx >= 1e6:
        return f"{x/1e6:,.2f} M"
    if absx >= 1e3:
        return f"{x/1e3:,.2f} K"
    return f"{x:,.2f}"

def compute_pv_paths(
    base_fcf: float,
    year_growth: int,
    year_sustain: int,
    gr: float,
    sustain_ratio: float,
    dr: float
) -> Tuple[pd.DataFrame, float]:
    years, fcf, disc, pv = [], [], [], []

    for i in range(1, year_growth + 1):
        cf = base_fcf * ((1 + gr) ** i)
        pv_i = cf / ((1 + dr) ** i)
        years.append(i); fcf.append(cf); disc.append((1 + dr) ** i); pv.append(pv_i)

    for j in range(1, year_sustain + 1):
        i = year_growth + j
        cf = base_fcf * ((1 + gr) ** year_growth) * ((1 + (gr * sustain_ratio)) ** j)
        pv_i = cf / ((1 + dr) ** i)
        years.append(i); fcf.append(cf); disc.append((1 + dr) ** i); pv.append(pv_i)

    df = pd.DataFrame({"Year": years, "FCF": fcf, "DiscountFactor": disc, "PV": pv})
    return df, float(df["PV"].sum())

def terminal_value_pv(
    base_fcf: float,
    year_growth: int,
    year_sustain: int,
    gr: float,
    sustain_ratio: float,
    dr: float,
    tgr: float
) -> float:
    fcf_T = base_fcf * ((1 + gr) ** year_growth) * ((1 + (gr * sustain_ratio)) ** year_sustain)
    tv = fcf_T * (1 + tgr) / (dr - tgr)
    return float(tv / ((1 + dr) ** (year_growth + year_sustain)))

def intrinsic_value_per_share(
    shares: float,
    net_debt: float,
    base_fcf: float,
    year_growth: int,
    year_sustain: int,
    sustain_ratio: float,
    dr: float,
    tgr: float,
    gr: float
) -> float:
    if shares <= 0 or base_fcf is None or np.isnan(base_fcf):
        return np.nan
    if dr <= tgr:
        return np.nan

    _, pv_sum = compute_pv_paths(base_fcf, year_growth, year_sustain, gr, sustain_ratio, dr)
    pv_tv = terminal_value_pv(base_fcf, year_growth, year_sustain, gr, sustain_ratio, dr, tgr)
    equity_value = pv_sum + pv_tv - net_debt
    return float(equity_value / shares)

def diff_vs_price(
    shares: float,
    net_debt: float,
    base_fcf: float,
    year_growth: int,
    year_sustain: int,
    sustain_ratio: float,
    dr: float,
    tgr: float,
    current_price: float,
    gr: float
) -> float:
    iv = intrinsic_value_per_share(shares, net_debt, base_fcf, year_growth, year_sustain, sustain_ratio, dr, tgr, gr)
    if np.isnan(iv) or current_price <= 0:
        return np.nan
    return (iv - current_price) / current_price

def solve_implied_growth_fixed_step(
    shares: float,
    net_debt: float,
    base_fcf: float,
    year_growth: int,
    year_sustain: int,
    sustain_ratio: float,
    dr: float,
    tgr: float,
    current_price: float,
    initial_guess: float = 0.01,
    step: float = 0.01,
    tolerance: float = 0.01,
    max_iterations: int = 20000,
    gr_min: float = -0.50,
    gr_max: float = 1.50,
    stagnation_patience: int = 800,
    min_improvement: float = 0.9995
) -> float:
    # Fixed-step update: gr <- gr - step * f(gr)  (NOT Newton-Raphson)
    if dr <= tgr or current_price <= 0:
        return np.nan

    x = float(initial_guess)
    x = max(gr_min, min(gr_max, x))

    prev_abs = None
    stagnant = 0

    for _ in range(int(max_iterations)):
        fx = diff_vs_price(
            shares=shares,
            net_debt=net_debt,
            base_fcf=base_fcf,
            year_growth=year_growth,
            year_sustain=year_sustain,
            sustain_ratio=sustain_ratio,
            dr=dr,
            tgr=tgr,
            current_price=current_price,
            gr=x
        )
        if np.isnan(fx):
            return np.nan

        abs_fx = abs(float(fx))
        if abs_fx < tolerance:
            return float(x)

        # early-stop if not improving
        if prev_abs is not None:
            if abs_fx >= prev_abs * float(min_improvement):
                stagnant += 1
            else:
                stagnant = 0
        prev_abs = abs_fx
        if stagnant >= int(stagnation_patience):
            break

        x = x - float(step) * float(fx)
        x = max(gr_min, min(gr_max, x))

    return float(x)

def infer_net_debt(info: Dict[str, Any], balancesheet: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    components: Dict[str, float] = {}

    cash = safe_row_value(balancesheet, "Cash And Cash Equivalents")
    if cash is None:
        cash = safe_row_value(balancesheet, "Cash Cash Equivalents And Short Term Investments")
    components["cash"] = to_number(cash)

    long_debt = safe_row_value(balancesheet, "Long Term Debt")
    if long_debt is None:
        long_debt = safe_row_value(balancesheet, "Long Term Debt And Capital Lease Obligation")
    components["long_debt"] = to_number(long_debt)

    short_debt = safe_row_value(balancesheet, "Current Debt")
    if short_debt is None:
        short_debt = safe_row_value(balancesheet, "Short Long Term Debt")
    if short_debt is None:
        short_debt = safe_row_value(balancesheet, "Current Debt And Capital Lease Obligation")
    components["short_debt"] = to_number(short_debt)

    if not np.isnan(components["cash"]) and (not np.isnan(components["long_debt"]) or not np.isnan(components["short_debt"])):
        total_debt = np.nan_to_num(components["long_debt"], nan=0.0) + np.nan_to_num(components["short_debt"], nan=0.0)
        net_debt = total_debt - components["cash"]
        components["total_debt"] = float(total_debt)
        components["net_debt"] = float(net_debt)
        return float(net_debt), components

    nd = safe_get(info, "netDebt", None)
    if nd is not None:
        components["netDebt_info"] = to_number(nd)
        return float(to_number(nd)), components

    tnc = safe_row_value(balancesheet, "Total Non Current Liabilities Net Minority Interest")
    if tnc is None:
        tnc = safe_row_value(balancesheet, "Total Non Current Liabilities")
    components["total_non_current_liabilities"] = to_number(tnc)

    if not np.isnan(components["total_non_current_liabilities"]) and not np.isnan(components["cash"]):
        approx = components["total_non_current_liabilities"] - components["cash"]
        components["approx_net_liabilities"] = float(approx)
        return float(approx), components

    return 0.0, components

def get_fcf_series(cashflow: pd.DataFrame) -> pd.Series:
    if cashflow is None or cashflow.empty:
        return pd.Series(dtype=float)
    if "Free Cash Flow" not in cashflow.index:
        return pd.Series(dtype=float)
    s = cashflow.loc["Free Cash Flow"].dropna()
    try:
        s.index = pd.to_datetime(s.index)
        s = s.sort_index(ascending=True)
    except Exception:
        pass
    return s.astype(float)

# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker (Yahoo Finance)", value="mc.bk").strip()
    info, balancesheet, cashflow = load_yf_data(ticker)

    sector = safe_get(info, "sector", "Unknown")
    industry = safe_get(info, "industry", "Unknown")
    st.markdown(f"**Sector:** {sector}")
    st.markdown(f"**Industry:** {industry}")
    if sector == "Financial Services":
        st.warning("Financial Services: FCF-based DCF can be unreliable. Interpret cautiously.")

    shares_default = safe_get(info, "sharesOutstanding", None)
    if shares_default is None:
        shares_default = 1.0
        st.info("Shares outstanding missing from Yahoo. Defaulting to 1; please set manually.")
    input_shares = st.number_input("Shares Outstanding", value=float(shares_default), min_value=0.0, format="%.0f")

    inferred_net_debt, nd_components = infer_net_debt(info, balancesheet)
    input_net_debt = st.number_input(
        "Net Debt (Debt − Cash) / Net Liabilities (fallback)",
        value=float(inferred_net_debt),
        format="%.2f"
    )

    fcf_series = get_fcf_series(cashflow)
    st.subheader("Base FCF")
    years_to_use = st.slider("Years of history to use", 1, 10, 4)
    agg_method = st.selectbox("Aggregation method", ["Mean", "Median"], index=0)

    if len(fcf_series) > 0:
        recent = fcf_series.tail(years_to_use)
        base_fcf_auto = float(recent.mean()) if agg_method == "Mean" else float(recent.median())
    else:
        base_fcf_auto = 0.0
        st.info("FCF series missing. Please input Base FCF manually.")
    input_base_fcf = st.number_input("Base Free Cash Flow (FCF)", value=float(base_fcf_auto), format="%.2f")

    st.subheader("Growth assumptions")
    input_year_growth = int(st.number_input("Years (high growth)", value=3, min_value=0, step=1))
    input_year_sustain = int(st.number_input("Years (sustaining growth)", value=7, min_value=0, step=1))
    input_growthrate = st.number_input("High-growth rate (gr)", value=0.07, format="%.4f")
    input_growth_to_sustain_ratio = st.number_input("Sustain ratio", value=0.50, format="%.4f")

    st.subheader("Discounting")
    input_terminalgrowthrate = st.number_input("Terminal growth rate (tgr)", value=0.03, format="%.4f")
    input_discountrate = st.number_input("Discount rate (dr)", value=0.06, format="%.4f")

    st.subheader("Reverse DCF & performance controls")
    gd_step = st.number_input("Fixed step size", value=0.01, min_value=0.0001, format="%.4f")
    gd_tol = st.number_input("Tolerance (relative error)", value=0.01, min_value=0.00001, format="%.5f")
    gd_max_iter = st.number_input("Max iterations", value=20000, min_value=1000, step=1000)
    reverse_points = st.slider("Reverse DCF points (dr grid size)", min_value=8, max_value=60, value=20, step=1)

    enable_sensitivity = st.checkbox("Enable sensitivity table", value=True)
    enable_monte_carlo = st.checkbox("Enable Monte Carlo", value=False)

    run_visuals = st.button("Run Reverse DCF & Visuals")

# -----------------------------
# Main Layout
# -----------------------------
colA, colB = st.columns([1, 1])

with colA:
    st.header("DCF Result")

    current_price = safe_get(info, "currentPrice", None)
    if current_price is None:
        st.warning("Current price missing from Yahoo. Reverse DCF needs current price.")
        current_price = np.nan

    iv = intrinsic_value_per_share(
        shares=input_shares,
        net_debt=input_net_debt,
        base_fcf=input_base_fcf,
        year_growth=input_year_growth,
        year_sustain=input_year_sustain,
        sustain_ratio=input_growth_to_sustain_ratio,
        dr=input_discountrate,
        tgr=input_terminalgrowthrate,
        gr=input_growthrate
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Intrinsic value / share", "NaN" if np.isnan(iv) else f"{iv:,.4f}")
    m2.metric("Current price", "NaN" if np.isnan(current_price) else f"{current_price:,.4f}")
    if not (np.isnan(iv) or np.isnan(current_price) or current_price == 0):
        mos = (current_price - iv) * 100 / current_price
        m3.metric("Margin of safety (%)", f"{mos:,.2f}%")
    else:
        m3.metric("Margin of safety (%)", "NaN")

    with st.expander("Balance sheet components used (net debt inference)"):
        st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in nd_components.items()})

    if len(fcf_series) > 0:
        st.subheader("FCF history")
        df_fcf = fcf_series.rename("Free Cash Flow").to_frame()
        df_fcf.index = df_fcf.index.astype(str)
        st.line_chart(df_fcf)

    if not np.isnan(iv):
        paths_df, pv_sum = compute_pv_paths(
            base_fcf=input_base_fcf,
            year_growth=input_year_growth,
            year_sustain=input_year_sustain,
            gr=input_growthrate,
            sustain_ratio=input_growth_to_sustain_ratio,
            dr=input_discountrate
        )
        pv_tv = terminal_value_pv(
            base_fcf=input_base_fcf,
            year_growth=input_year_growth,
            year_sustain=input_year_sustain,
            gr=input_growthrate,
            sustain_ratio=input_growth_to_sustain_ratio,
            dr=input_discountrate,
            tgr=input_terminalgrowthrate
        )

        st.subheader("Projected FCF & PV table")
        show_df = paths_df.copy()
        show_df["FCF"] = show_df["FCF"].map(lambda x: f"{x:,.2f}")
        show_df["PV"] = show_df["PV"].map(lambda x: f"{x:,.2f}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
**PV of explicit forecast:** {summarize_money(pv_sum)}  
**PV of terminal value:** {summarize_money(pv_tv)}  
**Net debt / liabilities:** {summarize_money(input_net_debt)}
"""
        )

with colB:
    st.header("Reverse DCF & Sensitivity Visuals")

    if np.isnan(current_price) or current_price <= 0:
        st.info("Reverse DCF visualizations require a valid current price.")
    elif not run_visuals:
        st.info("Click **Run Reverse DCF & Visuals** to compute the reverse curve (and optional sensitivity / Monte Carlo).")
    else:
        # Reverse DCF curve (implied gr vs dr)
        dr_start = float(input_terminalgrowthrate) + 0.01
        dr_grid = np.round(np.linspace(dr_start, 0.30, int(reverse_points)), 4)

        implied = []
        prog = st.progress(0)
        status = st.empty()

        for i, dr in enumerate(dr_grid):
            status.write(f"Computing implied growth… {i+1}/{len(dr_grid)} (dr={dr:.2%})")
            g = solve_implied_growth_fixed_step(
                shares=input_shares,
                net_debt=input_net_debt,
                base_fcf=input_base_fcf,
                year_growth=input_year_growth,
                year_sustain=input_year_sustain,
                sustain_ratio=input_growth_to_sustain_ratio,
                dr=float(dr),
                tgr=float(input_terminalgrowthrate),
                current_price=float(current_price),
                initial_guess=input_growthrate,
                step=float(gd_step),
                tolerance=float(gd_tol),
                max_iterations=int(gd_max_iter),
                gr_min=-0.50,
                gr_max=1.50
            )
            implied.append(g)
            prog.progress((i + 1) / len(dr_grid))

        status.empty()

        df_curve = pd.DataFrame({"Discount rate": dr_grid, "Implied growth (gr)": implied})

        st.subheader("Implied growth vs discount rate (Reverse DCF)")
        chart = (
            alt.Chart(df_curve)
            .mark_line()
            .encode(
                x=alt.X("Discount rate:Q", title="Discount rate (dr)", axis=alt.Axis(format=".0%")),
                y=alt.Y("Implied growth (gr):Q", title="Implied FCF growth rate (gr)", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("Discount rate:Q", format=".2%"),
                    alt.Tooltip("Implied growth (gr):Q", format=".2%")
                ],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        # Sensitivity table (optional)
        if enable_sensitivity:
            st.subheader("Sensitivity table (Intrinsic value / share)")
            st.caption(
                "Rows vary the **Discount Rate (dr)**, columns vary the **FCF Growth Rate (gr)**. "
                "Each cell shows the intrinsic value per share from a full DCF."
            )

            gr_list = np.round(np.linspace(-0.10, 0.25, 15), 4)
            dr_list = np.round(np.linspace(max(input_terminalgrowthrate + 0.01, 0.03), 0.20, 13), 4)

            mat = []
            for dr in dr_list:
                row = []
                for gr in gr_list:
                    val = intrinsic_value_per_share(
                        shares=input_shares,
                        net_debt=input_net_debt,
                        base_fcf=input_base_fcf,
                        year_growth=input_year_growth,
                        year_sustain=input_year_sustain,
                        sustain_ratio=input_growth_to_sustain_ratio,
                        dr=float(dr),
                        tgr=float(input_terminalgrowthrate),
                        gr=float(gr)
                    )
                    row.append(val)
                mat.append(row)

            df_sens = pd.DataFrame(mat, index=[f"{x:.2%}" for x in dr_list], columns=[f"{x:.2%}" for x in gr_list])
            df_sens.index.name = "Discount Rate (dr)"
            df_sens.columns.name = "Growth Rate (gr)"
            st.dataframe(df_sens, use_container_width=True)
            st.caption("Values are intrinsic value **per share**, in the same currency as Yahoo Finance price data.")

        # Monte Carlo (optional, and only after button press)
        if enable_monte_carlo:
            st.subheader("Monte Carlo DCF (intrinsic value distribution)")
            with st.expander("Monte Carlo settings"):
                n_sims = st.slider("Simulations", 200, 5000, 1000, 200)
                gr_mu = st.number_input("gr mean", value=float(input_growthrate), format="%.4f")
                gr_sigma = st.number_input("gr std dev", value=0.03, min_value=0.0, format="%.4f")
                dr_mu = st.number_input("dr mean", value=float(input_discountrate), format="%.4f")
                dr_sigma = st.number_input("dr std dev", value=0.02, min_value=0.0, format="%.4f")
                tgr_mu = st.number_input("tgr mean", value=float(input_terminalgrowthrate), format="%.4f")
                tgr_sigma = st.number_input("tgr std dev", value=0.01, min_value=0.0, format="%.4f")
                fcf_mu = st.number_input("Base FCF mean", value=float(input_base_fcf), format="%.2f")
                fcf_sigma = st.number_input(
                    "Base FCF std dev",
                    value=abs(float(input_base_fcf)) * 0.15 if input_base_fcf != 0 else 1.0,
                    min_value=0.0,
                    format="%.2f"
                )

            rng = np.random.default_rng(42)
            gr_draw = np.clip(rng.normal(gr_mu, gr_sigma, n_sims), -0.50, 1.50)
            tgr_draw = np.clip(rng.normal(tgr_mu, tgr_sigma, n_sims), -0.05, 0.06)
            dr_draw = np.clip(rng.normal(dr_mu, dr_sigma, n_sims), 0.01, 0.40)
            dr_draw = np.maximum(dr_draw, tgr_draw + 0.005)
            fcf_draw = rng.normal(fcf_mu, fcf_sigma, n_sims)

            sims = []
            for i in range(n_sims):
                sims.append(
                    intrinsic_value_per_share(
                        shares=input_shares,
                        net_debt=input_net_debt,
                        base_fcf=float(fcf_draw[i]),
                        year_growth=input_year_growth,
                        year_sustain=input_year_sustain,
                        sustain_ratio=input_growth_to_sustain_ratio,
                        dr=float(dr_draw[i]),
                        tgr=float(tgr_draw[i]),
                        gr=float(gr_draw[i])
                    )
                )
            sims = np.array(sims, dtype=float)
            sims = sims[~np.isnan(sims)]

            if len(sims) == 0:
                st.warning("Monte Carlo produced no valid simulations (check inputs).")
            else:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("P5", f"{np.percentile(sims, 5):,.4f}")
                s2.metric("P50", f"{np.percentile(sims, 50):,.4f}")
                s3.metric("P95", f"{np.percentile(sims, 95):,.4f}")
                s4.metric("P(IV > Price)", "NaN" if np.isnan(current_price) else f"{float(np.mean(sims > current_price)):.1%}")

                counts = pd.cut(pd.Series(sims), bins=30).value_counts().sort_index()
                df_hist = pd.DataFrame({"Count": counts.values}, index=counts.index.astype(str))
                st.bar_chart(df_hist)

st.divider()
st.subheader("Quick stats pulled from Yahoo (best-effort)")
stats_keys = [
    "marketCap", "enterpriseValue", "totalCash", "totalDebt",
    "profitMargins", "priceToBook", "debtToEquity",
    "returnOnEquity", "currentRatio", "trailingEps", "dividendRate"
]
left, right = {}, {}
for idx, k in enumerate(stats_keys):
    v = safe_get(info, k, None)
    (left if idx % 2 == 0 else right)[k] = v

c1, c2 = st.columns(2)
with c1:
    st.json(left)
with c2:
    st.json(right)

st.caption("Note: Yahoo Finance fields can be missing/inconsistent depending on ticker/exchange.")
