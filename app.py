import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from dune_api import (
    fetch_dune_results,
    fetch_execution_results,
    wait_for_execution,
    try_run_and_fetch,
)

load_dotenv()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Pump.fun Regime Dashboard", layout="wide")
st.title("Pump.fun Regime Dashboard")
st.caption(
    "Signal-first view: ratios vs rolling median + regime score. "
    "Core thesis: runner environments show up as HIGH grad rate + FAST graduation speed. "
    "Flow + Crowding are intraday-aligned (00:00 UTC → now) for apples-to-apples comparison."
)

api_key = os.getenv("DUNE_API_KEY", "")
query_id = os.getenv("DUNE_QUERY_ID", "")
default_lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

# ----------------------------
# Basic validation
# ----------------------------
if not api_key or not query_id:
    st.error("Missing DUNE_API_KEY or DUNE_QUERY_ID in .env")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Settings")
    lookback = st.slider("Rolling window (days)", 7, 60, default_lookback, 1)
    st.text_input("Dune Query ID", value=query_id, disabled=True)

    st.divider()
    st.subheader("Refresh")
    fast_refresh = st.button("⚡ Fast Refresh (cached)", use_container_width=True)
    run_fresh = st.button("🔥 Run Fresh Query", use_container_width=True)
    retry_fetch = st.button("🔁 Fetch Last Fresh Result", use_container_width=True)

    st.caption(
        "Fast refresh pulls Dune’s latest stored results. "
        "Run Fresh Query triggers a fresh execution. "
        "If the fresh run isn’t ready yet, the app shows cached results and you can fetch later."
    )

# ----------------------------
# Caching (cached endpoint)
# ----------------------------
@st.cache_data(ttl=300)
def load_data_cached(qid: str, key: str) -> pd.DataFrame:
    return fetch_dune_results(qid, key)

# ----------------------------
# Schema helpers
# ----------------------------
def normalize_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day"] = pd.to_datetime(out["day"], errors="coerce", utc=True)
    out = out.dropna(subset=["day"]).sort_values("day")
    # keep as UTC midnight timestamps (works fine for charts); also store a date for display if you want
    out["day_date"] = out["day"].dt.date
    return out

def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected schema (time-fixed):
      - day
      - volume_sol (or volume)                    [full-day]
      - tokens_created                            [full-day]
      - volume_to_now_sol (or volume_to_now)      [intraday aligned 00:00->now UTC for each day]
      - tokens_created_to_now                     [intraday aligned]
      - graduated_tokens, grad_rate
      - median_minutes_to_grad
    Optional debug:
      - cutoff_seconds, now_utc, as_of_ts, day_rows, max_day, grads_with_launch_time
    """
    out = df.copy()

    # normalize naming
    if "volume" not in out.columns and "volume_sol" in out.columns:
        out = out.rename(columns={"volume_sol": "volume"})
    if "volume_to_now" not in out.columns and "volume_to_now_sol" in out.columns:
        out = out.rename(columns={"volume_to_now_sol": "volume_to_now"})

    required = {
        "day",
        "volume",
        "tokens_created",
        "volume_to_now",
        "tokens_created_to_now",
        "graduated_tokens",
        "grad_rate",
        "median_minutes_to_grad",
    }
    missing = required - set(out.columns)
    if missing:
        st.error(
            "Your Dune query must return:\n"
            "- `day`\n"
            "- `volume_sol` (or `volume`)\n"
            "- `tokens_created`\n"
            "- `volume_to_now_sol` (or `volume_to_now`)\n"
            "- `tokens_created_to_now`\n"
            "- `graduated_tokens`, `grad_rate`, `median_minutes_to_grad`\n\n"
            f"Missing: {missing}\n\n"
            f"Columns found: {list(out.columns)}"
        )
        st.stop()

    return out

# ----------------------------
# Regime math (4-factor)
# ----------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def norm_log_ratio(r: float, k: float) -> float:
    """
    Ratio -> ~[0,1] using log symmetry around 1.0
    1.0 -> 0.5, 2.0 and 0.5 are symmetric.
    """
    if r is None or not np.isfinite(r) or r <= 0:
        return 0.0
    return clamp(0.5 + k * float(np.log(r)))

def score_from_metrics(vol_ratio: float, grad_ratio: float, grad_speed_ratio: float, tokens_ratio: float) -> int:
    """
    Graduation likelihood (grad_rate vs median) 35%
    Graduation speed (faster vs median)         30%  (lower minutes_to_grad is better -> invert)
    Flow (volume vs median)                     20%
    Crowding inverse (tokens vs median)         15%  (more tokens => worse)
    """
    flow = norm_log_ratio(vol_ratio, k=0.35)
    grad = norm_log_ratio(grad_ratio, k=0.60)
    speed = norm_log_ratio(grad_speed_ratio, k=0.50)

    lt = np.log(tokens_ratio) if tokens_ratio and np.isfinite(tokens_ratio) and tokens_ratio > 0 else 0.0
    crowd = clamp(0.5 - 0.30 * lt)

    raw = (
        0.20 * flow +
        0.35 * grad +
        0.30 * speed +
        0.15 * crowd
    )
    return int(round(100 * clamp(raw)))

def regime_label(vol_ratio: float, tokens_ratio: float, grad_ratio: float, grad_speed_ratio: float):
    """
    Labels tuned for "runner likelihood".
    grad_speed_ratio > 1 means faster-than-median graduation.
    """
    if grad_ratio >= 1.10 and grad_speed_ratio >= 1.10 and tokens_ratio <= 1.25:
        return "GREEN", "High graduations + fast bonding curve (runner environment)"

    if grad_ratio < 0.85:
        return "RED", "Low graduations (bad environment)"

    if grad_speed_ratio < 0.85:
        return "RED", "Slow graduations (grind/distribution; fewer runners)"

    if tokens_ratio > 1.20 and grad_ratio < 1.00:
        return "RED", "Crowded + not enough winners"

    return "YELLOW", "Mixed / transition"

def regime_badge(regime: str) -> str:
    return {"GREEN": "🟩 GREEN", "YELLOW": "🟨 YELLOW", "RED": "🟥 RED"}.get(regime, f"⬜ {regime}")

def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df.copy().sort_values("day")

    # rolling medians
    # IMPORTANT: Flow + Crowding use intraday-aligned series
    d["vol_med"] = d["volume_to_now"].rolling(window).median()
    d["tok_med"] = d["tokens_created_to_now"].rolling(window).median()

    # Graduation metrics use full-day daily values
    d["grad_med"] = d["grad_rate"].rolling(window).median()
    d["grad_minutes_med"] = d["median_minutes_to_grad"].rolling(window).median()

    # ratios vs rolling medians
    d["vol_ratio"] = d["volume_to_now"] / d["vol_med"]
    d["tokens_ratio"] = d["tokens_created_to_now"] / d["tok_med"]
    d["grad_ratio"] = d["grad_rate"] / d["grad_med"]

    # faster = lower minutes => bullish => invert: ratio > 1 => faster-than-median graduation
    d["grad_speed_ratio"] = d["grad_minutes_med"] / d["median_minutes_to_grad"]

    d["regime_score"] = d.apply(
        lambda r: score_from_metrics(
            r["vol_ratio"], r["grad_ratio"], r["grad_speed_ratio"], r["tokens_ratio"]
        )
        if pd.notna(r["vol_ratio"]) and pd.notna(r["grad_ratio"]) and pd.notna(r["grad_speed_ratio"]) and pd.notna(r["tokens_ratio"])
        else np.nan,
        axis=1,
    )

    return d

# ----------------------------
# Session state
# ----------------------------
if "last_execution_id" not in st.session_state:
    st.session_state["last_execution_id"] = None
if "last_fresh_state" not in st.session_state:
    st.session_state["last_fresh_state"] = None
if "data_origin" not in st.session_state:
    st.session_state["data_origin"] = "cached"

# ----------------------------
# Data load / refresh logic (resilient)
# ----------------------------
df = None

if fast_refresh:
    st.cache_data.clear()
    st.session_state["data_origin"] = "cached"
    st.session_state["last_fresh_state"] = None

if retry_fetch:
    exid = st.session_state.get("last_execution_id")
    if not exid:
        st.info("No previous fresh execution found. Click “Run Fresh Query” first.")
    else:
        try:
            with st.spinner("Checking last execution status..."):
                state = wait_for_execution(exid, api_key, max_wait=60, raise_on_timeout=False)
                st.session_state["last_fresh_state"] = state

            if state == "QUERY_STATE_COMPLETED":
                with st.spinner("Fetching execution results..."):
                    df = fetch_execution_results(exid, api_key)
                st.session_state["data_origin"] = "fresh"
            else:
                st.warning(f"Fresh execution still running (state={state}). Showing cached results.")
                df = load_data_cached(query_id, api_key)
                st.session_state["data_origin"] = "cached"

        except Exception as e:
            st.error(f"Could not fetch last execution results. Showing cached results. Error: {e}")
            df = load_data_cached(query_id, api_key)
            st.session_state["data_origin"] = "cached"

if run_fresh and df is None:
    try:
        with st.spinner("Triggering fresh Dune execution..."):
            execution_id, df_fresh, state = try_run_and_fetch(
                query_id=query_id,
                api_key=api_key,
                max_wait=240,
                poll_seconds=2,
            )
        st.session_state["last_execution_id"] = execution_id
        st.session_state["last_fresh_state"] = state

        if df_fresh is not None:
            df = df_fresh
            st.session_state["data_origin"] = "fresh"
        else:
            st.warning(f"Fresh execution still running (state={state}). Showing cached results for now.")
            df = load_data_cached(query_id, api_key)
            st.session_state["data_origin"] = "cached"

    except Exception as e:
        st.error(f"Fresh query failed. Showing cached results. Error: {e}")
        df = load_data_cached(query_id, api_key)
        st.session_state["data_origin"] = "cached"

if df is None:
    df = load_data_cached(query_id, api_key)
    st.session_state["data_origin"] = "cached"

# ----------------------------
# Clean + normalize schema
# ----------------------------
df = pick_columns(df)

num_cols = [
    "volume",
    "tokens_created",
    "volume_to_now",
    "tokens_created_to_now",
    "graduated_tokens",
    "grad_rate",
    "median_minutes_to_grad",
    # optional:
    "cutoff_seconds",
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["day", "volume", "tokens_created", "volume_to_now", "tokens_created_to_now", "grad_rate", "median_minutes_to_grad"])
df = normalize_day(df)

# ----------------------------
# Feature engineering
# ----------------------------
window = max(7, int(lookback))
df_feat = compute_features(df, window)

# use the last `lookback` days where score is available
df_valid = df_feat[df_feat["regime_score"].notna()].sort_values("day").tail(window)

if df_valid.empty:
    st.warning("Not enough data to compute rolling medians yet. Increase history in Dune or reduce lookback.")
    st.stop()

latest = df_valid.iloc[-1]
vol_ratio = float(latest["vol_ratio"])
tok_ratio = float(latest["tokens_ratio"])
grad_rate = float(latest["grad_rate"])
grad_ratio = float(latest["grad_ratio"])
grad_speed_ratio = float(latest["grad_speed_ratio"])
minutes_to_grad = float(latest["median_minutes_to_grad"])
score = int(latest["regime_score"])

regime, rationale = regime_label(vol_ratio, tok_ratio, grad_ratio, grad_speed_ratio)

# ----------------------------
# TOP KPIs
# ----------------------------
c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.3, 1.1, 1.4, 1.4, 1.6])

with c1:
    st.caption("Regime")
    st.metric(label="", value=regime_badge(regime))
    st.caption(rationale)

with c2:
    st.caption("Regime Score")
    st.metric(label="", value=f"{score}/100")
    st.progress(score / 100)

with c3:
    st.caption("Flow ratio (intraday vs median)")
    st.metric(label="", value=f"{vol_ratio:.2f}x")
    st.caption("Flow uses 00:00 UTC → now vs same window historically")

with c4:
    st.caption("Graduation ratio (full-day vs median)")
    st.metric(label="", value=f"{grad_ratio:.2f}x")
    st.caption(f"Today grad rate: {grad_rate*100:.2f}%")

with c5:
    st.caption("Graduation speed (full-day vs median)")
    st.metric(label="", value=f"{grad_speed_ratio:.2f}x")
    st.caption(f"Median minutes to grad: {minutes_to_grad:.2f} min")

with c6:
    st.caption("Crowding ratio (intraday vs median)")
    st.metric(label="", value=f"{tok_ratio:.2f}x")
    st.caption(f"Rolling median window: {window}d")

# ----------------------------
# Freshness / execution info
# ----------------------------
as_of_ts = None
max_day = None
row_count = None

if "as_of_ts" in df.columns:
    ts = pd.to_datetime(df["as_of_ts"], errors="coerce", utc=True)
    if ts.notna().any():
        as_of_ts = ts.max()

if "max_day" in df.columns:
    md = pd.to_datetime(df["max_day"], errors="coerce", utc=True)
    if md.notna().any():
        max_day = md.max()

if "day_rows" in df.columns:
    rc = pd.to_numeric(df["day_rows"], errors="coerce")
    if rc.notna().any():
        row_count = int(rc.max())

origin = st.session_state.get("data_origin", "cached")
exec_id = st.session_state.get("last_execution_id", None)
state = st.session_state.get("last_fresh_state", None)

fresh_bits = []
if as_of_ts is not None:
    fresh_bits.append(f"🧠 Data as of: **{as_of_ts}**")
if max_day is not None:
    fresh_bits.append(f"Max day: **{max_day.date()}**")
if row_count is not None:
    fresh_bits.append(f"Rows: **{row_count}**")

# show intraday cutoff (nice sanity check)
if "cutoff_seconds" in df.columns and pd.notna(latest.get("cutoff_seconds", np.nan)):
    cutoff_seconds = float(latest["cutoff_seconds"])
    fresh_bits.append(f"UTC cutoff: **{cutoff_seconds/3600:.2f}h** after midnight")

st.caption(" | ".join(fresh_bits) if fresh_bits else "🧠 Freshness info unavailable.")
if origin == "fresh" and exec_id:
    st.caption(f"Origin: **fresh** (execution_id: `{exec_id}`)")
elif exec_id and state and state != "QUERY_STATE_COMPLETED":
    st.caption(f"Origin: **cached** (fresh execution still running: state=`{state}`, execution_id=`{exec_id}`)")
else:
    st.caption("Origin: **cached**")

st.divider()

# ----------------------------
# Charts (ratios + score)
# ----------------------------
st.subheader("Regime Signals")

st.plotly_chart(
    px.line(df_valid, x="day", y="regime_score", title="Regime Score (0–100)"),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.line(df_valid, x="day", y="vol_ratio", title="Flow Ratio (intraday volume vs rolling median)"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        px.line(df_valid, x="day", y="grad_ratio", title="Graduation Ratio (full-day grad_rate vs rolling median)"),
        use_container_width=True,
    )

midl, midr = st.columns(2)
with midl:
    st.plotly_chart(
        px.line(df_valid, x="day", y="tokens_ratio", title="Crowding Ratio (intraday tokens created vs rolling median)"),
        use_container_width=True,
    )
with midr:
    st.plotly_chart(
        px.line(df_valid, x="day", y="grad_speed_ratio", title="Graduation Speed Ratio (full-day; faster is higher)"),
        use_container_width=True,
    )

# ----------------------------
# Optional: raw charts (debug)
# ----------------------------
with st.expander("Optional: Raw series (for debugging only)", expanded=False):
    st.caption("Flow/Crowding show BOTH full-day and intraday-aligned (to-now) values.")
    l2, r2 = st.columns(2)
    with l2:
        st.plotly_chart(px.bar(df_feat, x="day", y="volume", title="Full-day Volume (SOL units)"), use_container_width=True)
        st.plotly_chart(px.bar(df_feat, x="day", y="volume_to_now", title="Volume to-now (00:00 UTC → now)"), use_container_width=True)
    with r2:
        st.plotly_chart(px.bar(df_feat, x="day", y="tokens_created", title="Full-day Tokens Created"), use_container_width=True)
        st.plotly_chart(px.bar(df_feat, x="day", y="tokens_created_to_now", title="Tokens Created to-now (00:00 UTC → now)"), use_container_width=True)

    st.plotly_chart(px.bar(df_feat, x="day", y="graduated_tokens", title="Graduated Tokens (raw count)"), use_container_width=True)
    st.plotly_chart(px.line(df_feat, x="day", y="grad_rate", title="Graduation Rate (raw; full-day)"), use_container_width=True)
    st.plotly_chart(px.line(df_feat, x="day", y="median_minutes_to_grad", title="Median Minutes to Graduate (raw; full-day)"), use_container_width=True)

with st.expander("Recent rows (ratios only)", expanded=False):
    cols = [
        "day",
        "volume_to_now",
        "tokens_created_to_now",
        "grad_rate",
        "median_minutes_to_grad",
        "vol_ratio",
        "tokens_ratio",
        "grad_ratio",
        "grad_speed_ratio",
        "regime_score",
    ]
    cols = [c for c in cols if c in df_feat.columns]
    st.dataframe(df_feat[cols].sort_values("day").tail(60), use_container_width=True)
