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
    "Signal-first view: ratios vs rolling median + regime score (raw volumes are treated as units; focus on ratios)."
)

api_key = os.getenv("DUNE_API_KEY", "")
query_id = os.getenv("DUNE_QUERY_ID", "")
default_lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

# ----------------------------
# Session state (init early)
# ----------------------------
if "last_execution_id" not in st.session_state:
    st.session_state["last_execution_id"] = None
if "last_refresh_mode" not in st.session_state:
    st.session_state["last_refresh_mode"] = "cached"
if "last_fresh_state" not in st.session_state:
    st.session_state["last_fresh_state"] = None
if "data_origin" not in st.session_state:
    st.session_state["data_origin"] = "cached"
if "refresh_action" not in st.session_state:
    st.session_state["refresh_action"] = None  # "fast" | "fresh" | "retry" | None

def trigger(action: str):
    st.session_state["refresh_action"] = action

# ----------------------------
# Quick Actions (VISIBLE ON MOBILE)
# ----------------------------
st.markdown("### Quick Actions")
qa1, qa2, qa3 = st.columns(3)

with qa1:
    if st.button("⚡ Fast", use_container_width=True, key="top_fast"):
        trigger("fast")

with qa2:
    if st.button("🔥 Fresh", use_container_width=True, key="top_fresh"):
        trigger("fresh")

with qa3:
    if st.button("🔁 Fetch", use_container_width=True, key="top_retry"):
        trigger("retry")

st.divider()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Settings")

    lookback = st.slider("Rolling window (days)", 7, 60, default_lookback, 1)
    st.text_input("Dune Query ID", value=query_id, disabled=True)

    st.divider()
    st.subheader("Refresh")

    if st.button("⚡ Fast Refresh (cached)", use_container_width=True, key="sb_fast"):
        trigger("fast")
    if st.button("🔥 Run Fresh Query", use_container_width=True, key="sb_fresh"):
        trigger("fresh")
    if st.button("🔁 Fetch Last Fresh Result", use_container_width=True, key="sb_retry"):
        trigger("retry")

    st.caption(
        "Fast refresh pulls Dune’s latest stored results. "
        "Run Fresh Query triggers a fresh execution. "
        "If the fresh run isn’t ready yet, the app shows cached results and you can fetch later."
    )

# ----------------------------
# Basic validation
# ----------------------------
if not api_key or not query_id:
    st.error("Missing DUNE_API_KEY or DUNE_QUERY_ID in .env")
    st.stop()

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
    out["day"] = pd.to_datetime(out["day"], errors="coerce")
    out = out.dropna(subset=["day"]).sort_values("day")
    return out

def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports schemas:
      - day, volume_sol, tokens_created, volume_per_token (+ optional: graduated_tokens, grad_rate, as_of_ts...)
      - day, volume,     tokens_created, volume_per_token (+ optional: graduated_tokens, grad_rate, as_of_ts...)
    Normalizes to: day, volume, tokens_created, volume_per_token
    Leaves extra columns intact.
    """
    out = df.copy()

    if "volume" not in out.columns and "volume_sol" in out.columns:
        out = out.rename(columns={"volume_sol": "volume"})

    required = {"day", "volume", "tokens_created", "volume_per_token"}
    missing = required - set(out.columns)
    if missing:
        st.error(
            "Your Dune query must return either:\n"
            "- `day, volume_sol, tokens_created, volume_per_token`\n"
            "- `day, volume, tokens_created, volume_per_token`\n\n"
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

def score_from_4_ratios(vol_ratio: float, vpt_ratio: float, tokens_ratio: float, grad_ratio: float) -> int:
    """
    4-factor regime score:

      Quality (VPT vs median)          30%
      Flow    (Volume vs median)       25%
      Crowding inverse (Tokens vs med) 15%  (more tokens => worse)
      Graduation (Grad rate vs median) 30%

    Graduation is the "market produces winners" signal.
    """
    quality = norm_log_ratio(vpt_ratio, k=0.45)
    flow = norm_log_ratio(vol_ratio, k=0.35)

    lt = np.log(tokens_ratio) if tokens_ratio and np.isfinite(tokens_ratio) and tokens_ratio > 0 else 0.0
    crowd = clamp(0.5 - 0.30 * lt)

    grad = norm_log_ratio(grad_ratio, k=0.60)

    raw = (
        0.30 * quality +
        0.25 * flow +
        0.15 * crowd +
        0.30 * grad
    )
    return int(round(100 * clamp(raw)))

def regime_label(vol_ratio: float, vpt_ratio: float, tokens_ratio: float, grad_ratio: float):
    if grad_ratio >= 1.10 and vpt_ratio >= 1.05 and vol_ratio >= 1.00 and tokens_ratio <= 1.20:
        return "GREEN", "Strong winners + quality flow"

    if grad_ratio < 0.85:
        return "RED", "Low graduations (bad environment)"

    if vpt_ratio < 0.90 and tokens_ratio > 1.05:
        return "RED", "Crowded + low quality"

    return "YELLOW", "Mixed / transition"

def regime_badge(regime: str) -> str:
    return {"GREEN": "🟩 GREEN", "YELLOW": "🟨 YELLOW", "RED": "🟥 RED"}.get(regime, f"⬜ {regime}")

def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df.copy().sort_values("day")

    d["vol_med"] = d["volume"].rolling(window).median()
    d["tok_med"] = d["tokens_created"].rolling(window).median()
    d["vpt_med"] = d["volume_per_token"].rolling(window).median()

    d["vol_ratio"] = d["volume"] / d["vol_med"]
    d["tokens_ratio"] = d["tokens_created"] / d["tok_med"]
    d["vpt_ratio"] = d["volume_per_token"] / d["vpt_med"]

    if "grad_rate" in d.columns:
        d["grad_med"] = d["grad_rate"].rolling(window).median()
        d["grad_ratio"] = d["grad_rate"] / d["grad_med"]
    else:
        d["grad_rate"] = np.nan
        d["grad_med"] = np.nan
        d["grad_ratio"] = np.nan

    d["regime_score"] = d.apply(
        lambda r: score_from_4_ratios(
            r["vol_ratio"], r["vpt_ratio"], r["tokens_ratio"], r["grad_ratio"]
        )
        if pd.notna(r["vol_ratio"]) and pd.notna(r["vpt_ratio"]) and pd.notna(r["tokens_ratio"]) and pd.notna(r["grad_ratio"])
        else np.nan,
        axis=1,
    )

    return d

# ----------------------------
# Data load / refresh logic (single-source action)
# ----------------------------
df = None
action = st.session_state.get("refresh_action", None)

# Fast refresh: clear cache then load cached results
if action == "fast":
    st.cache_data.clear()
    st.session_state["last_refresh_mode"] = "cached"
    st.session_state["data_origin"] = "cached"
    st.session_state["last_fresh_state"] = None

# Retry fetch: attempt to fetch results for last execution without re-running
if action == "retry":
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
                st.session_state["last_refresh_mode"] = "fresh"
                st.session_state["data_origin"] = "fresh"
            else:
                st.warning(f"Fresh execution still running (state={state}). Showing cached results.")
                df = load_data_cached(query_id, api_key)
                st.session_state["last_refresh_mode"] = "cached"
                st.session_state["data_origin"] = "cached"

        except Exception as e:
            st.error(f"Could not fetch last execution results. Showing cached results. Error: {e}")
            df = load_data_cached(query_id, api_key)
            st.session_state["last_refresh_mode"] = "cached"
            st.session_state["data_origin"] = "cached"

# Run fresh: trigger new execution, wait best-effort, fetch if ready; otherwise fall back
if action == "fresh" and df is None:
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
            st.session_state["last_refresh_mode"] = "fresh"
            st.session_state["data_origin"] = "fresh"
        else:
            st.warning(f"Fresh execution still running (state={state}). Showing cached results for now.")
            df = load_data_cached(query_id, api_key)
            st.session_state["last_refresh_mode"] = "cached"
            st.session_state["data_origin"] = "cached"

    except Exception as e:
        st.error(f"Fresh query failed. Showing cached results. Error: {e}")
        df = load_data_cached(query_id, api_key)
        st.session_state["last_refresh_mode"] = "cached"
        st.session_state["data_origin"] = "cached"

# Default: cached results
if df is None:
    df = load_data_cached(query_id, api_key)
    st.session_state["last_refresh_mode"] = "cached"
    st.session_state["data_origin"] = "cached"

# IMPORTANT: clear action so it doesn't retrigger on the next rerun
st.session_state["refresh_action"] = None

# ----------------------------
# Clean + normalize schema
# ----------------------------
df = pick_columns(df)

for col in ["volume", "tokens_created", "volume_per_token"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "grad_rate" in df.columns:
    df["grad_rate"] = pd.to_numeric(df["grad_rate"], errors="coerce")
if "graduated_tokens" in df.columns:
    df["graduated_tokens"] = pd.to_numeric(df["graduated_tokens"], errors="coerce")

df = df.dropna(subset=["day", "volume", "tokens_created"])
df = normalize_day(df)

# ----------------------------
# Feature engineering
# ----------------------------
window = max(7, int(lookback))
df_feat = compute_features(df, window)

df_valid = df_feat[df_feat["regime_score"].notna()].sort_values("day").tail(int(lookback))

if df_valid.empty:
    if "grad_rate" not in df.columns:
        st.error("Your Dune query is missing `grad_rate`. Add it to the SQL output to use the 4-factor model.")
    else:
        st.warning("Not enough data to compute rolling medians yet. Increase history in Dune or increase lookback.")
    st.stop()

latest = df_valid.iloc[-1]
vol_ratio = float(latest["vol_ratio"])
tok_ratio = float(latest["tokens_ratio"])
vpt_ratio = float(latest["vpt_ratio"])
grad_rate = float(latest["grad_rate"]) if pd.notna(latest["grad_rate"]) else np.nan
grad_ratio = float(latest["grad_ratio"])
score = int(latest["regime_score"])
regime, rationale = regime_label(vol_ratio, vpt_ratio, tok_ratio, grad_ratio)

# ----------------------------
# TOP KPIs
# ----------------------------
c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.3, 1.1, 1.4, 1.4, 1.4])

with c1:
    st.caption("Regime")
    st.metric(label="", value=regime_badge(regime))
    st.caption(rationale)

with c2:
    st.caption("Regime Score")
    st.metric(label="", value=f"{score}/100")
    st.progress(score / 100)

with c3:
    st.caption("Flow ratio (vs median)")
    st.metric(label="", value=f"{vol_ratio:.2f}x")

with c4:
    st.caption("Quality ratio (Vol/Token vs median)")
    st.metric(label="", value=f"{vpt_ratio:.2f}x")

with c5:
    st.caption("Crowding ratio (Tokens vs median)")
    st.metric(label="", value=f"{tok_ratio:.2f}x")
    st.caption(f"Rolling median window: {window}d")

with c6:
    st.caption("Graduation ratio (Grad rate vs median)")
    st.metric(label="", value=f"{grad_ratio:.2f}x")
    if np.isfinite(grad_rate):
        st.caption(f"Today grad rate: {grad_rate*100:.2f}%")
    else:
        st.caption("Today grad rate: n/a")


# Mobile-friendly lookback control (main page)
lookback = st.slider(
    "Rolling window (days)",
    min_value=7,
    max_value=60,
    value=default_lookback,
    step=1,
    help="Controls the rolling median window used to compute the ratios + regime score.",
)


# ----------------------------
# Freshness / execution info
# ----------------------------
as_of_ts = None
max_day = None
row_count = None

if "as_of_ts" in df.columns:
    ts = pd.to_datetime(df["as_of_ts"], errors="coerce")
    if ts.notna().any():
        as_of_ts = ts.max()

if "max_day" in df.columns:
    md = pd.to_datetime(df["max_day"], errors="coerce")
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

st.caption(" | ".join(fresh_bits) if fresh_bits else "🧠 Freshness info unavailable (query not returning as_of_ts/day_rows/max_day).")

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
        px.line(df_valid, x="day", y="vol_ratio", title="Flow Ratio (volume vs rolling median)"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        px.line(df_valid, x="day", y="vpt_ratio", title="Quality Ratio (vol/token vs rolling median)"),
        use_container_width=True,
    )

midl, midr = st.columns(2)
with midl:
    st.plotly_chart(
        px.line(df_valid, x="day", y="tokens_ratio", title="Crowding Ratio (tokens created vs rolling median)"),
        use_container_width=True,
    )
with midr:
    st.plotly_chart(
        px.line(df_valid, x="day", y="grad_ratio", title="Graduation Ratio (grad_rate vs rolling median)"),
        use_container_width=True,
    )

# ----------------------------
# Optional: raw charts (debug)
# ----------------------------
with st.expander("Optional: Raw series (for debugging only)", expanded=False):
    st.caption("Raw units from events tables; use ratios above for decisions.")
    l2, r2 = st.columns(2)
    with l2:
        st.plotly_chart(px.bar(df_feat, x="day", y="volume", title="Raw Flow (units)"), use_container_width=True)
    with r2:
        st.plotly_chart(px.bar(df_feat, x="day", y="tokens_created", title="Tokens Created (raw count)"), use_container_width=True)

    if "graduated_tokens" in df_feat.columns:
        st.plotly_chart(px.bar(df_feat, x="day", y="graduated_tokens", title="Graduated Tokens (raw count)"), use_container_width=True)

    if "grad_rate" in df_feat.columns:
        st.plotly_chart(px.line(df_feat, x="day", y="grad_rate", title="Graduation Rate (raw)"), use_container_width=True)

with st.expander("Recent rows (ratios only)", expanded=False):
    cols = [
        "day",
        "tokens_created",
        "graduated_tokens",
        "grad_rate",
        "vol_ratio",
        "tokens_ratio",
        "vpt_ratio",
        "grad_ratio",
        "regime_score",
    ]
    cols = [c for c in cols if c in df_feat.columns]
    st.dataframe(df_feat[cols].tail(60), use_container_width=True)
