import pandas as pd


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_features(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Adds columns the dashboard expects:
      - vol_7d_ma, vol_ratio
      - token_7d_ma, token_ratio
      - vpt_7d_ma, vpt_ratio
      - (optional) vpt_z, vol_growth
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    # day
    if "day" in out.columns:
        out["day"] = _to_datetime_safe(out["day"])
        out = out.sort_values("day")

    # numeric columns
    for col in ["volume_sol", "tokens_created", "volume_per_token"]:
        if col in out.columns:
            out[col] = _to_numeric_safe(out[col])

    lb = int(lookback_days) if lookback_days and lookback_days > 1 else 30

    # -------------------------
    # Volume features
    # -------------------------
    if "volume_sol" in out.columns:
        out["vol_7d_ma"] = out["volume_sol"].rolling(7, min_periods=1).mean()
        denom = out["vol_7d_ma"].replace(0, pd.NA)
        out["vol_ratio"] = out["volume_sol"] / denom
        out["vol_growth"] = out["volume_sol"].pct_change().fillna(0.0)

    # -------------------------
    # Tokens created features
    # -------------------------
    if "tokens_created" in out.columns:
        out["token_7d_ma"] = out["tokens_created"].rolling(7, min_periods=1).mean()
        denom = out["token_7d_ma"].replace(0, pd.NA)
        out["token_ratio"] = out["tokens_created"] / denom

    # -------------------------
    # Volume per token features
    # -------------------------
    if "volume_per_token" in out.columns:
        out["vpt_7d_ma"] = out["volume_per_token"].rolling(7, min_periods=1).mean()
        denom = out["vpt_7d_ma"].replace(0, pd.NA)
        out["vpt_ratio"] = out["volume_per_token"] / denom

        # optional z-score (nice for regime logic)
        vpt_mu = out["volume_per_token"].rolling(lb, min_periods=5).mean()
        vpt_sd = out["volume_per_token"].rolling(lb, min_periods=5).std()
        out["vpt_z"] = (out["volume_per_token"] - vpt_mu) / vpt_sd.replace(0, pd.NA)

    return out


def classify_row(row: pd.Series):
    """
    Returns (regime, rationale) where regime matches what app.py expects
    (RED / YELLOW / GREEN).
    """
    vpt = row.get("volume_per_token")
    vpt_ratio = row.get("vpt_ratio")
    vol_ratio = row.get("vol_ratio")

    # defaults
    if pd.isna(vpt):
        return "YELLOW", "Missing volume_per_token; cannot classify."
    if pd.isna(vpt_ratio):
        vpt_ratio = 1.0
    if pd.isna(vol_ratio):
        vol_ratio = 1.0

    # simple, tune later
    if vpt_ratio >= 1.10 and vol_ratio >= 1.05:
        return "GREEN", f"Strong: vpt_ratio={vpt_ratio:.2f}, vol_ratio={vol_ratio:.2f}"
    if vpt_ratio <= 0.90 and vol_ratio <= 0.95:
        return "RED", f"Weak: vpt_ratio={vpt_ratio:.2f}, vol_ratio={vol_ratio:.2f}"
    return "YELLOW", f"Mixed: vpt_ratio={vpt_ratio:.2f}, vol_ratio={vol_ratio:.2f}"


