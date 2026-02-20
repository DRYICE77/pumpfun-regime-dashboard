import time
import requests
import pandas as pd
from typing import Optional, Tuple

DUNE_BASE = "https://api.dune.com/api/v1"

COMPLETED = "QUERY_STATE_COMPLETED"
FAILED = "QUERY_STATE_FAILED"
CANCELLED = "QUERY_STATE_CANCELLED"


def _headers(api_key: str) -> dict:
    if not api_key:
        raise ValueError("Missing DUNE_API_KEY")
    return {"X-DUNE-API-KEY": api_key}


def fetch_dune_results(query_id: str, api_key: str) -> pd.DataFrame:
    """
    FAST path: returns the latest cached results for a query.
    This does NOT re-run the query — it only fetches the most recent stored result.
    """
    if not query_id:
        raise ValueError("Missing DUNE_QUERY_ID")

    url = f"{DUNE_BASE}/query/{query_id}/results"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()

    data = r.json()
    rows = data.get("result", {}).get("rows", []) or []
    if not rows:
        raise RuntimeError(
            "No rows returned from Dune cached results. "
            "Run/schedule the query at least once."
        )
    return pd.DataFrame(rows)


def run_dune_query(query_id: str, api_key: str) -> str:
    """
    SLOW path step 1: triggers a fresh execution of the query.
    Returns an execution_id.
    """
    if not query_id:
        raise ValueError("Missing DUNE_QUERY_ID")

    url = f"{DUNE_BASE}/query/{query_id}/execute"
    r = requests.post(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()

    data = r.json()
    execution_id = data.get("execution_id") or data.get("executionId")
    if not execution_id:
        raise RuntimeError(f"Unexpected execute response (no execution_id): {data}")
    return str(execution_id)


def get_execution_status(execution_id: str, api_key: str) -> str:
    """
    Returns the execution state string.
    """
    if not execution_id:
        raise ValueError("Missing execution_id")

    url = f"{DUNE_BASE}/execution/{execution_id}/status"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()

    data = r.json()
    state = data.get("state") or data.get("status")
    if not state:
        raise RuntimeError(f"Unexpected status response (no state): {data}")
    return str(state)


def wait_for_execution(
    execution_id: str,
    api_key: str,
    max_wait: int = 90,
    poll_seconds: int = 2,
    raise_on_timeout: bool = True,
) -> str:
    """
    Polls until the execution completes (or fails / cancels / times out).

    Returns the final/last-known state.
    If raise_on_timeout=True, raises TimeoutError on timeout.
    If raise_on_timeout=False, returns the last-known state on timeout.
    """
    start = time.time()
    last_state = None

    while True:
        last_state = get_execution_status(execution_id, api_key)

        if last_state == COMPLETED:
            return last_state

        if last_state in (FAILED, CANCELLED):
            raise RuntimeError(
                f"Dune execution ended in state={last_state} (execution_id={execution_id})"
            )

        if time.time() - start > max_wait:
            if raise_on_timeout:
                raise TimeoutError(
                    f"Dune execution timed out after {max_wait}s "
                    f"(last_state={last_state}, execution_id={execution_id})"
                )
            return last_state

        time.sleep(poll_seconds)


def fetch_execution_results(execution_id: str, api_key: str) -> pd.DataFrame:
    """
    Fetch results for a specific execution_id (fresh run).
    """
    if not execution_id:
        raise ValueError("Missing execution_id")

    url = f"{DUNE_BASE}/execution/{execution_id}/results"
    r = requests.get(url, headers=_headers(api_key), timeout=30)
    r.raise_for_status()

    data = r.json()
    rows = data.get("result", {}).get("rows", []) or []
    if not rows:
        raise RuntimeError(f"No rows returned for execution_id={execution_id}.")
    return pd.DataFrame(rows)


def try_run_and_fetch(
    query_id: str,
    api_key: str,
    max_wait: int = 120,
    poll_seconds: int = 2,
) -> Tuple[str, Optional[pd.DataFrame], str]:
    """
    Convenience helper for UIs:
    - triggers a fresh execution
    - waits up to max_wait
    - if completed, returns (execution_id, dataframe, state)
    - if not completed in time, returns (execution_id, None, last_state)

    Never raises TimeoutError; still raises for FAILED/CANCELLED and HTTP errors.
    """
    execution_id = run_dune_query(query_id, api_key)
    state = wait_for_execution(
        execution_id,
        api_key,
        max_wait=max_wait,
        poll_seconds=poll_seconds,
        raise_on_timeout=False,
    )

    if state == COMPLETED:
        df = fetch_execution_results(execution_id, api_key)
        return execution_id, df, state

    return execution_id, None, state


