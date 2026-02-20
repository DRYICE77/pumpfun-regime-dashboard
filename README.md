# Pump.fun Regime Dashboard (Local)

## Setup
1. Create a virtualenv
2. Install deps
3. Add your Dune API key + query id to `.env`
4. Run Streamlit

## Run
```bash
cd pumpfun_regime_dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
