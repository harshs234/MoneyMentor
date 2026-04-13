import streamlit as st
import asyncio
import json
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from datetime import date, datetime
from typing import Any, Dict, List, Union
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import platform
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()
from ai.financial_agent import FinancialAgent
from ai.investment_planner import InvestmentPlanner
from ai.investment_advisor import InvestmentAdvisor

planner = InvestmentPlanner()
advisor = InvestmentAdvisor()
agent = FinancialAgent()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)
if GEMINI_ENABLED:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.0-flash"
    except Exception:
        GEMINI_ENABLED = False

if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinBot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Green & White Finance Theme CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');

/* ══════════════════════════════════════════
   CSS CUSTOM PROPERTIES — light & dark mode
   ══════════════════════════════════════════ */
:root {
    /* Dark defaults */
    --bg:           #030a05;
    --bg2:          #040d06;
    --card:         #0a1a0d;
    --card2:        #0d2010;
    --sidebar:      #040d06;
    --input:        #0a1a0d;
    --tab-bg:       #060f08;
    --form-bg:      #060f08;
    --text:         #e8f5ea;
    --text2:        #a7f3c0;
    --muted:        #4b6e54;
    --green:        #4ade80;
    --green2:       #22c55e;
    --green3:       #86efac;
    --green-deep:   #14532d;
    --border:       rgba(34,197,94,0.12);
    --border2:      rgba(34,197,94,0.22);
    --shadow-card:  0 4px 24px rgba(0,0,0,0.35), 0 0 0 1px rgba(34,197,94,0.06);
    --shadow-btn:   0 4px 14px rgba(34,197,94,0.28), 0 2px 6px rgba(0,0,0,0.35);
    --shadow-hover: 0 6px 22px rgba(34,197,94,0.42), 0 2px 10px rgba(0,0,0,0.45);
    --shadow-primary: 0 4px 18px rgba(34,197,94,0.45), 0 2px 6px rgba(0,0,0,0.3);
}
@media (prefers-color-scheme: light) {
    :root {
        --bg:           #f0fdf4;
        --bg2:          #f7fef9;
        --card:         #ffffff;
        --card2:        #f0fdf4;
        --sidebar:      #f7fef9;
        --input:        #ffffff;
        --tab-bg:       #e8f5ea;
        --form-bg:      #f7fef9;
        --text:         #14532d;
        --text2:        #166534;
        --muted:        #4b8c5a;
        --green:        #16a34a;
        --green2:       #15803d;
        --green3:       #14532d;
        --green-deep:   #052e16;
        --border:       rgba(34,197,94,0.2);
        --border2:      rgba(34,197,94,0.35);
        --shadow-card:  0 4px 20px rgba(0,0,0,0.06), 0 0 0 1px rgba(34,197,94,0.14);
        --shadow-btn:   0 4px 12px rgba(34,197,94,0.22), 0 2px 4px rgba(0,0,0,0.06);
        --shadow-hover: 0 6px 18px rgba(34,197,94,0.32), 0 2px 8px rgba(0,0,0,0.08);
        --shadow-primary: 0 4px 16px rgba(34,197,94,0.35), 0 2px 5px rgba(0,0,0,0.06);
    }
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 1.2rem 1.8rem 2rem; max-width: 100%; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border);
    width: 360px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Logo ── */
.finbot-logo-wrap {
    padding: 1.2rem 1.2rem 0.6rem;
    border-bottom: 1px solid rgba(34,197,94,0.1);
    margin-bottom: 0.8rem;
}
.finbot-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 50%, #86efac 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1;
}
.finbot-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Connection buttons ── */
.conn-row { padding: 0 1rem 0.8rem; }
.conn-status {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.conn-status.on  { background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3); color:var(--green); }
.conn-status.off { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.25); color:#f87171; }
.dot { width:6px;height:6px;border-radius:50%;display:inline-block; }
.dot.g { background:var(--green); box-shadow:0 0 5px var(--green); }
.dot.r { background:#f87171; box-shadow:0 0 5px #f87171; }

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    padding: 0.35rem 0.8rem !important;
    transition: all 0.18s !important;
}
[data-testid="stSidebar"] .stButton > button:first-child {
    background: linear-gradient(135deg, var(--green-deep), #166534) !important;
    border: 1px solid var(--border2) !important;
    color: var(--green) !important;
    box-shadow: var(--shadow-btn) !important;
}
[data-testid="stSidebar"] .stButton > button:first-child:hover {
    background: linear-gradient(135deg, #166534, #15803d) !important;
    box-shadow: var(--shadow-hover) !important;
}

/* ── AI Chat in Sidebar ── */
.sidebar-chat-wrap {
    padding: 0 0.8rem;
    display: flex;
    flex-direction: column;
    height: calc(100vh - 220px);
}
.sidebar-chat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #4b6e54;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    padding: 0.5rem 0.2rem 0.4rem;
    border-bottom: 1px solid rgba(34,197,94,0.08);
    margin-bottom: 0.6rem;
}
.chat-msgs {
    flex: 1;
    overflow-y: auto;
    padding-right: 4px;
    margin-bottom: 0.6rem;
}
.msg-user {
    background: rgba(34,197,94,0.07);
    border: 1px solid var(--border2);
    border-radius: 10px 10px 2px 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.5;
}
.msg-bot {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px 10px 10px 2px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: var(--text2);
    line-height: 1.5;
}
.msg-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 3px;
}

/* ── Main area ── */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 1rem;
    border-bottom: 1px solid rgba(34,197,94,0.1);
    margin-bottom: 1.2rem;
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
}
.main-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Tool pill buttons ── */
.tool-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.2rem; }
.tool-pill {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid rgba(34,197,94,0.2);
    background: #0a1a0d;
    color: #86efac;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.tool-pill:hover, .tool-pill.active {
    background: rgba(34,197,94,0.15);
    border-color: rgba(74,222,128,0.5);
    color: #4ade80;
}

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px;
    padding: 0.9rem 1.1rem !important;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--green2) !important;
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem !important;
    color: var(--green) !important;
    font-weight: 700 !important;
}

/* ── Main buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 9px !important;
    border: 1px solid var(--border2) !important;
    background: var(--card) !important;
    color: var(--green) !important;
    transition: all 0.18s !important;
    padding: 0.42rem 1.1rem !important;
    box-shadow: var(--shadow-btn) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--green-deep), #166534) !important;
    border-color: var(--green2) !important;
    box-shadow: var(--shadow-hover) !important;
    color: var(--green3) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #15803d, #16a34a) !important;
    border-color: var(--green2) !important;
    color: #f0fdf4 !important;
    box-shadow: var(--shadow-primary) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-1px);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: var(--input) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.83rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--green2) !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.12) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--tab-bg);
    border-radius: 10px;
    padding: 3px;
    gap: 3px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    padding: 0.35rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--green-deep), #166534) !important;
    color: var(--green) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(34,197,94,0.08) !important; border-left: 3px solid #22c55e !important; border-radius: 0 9px 9px 0 !important; }
.stWarning { background: rgba(234,179,8,0.08) !important;  border-left: 3px solid #eab308 !important; border-radius: 0 9px 9px 0 !important; }
.stError   { background: rgba(239,68,68,0.08) !important;  border-left: 3px solid #ef4444 !important; border-radius: 0 9px 9px 0 !important; }
.stInfo    { background: rgba(34,197,94,0.05) !important;  border-left: 3px solid var(--green2) !important; border-radius: 0 9px 9px 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green2) !important; }

/* ── Form ── */
[data-testid="stForm"] {
    background: var(--form-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: var(--shadow-card);
}

/* ── Profile card ── */
.profile-card {
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow-card);
}
.profile-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
}
.profile-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--green);
}

/* ── Onboarding modal overlay ── */
.onboard-card {
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2.5rem;
    max-width: 560px;
    width: 90%;
    box-shadow: var(--shadow-hover);
}
.onboard-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4ade80, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.onboard-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--green2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(34,197,94,0.2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,197,94,0.4); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly theme (green)
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#86efac", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#4ade80","#22c55e","#86efac","#bbf7d0","#a3e635","#84cc16","#65a30d"],
    xaxis=dict(gridcolor="rgba(34,197,94,0.07)", zerolinecolor="rgba(34,197,94,0.12)"),
    yaxis=dict(gridcolor="rgba(34,197,94,0.07)", zerolinecolor="rgba(34,197,94,0.12)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(34,197,94,0.15)"),
)

def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
# Event loop
# ─────────────────────────────────────────────
@st.cache_resource
def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

loop = get_event_loop()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
_defaults = {
    "connected": False,
    "tools": [],
    "mcp_client": None,
    "mcp_session": None,
    "read_stream": None,
    "write_stream": None,
    "artifact_paths": [],
    "last_payload": None,
    "chat_history": [],
    "selected_tool": None,
    "filtered_tickers": None,
    "filtered_details": None,
    "user_profile": None,        # Stores earnings, expenses, savings_goal, portfolio
    "profile_complete": False,   # Whether onboarding is done
    "onboarding_step": "form",   # 'form' | 'json'
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ARTIFACT_DIR = os.path.join(".", "data", "session_exports")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _save_artifact(name: str, payload: Any):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ARTIFACT_DIR, f"{ts}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        st.session_state.artifact_paths.append(path)
        st.session_state.artifact_paths = st.session_state.artifact_paths[-10:]
        return path
    except Exception:
        return None

def human_currency(n: Union[int, float, None], currency: str = "USD") -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "—"
    sign = "-" if float(n) < 0 else ""
    n = abs(float(n))
    unit = ""
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000.0:
            break
        n /= 1000.0
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
    sym = symbols.get(currency.upper(), "")
    return f"{sign}{sym}{n:,.2f}{unit}"

def infer_currency_from_ticker(ticker: str | None) -> str:
    if not ticker:
        return "USD"
    tick = str(ticker).upper()
    if tick.endswith(".NS") or tick.endswith(".BO"):
        return "INR"
    return "USD"

def tidy_float(x: Any) -> Any:
    try:
        f = float(x)
        return f"{f:,.0f}" if abs(f) >= 100 else f"{f:,.2f}"
    except Exception:
        return x

@st.cache_data(ttl=3600)
def resolve_ticker_smart(raw: str):
    raw = raw.strip().upper()
    if "." in raw:
        base = raw.split(".")[0]
        candidates = [raw, base, base+".NS", base+".BO", base+".L", base+".AX"]
    else:
        candidates = [raw, raw+".NS", raw+".BO", raw+".L", raw+".AX", raw+".TO", raw+".F"]
    for candidate in candidates:
        try:
            t = yf.Ticker(candidate)
            info = t.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price is None:
                hist = t.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None and float(price) > 0:
                return candidate, float(price)
        except Exception:
            continue
    return raw, None

def resolve_holdings_tickers(raw_holdings: dict) -> dict:
    resolved = {}
    bad = []
    for raw_ticker, qty in raw_holdings.items():
        good_ticker, price = resolve_ticker_smart(raw_ticker)
        if price is None:
            bad.append(raw_ticker)
        resolved[good_ticker] = qty
    if bad:
        st.warning(f"Could not resolve: {', '.join(bad)}")
    return resolved

@st.cache_data(ttl=86400)
def get_yf_universe():
    try:
        return sorted(set(yf.tickers_sp500() + yf.tickers_dow()))
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "TCS.NS", "INFY.NS", "RELIANCE.NS"]

# ─────────────────────────────────────────────
# Result renderer
# ─────────────────────────────────────────────
def render_result_payload(payload: Any, context: Dict[str, Any] | None = None):
    st.session_state.last_payload = payload

    def _err_box(msg):
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);
                    border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.4rem">
            <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#f87171;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px">⚠ Tool Error</div>
            <div style="font-size:0.88rem;color:#fca5a5;line-height:1.5">{msg}</div>
        </div>""", unsafe_allow_html=True)

    if isinstance(payload, dict) and list(payload.keys()) == ["error"]:
        _err_box(payload["error"]); return
    if isinstance(payload, str):
        if payload.lower().startswith("error"):
            _err_box(payload); return
        st.code(payload); return

    # Savings forecast
    if isinstance(payload, dict) and "Derived_Values" in payload and "Predicted_Savings" in payload:
        _save_artifact("forecast_savings", payload)
        derived = payload["Derived_Values"]
        savings = payload["Predicted_Savings"]
        st.markdown('<div class="sec-label">💰 Budget Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Income", human_currency(derived.get("Income"), "INR"))
        c2.metric("Desired Savings %", f"{derived.get('Desired_Savings_Percentage', 0)}%")
        c3.metric("Desired Savings", human_currency(derived.get("Desired_Savings"), "INR"))
        c4.metric("Disposable Income", human_currency(derived.get("Disposable_Income"), "INR"))
        rows = [{"Category": k.split("_")[-1], "Potential_Saving": float(v or 0)} for k, v in savings.items()]
        df = pd.DataFrame(rows).sort_values("Potential_Saving", ascending=False)
        df["Display"] = df["Potential_Saving"].apply(lambda x: human_currency(x, "INR"))
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Table"])
        with tab1:
            fig = go.Figure(go.Bar(
                x=df["Category"], y=df["Potential_Saving"],
                marker=dict(color=df["Potential_Saving"],
                            colorscale=[[0,"#14532d"],[0.5,"#22c55e"],[1,"#86efac"]],
                            line=dict(color="rgba(34,197,94,0.3)",width=1)),
                text=[human_currency(v,"INR") for v in df["Potential_Saving"]],
                textposition="outside", textfont=dict(color="#86efac",size=10),
            ))
            fig.update_layout(title="Potential Savings per Category", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')
        with tab2:
            fig2 = go.Figure(go.Pie(
                labels=df["Category"], values=df["Potential_Saving"], hole=0.45,
                textinfo="label+percent", textfont=dict(color="#e8f5ea"),
                marker=dict(colors=["#4ade80","#22c55e","#86efac","#a3e635","#84cc16","#65a30d","#bbf7d0"],
                            line=dict(color="#030a05",width=2)),
            ))
            fig2.update_layout(title="Savings Distribution", **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, width='stretch')
        with tab3:
            st.dataframe(df[["Category","Display"]], width='stretch', hide_index=True)
        quart = payload.get("Quartiles") or {}
        if quart:
            qt_rows = [{"Category": cat, "Q1": qv.get("Q1"), "Q2 (Median)": qv.get("Q2"), "Q3": qv.get("Q3")}
                       for cat, qv in quart.items()]
            qt_df = pd.DataFrame(qt_rows)
            fig3 = go.Figure()
            for col, color in [("Q1","#4ade80"),("Q2 (Median)","#22c55e"),("Q3","#86efac")]:
                fig3.add_trace(go.Bar(name=col, x=qt_df["Category"], y=qt_df[col], marker_color=color))
            fig3.update_layout(barmode="group", title="Spending Quartiles", **PLOTLY_LAYOUT)
            st.plotly_chart(fig3, width='stretch')
        fb = payload.get("Quartile_Feedback") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**⚠️ Slightly High**")
            for item in fb.get("Slightly_High_Q2_Q3", []):
                st.markdown(f'<span style="background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.3);border-radius:5px;padding:2px 9px;font-size:0.8rem;color:#fde047">{item}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown("**🚨 Highly Overspending**")
            for item in fb.get("Highly_Overspending_Above_Q3", []):
                st.markdown(f'<span style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.25);border-radius:5px;padding:2px 9px;font-size:0.8rem;color:#f87171">{item}</span>', unsafe_allow_html=True)
        return

    # Portfolio summary
    if isinstance(payload, dict) and "analytics" in payload and "holdings" in payload:
        _save_artifact("portfolio_summary", payload)
        base = payload.get("base_currency", "INR")
        analytics = payload.get("analytics", {})
        st.markdown('<div class="sec-label">🏦 Portfolio Overview</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Value", human_currency(payload.get("portfolio_value_base"), base))
        k2.metric("Holdings", analytics.get("num_holdings", 0))
        top_p = analytics.get("top_performers") or []
        worst_p = analytics.get("worst_performers") or []
        if top_p:
            k3.metric("Best Performer", top_p[0].get("ticker","—"), f"{top_p[0].get('return_pct_since_purchase',0):.2f}%")
        if worst_p:
            k4.metric("Worst Performer", worst_p[0].get("ticker","—"), f"{worst_p[0].get('return_pct_since_purchase',0):.2f}%")
        alloc = analytics.get("sector_allocation", {})
        tab1, tab2, tab3, tab4 = st.tabs(["Sector", "Performance", "Holdings", "Raw"])
        with tab1:
            if alloc:
                sectors, weights = list(alloc.keys()), [v * 100 for v in alloc.values()]
                ca, cb = st.columns(2)
                with ca:
                    fig = go.Figure(go.Pie(labels=sectors, values=weights, hole=0.5,
                        textinfo="label+percent", textfont=dict(color="#e8f5ea",size=11),
                        marker=dict(colors=["#4ade80","#22c55e","#86efac","#a3e635","#bbf7d0","#65a30d","#84cc16"],
                                    line=dict(color="#030a05",width=2))))
                    fig.update_layout(title="Sector Weights", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width='stretch')
                with cb:
                    fig2 = go.Figure(go.Bar(x=sectors, y=weights,
                        marker=dict(color=weights, colorscale=[[0,"#14532d"],[1,"#4ade80"]],
                                    line=dict(color="rgba(34,197,94,0.3)",width=1)),
                        text=[f"{w:.1f}%" for w in weights], textposition="outside",
                        textfont=dict(color="#86efac",size=10)))
                    fig2.update_layout(title="Sector Bar", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, width='stretch')
        with tab2:
            all_perf = (top_p or []) + (worst_p or [])
            if all_perf:
                pdf = pd.DataFrame(all_perf).sort_values("return_pct_since_purchase", ascending=False) if "return_pct_since_purchase" in pd.DataFrame(all_perf).columns else pd.DataFrame(all_perf)
                if "return_pct_since_purchase" in pdf.columns:
                    colors = ["#4ade80" if v >= 0 else "#f87171" for v in pdf["return_pct_since_purchase"]]
                    fig = go.Figure(go.Bar(x=pdf["ticker"], y=pdf["return_pct_since_purchase"],
                        marker=dict(color=colors), text=[f"{v:.2f}%" for v in pdf["return_pct_since_purchase"]],
                        textposition="outside", textfont=dict(color="#86efac")))
                    fig.update_layout(title="Return % Since Purchase", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width='stretch')
        with tab3:
            table = pd.DataFrame(payload["holdings"])
            cols_order = ["ticker","quantity","price","currency","value_native","value_base","weight","purchase_price","return_pct_since_purchase","sector"]
            table = table[[c for c in cols_order if c in table.columns]]
            if "weight" in table.columns:
                table["weight"] = table["weight"].apply(lambda x: f"{float(x)*100:.2f}%")
            st.dataframe(table, width='stretch', hide_index=True)
        with tab4:
            st.json(payload)
        return

    # Live price — guard against screener items (which also have "price" + "pe_ratio"/"sector")
    if isinstance(payload, dict) and "price" in payload and "pe_ratio" not in payload and "sector" not in payload:
        cur = infer_currency_from_ticker(payload.get("ticker"))
        st.markdown('<div class="sec-label">📈 Live Price</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ticker", payload.get("ticker","—"))
        c2.metric("Company", payload.get("company","—"))
        c3.metric("Price", human_currency(payload.get("price"), payload.get("currency", cur)))
        c4.metric("Market", payload.get("market","—"))
        mcap = payload.get("market_cap")
        c5.metric("Market Cap", human_currency(mcap, payload.get("currency", cur)) if mcap else "—")
        # 1-year price history chart with moving averages
        raw_ticker = payload.get("ticker", "")
        if raw_ticker:
            try:
                hist = yf.download(raw_ticker, period="1y", progress=False)
                if not hist.empty:
                    prices = hist["Close"].squeeze()
                    ma50 = prices.rolling(50).mean()
                    ma200 = prices.rolling(200).mean()
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=prices.index, y=prices.values,
                        mode="lines", name="Price",
                        line=dict(color="#4ade80", width=1.5),
                        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)"
                    ))
                    fig_hist.add_trace(go.Scatter(
                        x=ma50.index, y=ma50.values,
                        mode="lines", name="MA 50",
                        line=dict(color="#22c55e", width=1.5, dash="dot")
                    ))
                    fig_hist.add_trace(go.Scatter(
                        x=ma200.index, y=ma200.values,
                        mode="lines", name="MA 200",
                        line=dict(color="#86efac", width=1.5, dash="dash")
                    ))
                    fig_hist.update_layout(
                        title=f"{raw_ticker} — 1-Year Price History",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({payload.get('currency', cur)})",
                        height=320,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_hist, width='stretch')
            except Exception:
                pass
        return

    # Predict stock profit
    if isinstance(payload, dict) and {"purchase_price","latest_price","predicted_future_price"} <= payload.keys():
        tick = payload.get("ticker","—")
        cur = infer_currency_from_ticker(tick)
        purchase, latest, future = payload.get("purchase_price"), payload.get("latest_price"), payload.get("predicted_future_price")
        pnl = payload.get("profit_loss_purchase_to_future")
        st.markdown('<div class="sec-label">🔮 Prediction Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", tick)
        c2.metric("Buy Price", human_currency(purchase, cur))
        c3.metric("Latest", human_currency(latest, cur))
        c4.metric("Forecast", human_currency(future, cur))
        try:
            pct = ((future - purchase) / purchase) * 100.0
        except Exception:
            pct = None
        colA, colB = st.columns([1, 2])
        with colA:
            colA.metric("Expected P/L", human_currency(pnl, cur), f"{pct:.2f}%" if pct else None)
        with colB:
            labels, values = ["Buy","Latest","Forecast"], [purchase, latest, future]
            colors_bar = ["#22c55e","#4ade80","#86efac" if (future or 0) >= (purchase or 0) else "#f87171"]
            fig = go.Figure(go.Bar(x=labels, y=values,
                marker=dict(color=colors_bar), text=[human_currency(v,cur) for v in values],
                textposition="outside", textfont=dict(color="#86efac")))
            fig.update_layout(title=f"{tick} Price Comparison", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')
        return

    # Screened stocks list
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        df = pd.DataFrame(payload)
        st.markdown('<div class="sec-label">🔍 Screened Results</div>', unsafe_allow_html=True)
        num_cols = [c for c in df.columns if df[c].dtype in ["float64","int64"]]
        tab1, tab2 = st.tabs(["Table","Chart"])
        with tab1:
            st.dataframe(df, width='stretch', hide_index=True)
        with tab2:
            if "pe_ratio" in df.columns and "ticker" in df.columns:
                fig = go.Figure(go.Bar(x=df["ticker"], y=df["pe_ratio"],
                    marker=dict(color=df["pe_ratio"],
                                colorscale=[[0,"#4ade80"],[0.5,"#22c55e"],[1,"#14532d"]],
                                line=dict(color="rgba(34,197,94,0.2)",width=1)),
                    text=[f"{v:.1f}" for v in df["pe_ratio"]], textposition="outside",
                    textfont=dict(color="#86efac")))
                fig.update_layout(title="P/E Ratios", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
        return

    # Generic dict
    if isinstance(payload, dict):
        if any(isinstance(v, dict) for v in payload.values()):
            for k, v in payload.items():
                label = k.replace("_"," ").title()
                if isinstance(v, dict):
                    st.markdown(f'<div class="sec-label" style="margin-top:0.8rem">{label}</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([[ik,iv] for ik,iv in v.items()], columns=["Metric","Value"]), width='stretch', hide_index=True)
                elif isinstance(v, list):
                    st.markdown(f'<div class="sec-label" style="margin-top:0.8rem">{label}</div>', unsafe_allow_html=True)
                    if v and isinstance(v[0], dict):
                        st.dataframe(pd.DataFrame(v), width='stretch', hide_index=True)
                    else:
                        st.write(", ".join(str(i) for i in v))
                else:
                    st.metric(label, str(v))
        else:
            st.dataframe(pd.DataFrame([[k,v] for k,v in payload.items()], columns=["Key","Value"]), width='stretch', hide_index=True)
        return

    if isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), width='stretch', hide_index=True); return
    st.code(json.dumps(payload, indent=2))

def render_result(raw: Any):
    if isinstance(raw, list):
        items = []
        for chunk in raw:
            try:
                data = json.loads(chunk.text)
            except Exception:
                data = getattr(chunk, "text", chunk)
            items.append(data)
        # If multiple dict chunks (e.g. screener returning one row per chunk), merge into list
        if len(items) > 1 and all(isinstance(i, dict) for i in items):
            render_result_payload(items)
        else:
            for item in items:
                render_result_payload(item)
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = raw
        render_result_payload(data)

# ─────────────────────────────────────────────
# MCP Connection
# ─────────────────────────────────────────────
async def _connect_async():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SERVER_PATH = os.path.join(BASE_DIR, "server.py")
    server = StdioServerParameters(command=sys.executable, args=[SERVER_PATH], cwd=BASE_DIR)
    client = stdio_client(server)
    read_stream, write_stream = await client.__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    init = await session.initialize()
    tools = await session.list_tools()
    st.session_state.mcp_client = client
    st.session_state.read_stream = read_stream
    st.session_state.write_stream = write_stream
    st.session_state.mcp_session = session
    st.session_state.tools = [{"name": t.name, "description": t.description} for t in tools.tools]
    st.session_state.connected = True
    st.success(f"Connected — {init.serverInfo.name}")

def connect_to_server():
    try:
        loop.run_until_complete(_connect_async())
    except Exception as e:
        st.error(f"Connection failed: {e}")

async def _disconnect_async():
    for attr in ("mcp_session", "mcp_client"):
        obj = st.session_state.get(attr)
        if obj:
            try: await obj.__aexit__(None, None, None)
            except Exception: pass
    for k in ("mcp_client","mcp_session","read_stream","write_stream"):
        st.session_state[k] = None

def _disconnect_click():
    try:
        loop.run_until_complete(_disconnect_async())
    finally:
        st.session_state.connected = False
        st.session_state.tools = []
        st.session_state.filtered_tickers = None
        st.session_state.filtered_details = None
        st.rerun()

def _require_session() -> ClientSession | None:
    if not st.session_state.connected or st.session_state.mcp_session is None:
        st.warning("Connect to the MCP server first.")
        return None
    return st.session_state.mcp_session

def execute_tool(tool_name: str, args: dict):
    session = _require_session()
    if session is None: return
    async def run(sess):
        with st.spinner(f"Running {tool_name}…"):
            result = await sess.call_tool(tool_name, args)
        st.success("Done")
        st.markdown('<div class="sec-label" style="margin-top:1rem">📊 Result</div>', unsafe_allow_html=True)
        render_result(result.content)
    loop.run_until_complete(run(session))

def call_tool_return_json(tool_name: str, args: dict):
    session = _require_session()
    if session is None: return None
    async def run(sess):
        result = await sess.call_tool(tool_name, args)
        rows = []
        for item in result.content:
            try: rows.append(json.loads(item.text))
            except Exception: rows.append(item.text)
        return rows[0] if len(rows) == 1 else rows
    return loop.run_until_complete(run(session))

# ─────────────────────────────────────────────
# Onboarding dialog (shown on first load)
# ─────────────────────────────────────────────
@st.dialog("Welcome to FinBot — Build Your Profile", width="large")
def show_onboarding():
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#4b6e54;
                    text-transform:uppercase;letter-spacing:0.15em">
            I'm your AI financial co-pilot. Let me learn about your finances to give you personalised advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio("Input method", ["Manual Form", "Paste JSON"], horizontal=True, label_visibility="collapsed")

    if input_mode == "Manual Form":
        st.markdown("#### 💵 Earnings & Savings")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"], index=0)
        with col_b:
            income = st.number_input("Monthly Income", min_value=0.0, value=90000.0, step=1000.0)
        with col_c:
            savings_goal = st.slider("Savings Goal %", 1, 80, 20)

        st.markdown("#### 🧾 Monthly Expenditure")
        e1, e2, e3 = st.columns(3)
        with e1:
            rent        = st.number_input("Rent / Housing", min_value=0.0, value=15000.0, step=500.0)
            groceries   = st.number_input("Groceries", min_value=0.0, value=8000.0, step=500.0)
            transport   = st.number_input("Transport", min_value=0.0, value=3000.0, step=500.0)
            loan        = st.number_input("Loan Repayment", min_value=0.0, value=5000.0, step=500.0)
        with e2:
            utilities   = st.number_input("Utilities", min_value=0.0, value=3000.0, step=500.0)
            eating_out  = st.number_input("Eating Out", min_value=0.0, value=3000.0, step=500.0)
            entertainment = st.number_input("Entertainment", min_value=0.0, value=2000.0, step=500.0)
            insurance   = st.number_input("Insurance", min_value=0.0, value=2000.0, step=500.0)
        with e3:
            healthcare  = st.number_input("Healthcare", min_value=0.0, value=1500.0, step=500.0)
            education   = st.number_input("Education", min_value=0.0, value=0.0, step=500.0)
            misc        = st.number_input("Miscellaneous", min_value=0.0, value=3000.0, step=500.0)

        total_exp = rent+groceries+transport+loan+utilities+eating_out+entertainment+insurance+healthcare+education+misc
        surplus = income - total_exp
        ca, cb, cc = st.columns(3)
        ca.metric("Total Expenses", human_currency(total_exp, currency))
        cb.metric("Surplus", human_currency(surplus, currency), f"{surplus/income*100:.1f}%" if income > 0 else None)
        cc.metric("Target Savings", human_currency(income * savings_goal / 100, currency))

        st.markdown("#### 📊 Your Portfolio (optional)")
        portfolio_text = st.text_input(
            "Holdings — format: TICKER:QTY, …",
            placeholder="AAPL:10, TSLA:5, TCS.NS:20",
        )

        if st.button("Save Profile & Continue", type="primary"):
            raw_holdings = {}
            if portfolio_text.strip():
                for item in portfolio_text.split(","):
                    try:
                        t, q = item.strip().split(":")
                        raw_holdings[t.strip().upper()] = float(q.strip())
                    except Exception:
                        pass
            st.session_state.user_profile = {
                "currency": currency,
                "income": income,
                "savings_goal_pct": savings_goal,
                "expenses": {
                    "Rent": rent, "Groceries": groceries, "Transport": transport,
                    "Loan_Repayment": loan, "Utilities": utilities, "Eating_Out": eating_out,
                    "Entertainment": entertainment, "Insurance": insurance,
                    "Healthcare": healthcare, "Education": education, "Miscellaneous": misc,
                },
                "total_expenses": total_exp,
                "surplus": surplus,
                "portfolio": raw_holdings,
            }
            st.session_state.profile_complete = True
            st.rerun()

    else:  # JSON paste
        st.markdown("Paste your financial profile as JSON:")
        sample = {
            "currency": "INR",
            "income": 90000,
            "savings_goal_pct": 20,
            "expenses": {"Rent": 15000, "Groceries": 8000, "Transport": 3000, "Miscellaneous": 5000},
            "portfolio": {"AAPL": 10, "TCS.NS": 20}
        }
        json_text = st.text_area("JSON Profile", value=json.dumps(sample, indent=2), height=260)

        if st.button("Import & Continue", type="primary"):
            try:
                profile = json.loads(json_text)
                profile.setdefault("currency", "INR")
                profile.setdefault("income", 0)
                profile.setdefault("savings_goal_pct", 20)
                profile.setdefault("expenses", {})
                profile.setdefault("portfolio", {})
                profile["total_expenses"] = sum(profile["expenses"].values())
                profile["surplus"] = profile["income"] - profile["total_expenses"]
                st.session_state.user_profile = profile
                st.session_state.profile_complete = True
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    col_skip, _ = st.columns([1, 3])
    with col_skip:
        if st.button("Skip for now"):
            st.session_state.profile_complete = True
            st.rerun()

# ─────────────────────────────────────────────
# Trigger onboarding on first load
# ─────────────────────────────────────────────
if not st.session_state.profile_complete:
    show_onboarding()

# ─────────────────────────────────────────────
# SIDEBAR — Logo + Connect/Disconnect + AI Chat
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="finbot-logo-wrap">
        <div class="finbot-logo">📈 FinBot</div>
        <div class="finbot-tagline">AI Financial Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Connection status + buttons
    status_class = "on" if st.session_state.connected else "off"
    dot_class = "g" if st.session_state.connected else "r"
    status_text = "Connected" if st.session_state.connected else "Disconnected"
    st.markdown(f"""
    <div class="conn-row">
        <div class="conn-status {status_class}">
            <span class="dot {dot_class}"></span>{status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        st.button("⚡ Connect", disabled=st.session_state.connected,
                  on_click=connect_to_server, width='stretch')
    with btn_c2:
        st.button("✕ Disconnect", disabled=not st.session_state.connected,
                  on_click=_disconnect_click, width='stretch')

    st.markdown('<hr style="margin:0.6rem 0"/>', unsafe_allow_html=True)

    # ── AI Chat ──
    st.markdown('<div style="padding:0 0.8rem"><div class="sidebar-chat-label">🤖 AI Chat — Ask FinBot</div></div>', unsafe_allow_html=True)

    # Profile context pill
    if st.session_state.user_profile:
        p = st.session_state.user_profile
        cur = p.get("currency", "INR")
        st.markdown(f"""
        <div style="padding:0 0.8rem;margin-bottom:0.5rem">
        <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.15);border-radius:8px;
                    padding:0.45rem 0.7rem;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#4ade80">
            Income: {human_currency(p.get('income'), cur)} · Expenses: {human_currency(p.get('total_expenses'), cur)} ·
            Savings: {p.get('savings_goal_pct',0)}%
        </div></div>
        """, unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container(height=380)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0.5rem;color:#4b6e54;font-size:0.8rem;font-family:'JetBrains Mono',monospace">
                Ask me about your finances, stocks, or savings…
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.chat_history[-20:]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-label">You</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-bot">
                    <div class="msg-label">FinBot</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)

    # Chat input
    with st.container():
        chat_prompt = st.text_area("", height=70,
            placeholder="Ask anything about your money…",
            key="sidebar_chat_input", label_visibility="collapsed")

        sb_c1, sb_c2 = st.columns([3, 1])
        with sb_c1:
            ask_btn = st.button("Send ↗", type="primary", width='stretch')
        with sb_c2:
            if st.button("Clear", width='stretch'):
                st.session_state.chat_history = []
                st.rerun()

    if ask_btn and chat_prompt.strip():
        prompt = chat_prompt.strip()
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Inject profile context into prompt
        profile_ctx = ""
        if st.session_state.user_profile:
            p = st.session_state.user_profile
            profile_ctx = (
                f"\n\nUser financial profile: Income={p.get('income')}, "
                f"Expenses={p.get('total_expenses')}, "
                f"Savings goal={p.get('savings_goal_pct')}%, "
                f"Expense breakdown={json.dumps(p.get('expenses',{}))}, "
                f"Portfolio={json.dumps(p.get('portfolio',{}))}"
            )
        enriched_prompt = prompt + profile_ctx

        decision = agent.decide_tool(prompt)

        if decision.tool == "none" or not st.session_state.connected:
            with st.spinner("Thinking…"):
                answer = agent.general_answer(enriched_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        elif "buy" in prompt.lower() or "sell" in prompt.lower():
            ticker = agent.resolve_ticker(prompt)
            plan = planner.plan(enriched_prompt)
            tool_outputs = {}
            if ticker and st.session_state.connected:
                price = call_tool_return_json("get_current_stock_price", {"ticker": ticker})
                tool_outputs["price"] = price
                trend = call_tool_return_json("analyze_market_trend", {"ticker": ticker})
                tool_outputs["trend"] = trend
            advice = advisor.generate_advice(enriched_prompt, tool_outputs)
            full_answer = f"**Plan:** {plan}\n\n**Advice:** {advice}"
            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        else:
            params = dict(decision.parameters) if decision.parameters else {}
            if decision.tool in ("portfolio_summary","analyze_portfolio_risk") and "holdings" not in params:
                portfolio = st.session_state.user_profile.get("portfolio", {}) if st.session_state.user_profile else {}
                if portfolio:
                    params["holdings"] = portfolio
                    if decision.tool == "portfolio_summary":
                        params.setdefault("base_currency", st.session_state.user_profile.get("currency","INR"))
            result = call_tool_return_json(decision.tool, params)
            if result:
                answer = agent.general_answer(f"{enriched_prompt}\n\nTool result: {json.dumps(result)}")
            else:
                answer = agent.general_answer(enriched_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.rerun()

    # Edit profile button
    st.markdown('<hr style="margin:0.4rem 0"/>', unsafe_allow_html=True)
    if st.button("✏️ Edit Profile", width='stretch'):
        st.session_state.profile_complete = False
        st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA — Header + Tools
# ─────────────────────────────────────────────
# Header row
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="main-title">Financial Intelligence Dashboard</div>
            <div class="main-subtitle">Live market tools · Portfolio analytics · AI-powered insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Profile summary strip (if profile exists)
if st.session_state.user_profile:
    p = st.session_state.user_profile
    cur = p.get("currency", "INR")
    total_exp = p.get("total_expenses", 0)
    income = p.get("income", 0)
    surplus = p.get("surplus", 0)
    goal = p.get("savings_goal_pct", 0)
    portfolio = p.get("portfolio", {})
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Monthly Income", human_currency(income, cur))
    pc2.metric("Total Expenses", human_currency(total_exp, cur))
    pc3.metric("Surplus", human_currency(surplus, cur),
               f"{surplus/income*100:.1f}%" if income > 0 else None)
    pc4.metric("Savings Goal", f"{goal}%")
    pc5.metric("Portfolio Size", f"{len(portfolio)} holdings" if portfolio else "Not set")
    st.markdown("<hr/>", unsafe_allow_html=True)

# ── Tool selector ──
TOOL_META = {
    "get_current_stock_price": ("📈", "Live Price"),
    "portfolio_summary":       ("🏦", "Portfolio"),
    "analyze_portfolio_risk":  ("⚠️", "Risk Analysis"),
    "analyze_portfolio_risk_tool": ("⚠️", "Risk Analysis"),
    "analyze_market_trend":    ("📉", "Market Trend"),
    "screen_stocks":           ("🔍", "Stock Screener"),
    "predict_stock_profit":    ("🔮", "Profit Predictor"),
    "forecast_savings":        ("💰", "Savings Forecast"),
}

if not st.session_state.connected or not st.session_state.tools:
    st.markdown("""
    <div style="text-align:center;padding:5rem 0;border:1px dashed rgba(34,197,94,0.15);
                border-radius:16px;margin-top:1rem">
        <div style="font-size:2.5rem;margin-bottom:0.8rem">⚡</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#4b6e54;font-weight:600">
            Connect to the MCP Server
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#2d4a35;margin-top:0.4rem">
            Use the Connect button in the sidebar to load tools
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Deduplicate: drop analyze_portfolio_risk_tool if analyze_portfolio_risk is present
    _raw_tools = [t["name"] for t in st.session_state.tools]
    if "analyze_portfolio_risk" in _raw_tools and "analyze_portfolio_risk_tool" in _raw_tools:
        _raw_tools = [n for n in _raw_tools if n != "analyze_portfolio_risk_tool"]
    tool_names = _raw_tools
    if st.session_state.selected_tool not in tool_names:
        st.session_state.selected_tool = tool_names[0]

    # Interactive pill selector
    pill_labels = [f"{TOOL_META.get(tn, ('⚙️',''))[0]} {TOOL_META.get(tn, ('⚙️', tn.replace('_',' ').title()))[1]}" for tn in tool_names]
    _default_idx = tool_names.index(st.session_state.selected_tool) if st.session_state.selected_tool in tool_names else 0
    selected_pill = st.pills("Select Tool", pill_labels, default=pill_labels[_default_idx], label_visibility="collapsed")
    # Map pill label back to tool name
    pill_to_tool = {f"{TOOL_META.get(tn, ('⚙️',''))[0]} {TOOL_META.get(tn, ('⚙️', tn.replace('_',' ').title()))[1]}": tn for tn in tool_names}
    selected_tool = pill_to_tool.get(selected_pill, tool_names[_default_idx])
    st.session_state.selected_tool = selected_tool

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── get_current_stock_price ──
    if selected_tool == "get_current_stock_price":
        st.markdown("### 📈 Live Stock Price")
        ticker = st.text_input("Ticker Symbol", "AAPL", placeholder="AAPL, TCS, RELIANCE, MSFT")
        st.caption("Exchange suffix auto-detected (TCS → TCS.NS)")
        if st.button("▶ Get Live Price", type="primary"):
            resolved, _ = resolve_ticker_smart(ticker)
            execute_tool(selected_tool, {"ticker": resolved})

    # ── portfolio_summary ──
    elif selected_tool == "portfolio_summary":
        st.markdown("### 🏦 Portfolio Summary")
        base_ccy = st.selectbox("Base Currency", ["INR","USD","EUR","GBP"], index=0)
        up = st.file_uploader("Upload Holdings CSV (Ticker, Quantity, PurchasePrice)", type=["csv"])
        universe = get_yf_universe()
        with st.form("portfolio_form"):
            # Pre-fill from user profile if available
            init_rows = []
            if st.session_state.user_profile and st.session_state.user_profile.get("portfolio"):
                for tk, qty in st.session_state.user_profile["portfolio"].items():
                    init_rows.append({"Ticker": tk, "Quantity": float(qty), "PurchasePrice": ""})
            if not init_rows:
                init_rows = [{"Ticker": "AAPL", "Quantity": 10.0, "PurchasePrice": ""}]
            edited = st.data_editor(
                pd.DataFrame(init_rows), num_rows="dynamic",
                column_config={
                    "Ticker": st.column_config.SelectboxColumn(options=universe + ["MRF.NS","TCS.NS","INFY.NS","RELIANCE.NS"]),
                    "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
                    "PurchasePrice": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
                }, width='stretch')
            st.form_submit_button("Apply")

        if up is not None:
            try:
                df_csv = pd.read_csv(up)
                df_csv.columns = [c.strip().lower() for c in df_csv.columns]
                edited = pd.DataFrame([{
                    "Ticker": str(r.get("ticker") or r.get("symbol","")).strip(),
                    "Quantity": float(r.get("quantity") or r.get("qty") or 0.0),
                    "PurchasePrice": float(r.get("purchaseprice") or r.get("ppx") or 0.0) if (r.get("purchaseprice") or r.get("ppx")) else "",
                } for _, r in df_csv.iterrows()])
            except Exception as e:
                st.error(f"CSV error: {e}")

        if st.button("▶ Run Portfolio Summary", type="primary"):
            rows = [(str(r["Ticker"]).strip(), float(r.get("Quantity") or 0), r.get("PurchasePrice")) for _, r in edited.iterrows() if str(r["Ticker"]).strip()]
            if not rows:
                st.warning("Add at least one holding.")
            else:
                raw_h = {tk: qty for tk, qty, _ in rows}
                with st.spinner("Resolving tickers…"):
                    holdings = resolve_holdings_tickers(raw_h)
                ppx = {}
                for tk_raw, qty, pp in rows:
                    resolved_tk, _ = resolve_ticker_smart(tk_raw)
                    if pp not in [None, ""]:
                        try: ppx[resolved_tk] = float(pp)
                        except Exception: pass
                execute_tool(selected_tool, {"holdings": holdings, "purchase_prices": ppx, "base_currency": base_ccy})

    # ── analyze_portfolio_risk ──
    elif selected_tool in ("analyze_portfolio_risk", "analyze_portfolio_risk_tool"):
        st.markdown("### ⚠️ Portfolio Risk Analyzer")
        # Pre-fill from profile
        default_holdings = {}
        if st.session_state.user_profile and st.session_state.user_profile.get("portfolio"):
            default_holdings = st.session_state.user_profile["portfolio"]
        holdings_text = st.text_area("Holdings JSON",
            value=json.dumps(default_holdings if default_holdings else {"AAPL": 10, "MSFT": 5, "GOOGL": 3}, indent=2),
            height=140)
        st.caption("Auto-resolves Indian tickers (TCS → TCS.NS)")
        if st.button("▶ Analyze Risk", type="primary"):
            try:
                raw_h = json.loads(holdings_text)
                with st.spinner("Resolving tickers…"):
                    holdings = resolve_holdings_tickers(raw_h)
                execute_tool(selected_tool, {"holdings": holdings})
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # ── analyze_market_trend ──
    elif selected_tool == "analyze_market_trend":
        st.markdown("### 📉 Market Trend Analyzer")
        ticker = st.text_input("Stock Ticker", "AAPL", placeholder="AAPL, TCS, RELIANCE")
        st.caption("Exchange suffix auto-resolved")
        if st.button("▶ Analyze Trend", type="primary"):
            with st.spinner("Resolving ticker…"):
                resolved, price = resolve_ticker_smart(ticker)
            if price is None:
                st.error(f"No price data for '{ticker}'. Try e.g. TCS.NS")
            else:
                if resolved != ticker.upper():
                    st.info(f"Resolved {ticker} → {resolved}")
                execute_tool(selected_tool, {"ticker": resolved})

    # ── screen_stocks ──
    elif selected_tool == "screen_stocks":
        st.markdown("### 🔍 Portfolio Screener")
        scol1, scol2 = st.columns(2)
        with scol1:
            max_pe = st.number_input("Max P/E Ratio", value=40.0, step=1.0, min_value=0.0)
        with scol2:
            mcap_options = {
                "100M": 100_000_000, "500M": 500_000_000, "1B": 1_000_000_000,
                "5B": 5_000_000_000, "10B": 10_000_000_000, "50B": 50_000_000_000,
            }
            mcap_label = st.selectbox("Min Market Cap", list(mcap_options.keys()), index=2)
            min_mcap = mcap_options[mcap_label]

        manual = st.text_input("Custom tickers (comma-separated)", "", placeholder="AAPL, MSFT, TCS.NS")
        st.caption("Auto-screens S&P 500 + DOW 30 universe. Add custom tickers above.")

        sc1, sc2 = st.columns(2)
        with sc1:
            find_clicked = st.button("🔍 Find Matching Stocks", type="primary")
        with sc2:
            run_clicked = st.button("▶ Run Screener on Selection")

        if find_clicked:
            universe = get_yf_universe()
            with st.spinner(f"Screening {len(universe)} stocks…"):
                result = call_tool_return_json("screen_stocks", {
                    "criteria": {"tickers": universe, "max_pe": max_pe, "min_market_cap": min_mcap}
                })
                eligible = [r.get("ticker") for r in (result or []) if isinstance(r, dict) and not r.get("error")]
                st.session_state.filtered_details = result
                st.session_state.filtered_tickers = sorted(set(t for t in eligible if t))

        if st.session_state.filtered_tickers is not None:
            st.success(f"✅ {len(st.session_state.filtered_tickers)} stocks matched")
            selected_from_list = st.multiselect("Select from matched stocks", st.session_state.filtered_tickers)
        else:
            selected_from_list = []

        if run_clicked:
            final_tickers = list(selected_from_list)
            final_tickers += [t.strip() for t in manual.split(",") if t.strip()]
            final_tickers = sorted(set(final_tickers))
            if not final_tickers:
                st.warning("Select at least one ticker.")
            elif manual.strip():
                execute_tool("screen_stocks", {"criteria": {"tickers": final_tickers}})
            else:
                execute_tool("screen_stocks", {"criteria": {"tickers": final_tickers, "max_pe": max_pe, "min_market_cap": min_mcap}})

    # ── predict_stock_profit ──
    elif selected_tool == "predict_stock_profit":
        st.markdown("### 🔮 Stock Profit Predictor")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            ticker = st.text_input("Ticker", "MRF.NS")
        with pc2:
            purchase_date = st.date_input("Purchase Date", date(2025, 1, 1))
        with pc3:
            future_date = st.date_input("Future Date", date(2026, 6, 1))
        if st.button("▶ Predict Profit", type="primary"):
            pd_str = purchase_date.strftime("%Y-%m-%d")
            fd_str = future_date.strftime("%Y-%m-%d")
            with st.spinner("Resolving ticker…"):
                resolved_ticker, price = resolve_ticker_smart(ticker)
            if price is None:
                st.error(f"No price data for '{ticker}'.")
            else:
                if resolved_ticker != ticker.upper():
                    st.info(f"Resolved {ticker} → {resolved_ticker}")
                execute_tool(selected_tool, {"ticker": resolved_ticker, "purchase_date": pd_str, "future_date": fd_str})

    # ── forecast_savings ──
    elif selected_tool == "forecast_savings":
        st.markdown("### 💰 Savings Forecaster")

        # Pre-fill from user profile
        profile = st.session_state.user_profile or {}
        expenses = profile.get("expenses", {})

        fc1, fc2 = st.columns([2, 1])
        with fc1:
            income = st.number_input("Monthly Income", min_value=0.0,
                value=float(profile.get("income", 90000)), step=1000.0)
        with fc2:
            desired_pct = st.slider("Savings Goal %", 1, 80,
                int(profile.get("savings_goal_pct", 20)))

        st.markdown("#### Expense Breakdown")
        fe1, fe2, fe3 = st.columns(3)
        with fe1:
            rent       = st.number_input("Rent", min_value=0.0, value=float(expenses.get("Rent", 15000)), step=500.0)
            loan       = st.number_input("Loan Repayment", min_value=0.0, value=float(expenses.get("Loan_Repayment", 5000)), step=500.0)
            insurance  = st.number_input("Insurance", min_value=0.0, value=float(expenses.get("Insurance", 2000)), step=500.0)
            groceries  = st.number_input("Groceries", min_value=0.0, value=float(expenses.get("Groceries", 8000)), step=500.0)
        with fe2:
            transport  = st.number_input("Transport", min_value=0.0, value=float(expenses.get("Transport", 3000)), step=500.0)
            eating_out = st.number_input("Eating Out", min_value=0.0, value=float(expenses.get("Eating_Out", 3000)), step=500.0)
            entertainment = st.number_input("Entertainment", min_value=0.0, value=float(expenses.get("Entertainment", 2000)), step=500.0)
            utilities  = st.number_input("Utilities", min_value=0.0, value=float(expenses.get("Utilities", 3000)), step=500.0)
        with fe3:
            healthcare = st.number_input("Healthcare", min_value=0.0, value=float(expenses.get("Healthcare", 1500)), step=500.0)
            education  = st.number_input("Education", min_value=0.0, value=float(expenses.get("Education", 0)), step=500.0)
            misc       = st.number_input("Miscellaneous", min_value=0.0, value=float(expenses.get("Miscellaneous", 3000)), step=500.0)

        total_exp = rent+loan+insurance+groceries+transport+eating_out+entertainment+utilities+healthcare+education+misc
        surplus = income - total_exp
        cur = profile.get("currency", "INR")

        prev_c1, prev_c2, prev_c3 = st.columns(3)
        prev_c1.metric("Total Expenses", human_currency(total_exp, cur))
        prev_c2.metric("Surplus", human_currency(surplus, cur),
                       f"{surplus/income*100:.1f}%" if income > 0 else None)
        prev_c3.metric("Target Savings", human_currency(income * desired_pct / 100, cur))

        exp_df = pd.DataFrame({
            "Category": ["Rent","Loan","Insurance","Groceries","Transport","Eating Out","Entertainment","Utilities","Healthcare","Education","Misc"],
            "Amount": [rent,loan,insurance,groceries,transport,eating_out,entertainment,utilities,healthcare,education,misc]
        })
        exp_df = exp_df[exp_df["Amount"] > 0].sort_values("Amount", ascending=False)
        if not exp_df.empty:
            fig = go.Figure(go.Pie(
                labels=exp_df["Category"], values=exp_df["Amount"], hole=0.42,
                textinfo="label+percent", textfont=dict(color="#e8f5ea"),
                marker=dict(
                    colors=["#4ade80","#22c55e","#86efac","#a3e635","#bbf7d0","#65a30d","#84cc16","#16a34a","#15803d","#14532d","#052e16"],
                    line=dict(color="#030a05", width=2))
            ))
            fig.update_layout(title="Current Expense Distribution", height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')

        # Income quartile benchmark chart
        _INCOME_BENCHMARKS = {
            "INR": {"Q1": 25000, "Q2": 45000, "Q3": 80000, "Q4+": 150000},
            "USD": {"Q1": 3000,  "Q2": 5500,  "Q3": 9000,  "Q4+": 16000},
            "EUR": {"Q1": 2200,  "Q2": 3800,  "Q3": 6500,  "Q4+": 12000},
            "GBP": {"Q1": 2500,  "Q2": 4200,  "Q3": 7000,  "Q4+": 13000},
        }
        benchmarks = _INCOME_BENCHMARKS.get(cur, _INCOME_BENCHMARKS["INR"])
        bm_labels = list(benchmarks.keys())
        bm_values = list(benchmarks.values())
        bar_colors = []
        user_percentile = "Q4+"
        for label, val in benchmarks.items():
            if income <= val:
                user_percentile = label
                break
        for label in bm_labels:
            bar_colors.append("#4ade80" if label == user_percentile else "rgba(34,197,94,0.25)")
        fig_qrt = go.Figure()
        fig_qrt.add_trace(go.Bar(
            x=bm_labels, y=bm_values,
            marker=dict(color=bar_colors, line=dict(color="rgba(34,197,94,0.4)", width=1)),
            name="Benchmark",
            text=[human_currency(v, cur) for v in bm_values],
            textposition="outside", textfont=dict(color="#86efac", size=10),
        ))
        fig_qrt.add_hline(
            y=income,
            line=dict(color="#facc15", width=2, dash="dot"),
            annotation_text=f"Your Income: {human_currency(income, cur)}",
            annotation_position="top right",
            annotation_font=dict(color="#facc15", size=11),
        )
        fig_qrt.update_layout(
            title=f"Income Percentile — You are in <b>{user_percentile}</b>",
            xaxis_title="Percentile Band",
            yaxis_title=f"Monthly Income ({cur})",
            height=280,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_qrt, width='stretch')

        if st.button("▶ Forecast My Savings", type="primary"):
            budget_dict = {
                "Income": income, "Rent": rent, "Loan_Repayment": loan, "Insurance": insurance,
                "Groceries": groceries, "Transport": transport, "Eating_Out": eating_out,
                "Entertainment": entertainment, "Utilities": utilities, "Healthcare": healthcare,
                "Education": education, "Miscellaneous": misc,
            }
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            with open(tmp.name, "w") as f:
                json.dump(budget_dict, f, indent=2)
            execute_tool(selected_tool, {"json_path": tmp.name, "desired_savings_percentage": desired_pct})
