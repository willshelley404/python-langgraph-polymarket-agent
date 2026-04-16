#!/usr/bin/env python3
"""
PredictIQ — Agentic Prediction Market Intelligence Platform
═══════════════════════════════════════════════════════════
Orchestration : LangGraph supervisor + 4 specialist agents
LLM           : Groq (llama-3.3-70b-versatile)
Market data   : Polymarket Gamma API (free, no key)
Economic data : FRED (St. Louis Fed) API
ML            : K-Means clustering · PCA · Silhouette sweep
UI            : Shiny for Python
"""

import os, json, re, asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Dict, Any

import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import plotly.graph_objects as go
import plotly.express as px

from shiny import App, ui, render, reactive
from htmltools import HTML

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"
ACCENT       = "#00d4aa"

FRED_SERIES: Dict[str, str] = {
    "UNRATE":      "Unemployment Rate (%)",
    "CPIAUCSL":    "CPI / Inflation Index",
    "FEDFUNDS":    "Fed Funds Rate (%)",
    "T10Y2Y":      "Yield Curve 10Y-2Y",
    "VIXCLS":      "VIX Volatility Index",
    "UMCSENT":     "Consumer Sentiment",
    "PAYEMS":      "Nonfarm Payrolls (000s)",
    "HOUST":       "Housing Starts (000s)",
    "RETAILSMNSA": "Retail Sales",
    "GDP":         "Real GDP (Billion $)",
}

POLY_CATS: Dict[str, str] = {
    "":            "All Categories",
    "politics":    "Politics",
    "crypto":      "Crypto",
    "economics":   "Economics",
    "sports":      "Sports",
    "science":     "Science",
    "technology":  "Technology",
    "pop-culture": "Pop Culture",
}

# ══════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    query:           str
    category:        str
    fred_series:     List[str]
    market_data:     Any
    fred_data:       Dict[str, Any]
    cluster_results: Dict[str, Any]
    insights:        str
    log:             List[str]
    next:            str


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════
#  AGENT NODES
# ══════════════════════════════════════════════════════════════════

def supervisor_node(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] SUPERVISOR  evaluating pipeline state...")
    nxt = (
        "polymarket" if state.get("market_data") is None
        else "fred"       if not state.get("fred_data")
        else "analysis"   if not state.get("cluster_results")
        else "insights"   if not state.get("insights")
        else "END"
    )
    log.append(f"[{_ts()}]             routing -> {nxt}")
    return {"next": nxt, "log": log}


def polymarket_agent(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] POLYMARKET  connecting to Gamma API...")
    try:
        params: Dict[str, Any] = {
            "limit": 200, "active": "true", "closed": "false",
            "order": "volume", "ascending": "false",
        }
        if cat := state.get("category", ""):
            params["tag_slug"] = cat
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params=params, timeout=25)
        r.raise_for_status()
        out: List[Dict] = []
        for m in r.json():
            try:
                prices   = m.get("outcomePrices", [])
                prices   = json.loads(prices) if isinstance(prices, str) else (prices or [])
                yes_p    = float(prices[0]) if prices else 0.5
                out.append({
                    "id":        m.get("id", ""),
                    "question":  m.get("question", ""),
                    "yes_prob":  round(yes_p, 4),
                    "no_prob":   round(1 - yes_p, 4),
                    "volume":    float(m.get("volume",    0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "category":  m.get("category", "General") or "General",
                    "end_date":  (m.get("endDateIso", "") or "")[:10],
                })
            except Exception:
                continue
        log.append(f"[{_ts()}] POLYMARKET  {len(out)} markets loaded")
        return {"market_data": out, "log": log}
    except Exception as e:
        log.append(f"[{_ts()}] POLYMARKET  error: {e}")
        return {"market_data": [], "log": log}


def fred_agent(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] FRED        pulling economic indicators...")
    data: Dict[str, Any] = {}
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    for sid in state.get("fred_series", list(FRED_SERIES)[:5]):
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": sid, "api_key": FRED_API_KEY,
                        "file_type": "json", "observation_start": start,
                        "observation_end": end, "limit": 300, "sort_order": "desc"},
                timeout=14,
            )
            r.raise_for_status()
            obs  = r.json().get("observations", [])
            vals = [(o["date"], float(o["value"]))
                    for o in obs if o["value"] not in (".", "")]
            if vals:
                asc      = sorted(vals, key=lambda x: x[0])
                prev     = vals[min(12, len(vals) - 1)][1]
                pct      = round(((vals[0][1] / prev) - 1) * 100, 2) if prev else 0
                data[sid] = {
                    "name":        FRED_SERIES.get(sid, sid),
                    "dates":       [v[0] for v in asc],
                    "values":      [v[1] for v in asc],
                    "latest":      vals[0][1],
                    "latest_date": vals[0][0],
                    "pct_chg":     pct,
                }
            log.append(f"[{_ts()}] FRED        {sid}: {vals[0][1]:.2f} ({vals[0][0]})")
        except Exception as e:
            log.append(f"[{_ts()}] FRED        {sid} skipped: {str(e)[:80]}")
    return {"fred_data": data, "log": log}


def analysis_agent(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] ANALYSIS    engineering features & clustering...")
    markets = state.get("market_data", [])
    df = pd.DataFrame(markets)
    df = df[df["volume"] > 0].dropna(subset=["yes_prob", "volume", "liquidity"]).copy()
    if len(df) < 5:
        log.append(f"[{_ts()}] ANALYSIS    insufficient data ({len(df)} rows)")
        return {"cluster_results": {}, "log": log}

    df["vol_log"]   = np.log1p(df["volume"])
    df["liq_log"]   = np.log1p(df["liquidity"])
    df["certainty"] = np.abs(df["yes_prob"] - 0.5) * 2
    df["entropy"]   = -(
        df["yes_prob"].clip(1e-6, 1-1e-6) * np.log(df["yes_prob"].clip(1e-6, 1-1e-6)) +
        df["no_prob"].clip(1e-6, 1-1e-6)  * np.log(df["no_prob"].clip(1e-6, 1-1e-6))
    )
    feat_cols = ["yes_prob", "vol_log", "liq_log", "certainty"]
    X = StandardScaler().fit_transform(df[feat_cols].fillna(0))

    best_k, best_s, sil_map = 3, -1.0, {}
    for k in range(2, min(9, max(2, len(df) // 4)) + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(set(lbl)) > 1:
            s = silhouette_score(X, lbl)
            sil_map[k] = round(float(s), 4)
            if s > best_s:
                best_k, best_s = k, s

    km            = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    pca           = PCA(n_components=2)
    Xp            = pca.fit_transform(X)
    df["pc1"], df["pc2"] = Xp[:, 0], Xp[:, 1]

    summaries = []
    med_vol   = df["volume"].median()
    for c in range(best_k):
        cdf   = df[df["cluster"] == c]
        ap    = float(cdf["yes_prob"].mean())
        av    = float(cdf["volume"].mean())
        ac    = float(cdf["certainty"].mean())
        label = (
            "High-Confidence YES" if ap > 0.68 else
            "High-Confidence NO"  if ap < 0.32 else
            "Toss-Up / Uncertain" if ac < 0.15 else
            "High-Volume Active"  if av > med_vol * 2.0 else
            f"Mixed Signals {c}"
        )
        summaries.append({
            "cluster":       c,
            "label":         label,
            "count":         int(len(cdf)),
            "avg_prob":      round(ap, 3),
            "avg_volume":    round(av, 0),
            "avg_certainty": round(ac, 3),
            "top_markets":   cdf.nlargest(3, "volume")["question"].tolist(),
        })

    result = {
        "records":    df.to_dict(orient="records"),
        "n_clusters": best_k,
        "sil_score":  round(float(best_s), 4),
        "sil_map":    sil_map,
        "pca_var":    pca.explained_variance_ratio_.tolist(),
        "summaries":  summaries,
        "total":      int(len(df)),
        "total_vol":  round(float(df["volume"].sum()), 0),
        "avg_prob":   round(float(df["yes_prob"].mean()), 3),
        "fred_snap":  {k: {"name": v["name"], "latest": v["latest"],
                           "pct_chg": v.get("pct_chg", 0)}
                       for k, v in state.get("fred_data", {}).items()},
    }
    log.append(f"[{_ts()}] ANALYSIS    {best_k} clusters (sil={best_s:.3f}), {len(df)} mkts")
    return {"cluster_results": result, "log": log}


def insights_agent(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] INSIGHTS    querying Groq {MODEL}...")
    markets = state.get("market_data", [])
    fred_d  = state.get("fred_data", {})
    cr      = state.get("cluster_results", {})
    query   = state.get("query", "general market intelligence")

    top_m = sorted(markets, key=lambda x: x["volume"], reverse=True)[:12]
    m_txt = "\n".join(
        f"  {m['question']}: {m['yes_prob']*100:.1f}% YES (${m['volume']:,.0f} vol)"
        for m in top_m
    )
    f_txt = "\n".join(
        f"  {v['name']}: {v['latest']:.2f}  ({v.get('pct_chg', 0):+.1f}% yr-over-yr, {v['latest_date']})"
        for v in fred_d.values()
    ) or "  No FRED data available."
    c_txt = "".join(
        f"\n  [{s['label']}] {s['count']} markets | {s['avg_prob']*100:.0f}% avg prob | "
        f"${s['avg_volume']:,.0f} avg vol\n"
        f"    -> {'; '.join(s['top_markets'][:2])}\n"
        for s in cr.get("summaries", [])
    )

    prompt = f"""You are a senior quantitative analyst specialising in prediction markets and macroeconomics.

USER RESEARCH FOCUS: {query}

=== POLYMARKET DATA (top markets by volume) ===
{m_txt}

=== FRED ECONOMIC INDICATORS ===
{f_txt}

=== ML CLUSTERING RESULTS ({cr.get('n_clusters', 0)} clusters, silhouette={cr.get('sil_score', 0):.3f}) ===
{c_txt}

Write a structured intelligence brief with these exact headers:

## Market Sentiment
What collective wisdom do prediction markets reveal? Highlight striking probabilities and consensus signals.

## Macro-Economic Context
How do FRED indicators align with or contradict market signals? Cite specific numbers.

## Cluster Analysis
What market structure patterns emerge from the ML clustering? Which cluster is most actionable and why?

## Key Signals & Risks
Top 3 specific data-driven signals or risks an analyst should act on.

## Divergences & Anomalies
Where do prediction markets and economic data tell conflicting stories? Any informational edges?

Rules: cite actual numbers, be analytical and actionable, ~400 words total."""

    llm = ChatGroq(model=MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    try:
        txt = llm.invoke([HumanMessage(content=prompt)]).content
        log.append(f"[{_ts()}] INSIGHTS    complete ({len(txt):,} chars)")
    except Exception as e:
        txt = f"Groq API error: {e}\n\nEnsure GROQ_API_KEY is set in your environment."
        log.append(f"[{_ts()}] INSIGHTS    error: {e}")
    return {"insights": txt, "log": log}


# ══════════════════════════════════════════════════════════════════
#  LANGGRAPH WORKFLOW
# ══════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",  supervisor_node)
    g.add_node("polymarket",  polymarket_agent)
    g.add_node("fred",        fred_agent)
    g.add_node("analysis",    analysis_agent)
    g.add_node("insights",    insights_agent)
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"polymarket": "polymarket", "fred": "fred",
         "analysis": "analysis", "insights": "insights", "END": END},
    )
    for node in ["polymarket", "fred", "analysis", "insights"]:
        g.add_edge(node, "supervisor")
    return g.compile()


def run_graph(query: str, category: str, fred_series: List[str]) -> AgentState:
    graph = build_graph()
    return graph.invoke({
        "messages":        [],
        "query":           query,
        "category":        category,
        "fred_series":     fred_series,
        "market_data":     None,
        "fred_data":       {},
        "cluster_results": {},
        "insights":        "",
        "log":             [],
        "next":            "",
    })


# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,600&display=swap');

:root {
  --bg:       #0a0e1a;
  --surface:  #131c2e;
  --card:     #192438;
  --border:   #263550;
  --accent:   #00d4aa;
  --accent2:  #7c6af7;
  --warn:     #f59e0b;
  --success:  #22c55e;
  --danger:   #ef4444;
  --text:     #e4edf8;
  --text-dim: #99b0cc;
  --muted:    #527090;
  --mono:     'Space Mono', monospace;
  --sans:     'DM Sans', sans-serif;
}

html, body, .shiny-body-padding {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

/* ── Header ───────────────────────────────────────────────────── */
.app-header {
  background: linear-gradient(135deg, #07091a 0%, #0c1525 60%, #0f1e38 100%);
  border-bottom: 1px solid var(--border);
}
.header-inner {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 26px; gap: 14px;
}
.brand-name {
  font-family: var(--mono); font-size: 1.42rem; font-weight: 700;
  color: var(--accent); letter-spacing: -0.02em; margin: 0;
  text-shadow: 0 0 18px rgba(0,212,170,0.3);
}
.brand-sub {
  font-size: 0.61rem; color: var(--muted); letter-spacing: 0.1em;
  text-transform: uppercase; font-family: var(--mono); margin-top: 2px;
}
.header-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.hbadge {
  font-family: var(--mono); font-size: 0.59rem; padding: 3px 9px;
  border-radius: 4px; border: 1px solid; letter-spacing: 0.05em;
}
.hbadge-green  { border-color: var(--accent);  color: var(--accent);  background: rgba(0,212,170,0.07); }
.hbadge-purple { border-color: var(--accent2); color: var(--accent2); background: rgba(124,106,247,0.07); }
.hbadge-yellow { border-color: var(--warn);    color: var(--warn);    background: rgba(245,158,11,0.07); }

/* ── Help button ──────────────────────────────────────────────── */
.help-btn {
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--text-dim) !important; border-radius: 50% !important;
  width: 30px !important; height: 30px !important; padding: 0 !important;
  font-family: var(--mono) !important; font-size: 0.78rem !important;
  font-weight: 700 !important; cursor: pointer !important;
  line-height: 28px !important; text-align: center !important;
  transition: all 0.15s !important;
}
.help-btn:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Sidebar ──────────────────────────────────────────────────── */
.bslib-sidebar-layout { background: var(--bg) !important; border: none !important; }
.bslib-sidebar-layout > .sidebar {
  background: #0b1425 !important;
  border-right: 1px solid var(--border) !important;
}

.sidebar-label {
  font-family: var(--mono); font-size: 0.57rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
  margin: 14px 0 5px 0; display: block;
}
.query-hint {
  font-size: 0.67rem; color: #608090; margin: -2px 0 5px 0;
  font-family: var(--sans); font-style: italic;
}
.sidebar-hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

/* ── Input controls ───────────────────────────────────────────── */
.form-control, .form-select, textarea.form-control {
  background: #152030 !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; font-family: var(--sans) !important;
  font-size: 0.82rem !important; border-radius: 6px !important;
}
.form-control:focus, .form-select:focus, textarea.form-control:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,170,0.1) !important;
  outline: none !important; background: #1a2840 !important;
}

/* ── Checkbox group — HIGH CONTRAST FIX ──────────────────────── */
/* Target every nesting level Shiny/Bootstrap might generate      */
.form-check { padding: 2px 0 2px 24px !important; margin-bottom: 1px !important; }

.form-check-input {
  background-color: #152030 !important;
  border: 1px solid #4a6888 !important;
  border-radius: 3px !important;
  cursor: pointer !important;
}
.form-check-input:checked {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'%3e%3cpath fill='none' stroke='%230a0e1a' stroke-linecap='round' stroke-linejoin='round' stroke-width='3' d='m6 10 3 3 6-6'/%3e%3c/svg%3e") !important;
}

/* This is the critical rule — override Bootstrap's colour for labels */
.form-check-label,
.shiny-input-container .form-check-label,
#fred_series .form-check-label,
[id="fred_series"] label {
  color: #c0d4ec !important;
  font-size: 0.78rem !important;
  font-family: var(--sans) !important;
  line-height: 1.45 !important;
  cursor: pointer !important;
}
.form-check-label:hover,
.shiny-input-container .form-check-label:hover {
  color: #e4edf8 !important;
}

/* Checkbox group wrapper — force light text in all descendants */
.shiny-input-checkboxgroup label { color: #c0d4ec !important; }
.shiny-input-checkboxgroup .form-check-label { color: #c0d4ec !important; }

/* ── Run button ───────────────────────────────────────────────── */
.run-btn {
  width: 100% !important;
  background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
  color: #06101e !important; border: none !important;
  padding: 11px 16px !important; border-radius: 7px !important;
  font-family: var(--mono) !important; font-size: 0.8rem !important;
  font-weight: 700 !important; letter-spacing: 0.06em !important;
  cursor: pointer !important; transition: all 0.18s ease !important;
  text-transform: uppercase !important; margin-top: 4px !important;
}
.run-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(0,212,170,0.28) !important;
}
.run-btn:active { transform: translateY(0) !important; }

/* ── Agent log ────────────────────────────────────────────────── */
.log-box {
  background: #060c18; border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; max-height: 230px;
  overflow-y: auto; font-family: var(--mono); font-size: 0.57rem; line-height: 1.65;
}
.log-line      { color: #3a5570; white-space: pre-wrap; word-break: break-all; }
.log-line.ok   { color: #4a6888; }
.log-line.last { color: var(--accent); }
.log-running   { color: var(--accent); font-family: var(--mono); font-size: 0.6rem; }
.log-empty     { color: var(--muted); font-style: italic; font-size: 0.68rem; }

/* ── Nav tabs ─────────────────────────────────────────────────── */
.nav-tabs {
  border-bottom: 1px solid var(--border) !important;
  background: #0b1425 !important; padding: 0 16px !important;
}
.nav-tabs .nav-link {
  font-family: var(--mono) !important; font-size: 0.67rem !important;
  letter-spacing: 0.05em !important; text-transform: uppercase !important;
  color: var(--muted) !important; border: none !important;
  padding: 10px 13px !important; border-bottom: 2px solid transparent !important;
  border-radius: 0 !important; transition: color 0.15s !important;
}
.nav-tabs .nav-link:hover  { color: var(--text-dim) !important; }
.nav-tabs .nav-link.active {
  color: var(--accent) !important; background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
}
.tab-content { background: var(--bg) !important; padding: 20px 0 !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
.metrics-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
.metric-card {
  background: var(--card); border: 1px solid var(--border);
  border-top: 2px solid var(--accent); border-radius: 8px;
  padding: 12px 15px; flex: 1; min-width: 110px;
}
.metric-card.m-down { border-top-color: var(--danger); }
.metric-card.m-up   { border-top-color: var(--success); }
.metric-label {
  font-family: var(--mono); font-size: 0.55rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em;
}
.metric-value {
  font-family: var(--mono); font-size: 1.35rem; font-weight: 700;
  color: var(--accent); line-height: 1.15; margin-top: 3px;
}
.metric-delta { font-family: var(--mono); font-size: 0.6rem; color: var(--muted); margin-top: 2px; }
.metric-delta.pos { color: var(--success); }
.metric-delta.neg { color: var(--danger); }

/* ── Data table ───────────────────────────────────────────────── */
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; font-family: var(--sans); }
.data-table thead tr { background: var(--card); border-bottom: 2px solid var(--border); }
.data-table th {
  padding: 9px 13px; text-align: left; font-family: var(--mono);
  font-size: 0.56rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--muted); white-space: nowrap;
}
.data-table td {
  padding: 7px 13px; border-bottom: 1px solid var(--border);
  color: var(--text-dim); background: var(--surface);
}
.data-table tr:hover td { background: var(--card); color: var(--text); }
.q-cell { max-width: 360px; }
.prob-badge {
  display: inline-block; padding: 2px 8px; border-radius: 20px;
  font-family: var(--mono); font-size: 0.66rem; font-weight: 700;
}
.prob-high { background: rgba(34,197,94,0.1);  color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.prob-mid  { background: rgba(245,158,11,0.1); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
.prob-low  { background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

/* ── Loading / empty ──────────────────────────────────────────── */
.loading-wrap {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; min-height: 280px; gap: 14px;
}
.spinner {
  width: 38px; height: 38px; border: 2px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg) } }
.loading-msg { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
.empty-state {
  display: flex; flex-direction: column; align-items: center;
  min-height: 200px; justify-content: center; gap: 8px; color: var(--muted);
}
.empty-icon { font-size: 2rem; opacity: 0.3; font-family: var(--mono); }
.empty-msg  { font-family: var(--mono); font-size: 0.7rem; }

/* ── Welcome screen ───────────────────────────────────────────── */
.welcome-wrap { max-width: 640px; margin: 36px auto; text-align: center; padding: 0 20px; }
.welcome-title {
  font-family: var(--mono); font-size: 1.8rem; color: var(--accent);
  margin-bottom: 12px; text-shadow: 0 0 24px rgba(0,212,170,0.25);
}
.welcome-sub { color: var(--text-dim); font-size: 0.85rem; margin-bottom: 22px; line-height: 1.65; }
.tech-badges { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; margin-bottom: 24px; }
.tbadge {
  font-family: var(--mono); font-size: 0.58rem; padding: 4px 11px;
  border-radius: 4px; border: 1px solid var(--border);
  color: var(--text-dim); background: var(--card); letter-spacing: 0.06em;
}
.pipeline-vis {
  display: flex; align-items: center; justify-content: center;
  flex-wrap: wrap; gap: 6px; background: var(--card);
  border: 1px solid var(--border); border-radius: 10px; padding: 18px;
}
.pipe-node {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 9px 12px; font-family: var(--mono);
  font-size: 0.66rem; color: var(--text-dim); text-align: center; line-height: 1.4;
}
.pipe-node.sup  { border-color: var(--accent);  color: var(--accent); }
.pipe-node.data { border-color: var(--accent2); color: var(--accent2); }
.pipe-node.ml   { border-color: var(--warn);    color: var(--warn); }
.pipe-node.llm  { border-color: #ec4899;        color: #ec4899; }
.pipe-arrow { color: var(--muted); font-size: 0.85rem; }

/* ── Section headings ─────────────────────────────────────────── */
.section-heading {
  font-family: var(--mono); font-size: 0.6rem; text-transform: uppercase;
  letter-spacing: 0.12em; color: var(--muted); margin: 20px 0 10px 0;
  display: flex; align-items: center; gap: 8px;
}
.section-heading::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── 2-col plot grid ──────────────────────────────────────────── */
.plot-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }

/* ── Cluster cards ────────────────────────────────────────────── */
.cluster-cards { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
.cluster-card {
  background: var(--card); border: 1px solid var(--border);
  border-left: 3px solid var(--accent2); border-radius: 8px;
  padding: 14px 17px; flex: 1; min-width: 200px; max-width: 340px;
}
.cc-header {
  font-family: var(--mono); font-weight: 700; font-size: 0.72rem;
  color: var(--accent2); margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 0.05em;
}
.cc-stats {
  display: flex; gap: 12px; flex-wrap: wrap;
  font-family: var(--mono); font-size: 0.62rem; color: var(--muted); margin-bottom: 9px;
}
.cc-val { color: var(--text-dim); }
.cc-markets { font-size: 0.7rem; color: #6880a0; padding-left: 14px; margin: 0; line-height: 1.6; }

/* ── Insights card ────────────────────────────────────────────── */
.insights-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 24px 32px; max-width: 820px;
  line-height: 1.75; font-size: 0.86rem; color: var(--text);
}
.insights-card h3 {
  font-family: var(--mono); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 5px; margin: 22px 0 10px 0;
}
.insights-card h3:first-child { margin-top: 0; }
.insights-card strong { color: #dce8f8; }
.insights-card li { margin-bottom: 5px; color: var(--text-dim); }
.insights-card p  { color: var(--text-dim); margin-bottom: 8px; }
.insights-header {
  font-family: var(--mono); font-size: 0.63rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em;
  margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.insights-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  animation: pdot 2s ease-in-out infinite; flex-shrink: 0;
}
@keyframes pdot {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,212,170,0.4); }
  50%      { box-shadow: 0 0 0 5px rgba(0,212,170,0); }
}

/* ── Modal ────────────────────────────────────────────────────── */
.modal-content { background: #0e1a2e !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
.modal-header  { border-bottom: 1px solid var(--border) !important; }
.modal-footer  { border-top:    1px solid var(--border) !important; }
.modal-title   { font-family: var(--mono) !important; font-size: 0.88rem !important; color: var(--accent) !important; letter-spacing: 0.05em !important; }
.btn-close     { filter: invert(1) opacity(0.6) !important; }
.modal-body h5 {
  font-family: var(--mono); font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent); margin: 18px 0 7px 0;
  border-bottom: 1px solid var(--border); padding-bottom: 4px;
}
.modal-body h5:first-child { margin-top: 0; }
.modal-body p, .modal-body li { font-size: 0.82rem; color: var(--text-dim); line-height: 1.65; }
.modal-body ul { padding-left: 18px; }
.modal-body li { margin-bottom: 5px; }
.modal-body code {
  background: var(--card); border: 1px solid var(--border);
  padding: 1px 6px; border-radius: 3px; font-family: var(--mono);
  font-size: 0.73rem; color: var(--accent);
}
.modal-btn-close {
  background: var(--accent) !important; color: #06101e !important;
  border: none !important; padding: 7px 20px !important; border-radius: 6px !important;
  font-family: var(--mono) !important; font-size: 0.73rem !important;
  font-weight: 700 !important; cursor: pointer !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
"""

# ══════════════════════════════════════════════════════════════════
#  PLOTLY THEME HELPERS
#  NOTE: plotly_dark() deliberately excludes title/xaxis/yaxis.
#  Passing those keys AND the same keys as explicit kwargs causes:
#    "update_layout() got multiple values for keyword argument"
#  Use apply_theme(fig, title=...) which sets everything safely.
# ══════════════════════════════════════════════════════════════════

_AXIS_STYLE  = dict(gridcolor="#1b2c44", zerolinecolor="#1b2c44",
                    tickfont=dict(size=9, color="#527090"), linecolor="#263550")
_TITLE_STYLE = dict(family="DM Sans, sans-serif", size=12, color="#c0d4ec")


def plotly_dark() -> dict:
    """Base theme dict — NO title / xaxis / yaxis keys."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", size=10, color="#527090"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#263550",
                    font=dict(size=9, color="#99b0cc")),
        margin=dict(l=50, r=20, t=44, b=38),
    )


def apply_theme(fig: go.Figure, title: str = "", height: int = None) -> go.Figure:
    """Apply dark theme to any Figure type without keyword conflicts."""
    kw: dict = {**plotly_dark()}
    if title:
        kw["title"] = dict(text=title, font=_TITLE_STYLE, x=0.01)
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


def plotly_html(fig: go.Figure, first: bool = False) -> HTML:
    js = "cdn" if first else False
    return HTML(fig.to_html(include_plotlyjs=js, full_html=False))


# ══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, delta: str = "", trend: str = "") -> ui.Tag:
    delta_cls  = "pos" if trend == "up" else "neg" if trend == "down" else ""
    delta_html = f'<div class="metric-delta {delta_cls}">{delta}</div>' if delta else ""
    card_cls   = f"metric-card m-{trend}" if trend else "metric-card"
    return ui.div(
        HTML(f'<div class="metric-label">{label}</div>'
             f'<div class="metric-value">{value}</div>{delta_html}'),
        class_=card_cls,
    )


def loading_div(msg: str = "Running analysis...") -> ui.Tag:
    return ui.div(
        HTML(f'<div class="spinner"></div><p class="loading-msg">{msg}</p>'),
        class_="loading-wrap",
    )


def empty_div(msg: str) -> ui.Tag:
    return ui.div(
        HTML(f'<div class="empty-icon">[ ]</div><p class="empty-msg">{msg}</p>'),
        class_="empty-state",
    )


def md_to_html(text: str) -> str:
    text = re.sub(r"^## (.+)$",          r"<h3>\1</h3>",          text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$",         r"<h4>\1</h4>",          text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*",      r"<strong>\1</strong>",   text)
    text = re.sub(r"\*(.+?)\*",          r"<em>\1</em>",           text)
    text = re.sub(r"^[\-\*\d+\.] (.+)$", r"<li>\1</li>",          text, flags=re.MULTILINE)
    text = text.replace("\n\n", "</p><p>")
    return f"<p>{text}</p>"


# ══════════════════════════════════════════════════════════════════
#  HELP MODAL BODY
# ══════════════════════════════════════════════════════════════════

HELP_HTML = """
<h5>What is PredictIQ?</h5>
<p>A multi-agent intelligence platform that connects live prediction market data
to macroeconomic indicators, then synthesises findings with a large language model.
The full pipeline runs in the background each time you click <strong>RUN ANALYSIS</strong>.</p>

<h5>Supervisor / Agent Architecture</h5>
<ul>
  <li><strong>Supervisor</strong> — LangGraph orchestrator. Reads pipeline state and routes to
      whichever agent has not yet completed its stage.</li>
  <li><strong>Polymarket Agent</strong> — Fetches up to 200 live prediction markets from the
      free Gamma API (no key required).</li>
  <li><strong>FRED Agent</strong> — Pulls economic time-series from the St. Louis Fed for each
      checked indicator. Requires <code>FRED_API_KEY</code>.</li>
  <li><strong>Analysis Agent</strong> — Runs a silhouette sweep to find the optimal K, then
      K-Means clustering and PCA for 2-D visualisation.</li>
  <li><strong>Insights Agent</strong> — Sends all data to Groq LLaMA-3.3-70B and returns a
      structured intelligence brief tailored to your research query.</li>
</ul>

<h5>Research Query</h5>
<p>The text area in the sidebar is fully editable — click into it and type any question
or research focus before running. The Insights Agent uses it to shape the intelligence
brief. Examples:</p>
<ul>
  <li><em>How are markets pricing Fed rate cuts?</em></li>
  <li><em>What do crypto prediction markets imply about Bitcoin in 2025?</em></li>
  <li><em>Where do political markets diverge from economic consensus?</em></li>
</ul>

<h5>FRED Indicators</h5>
<p>Check or uncheck any combination before running. Requires
<code>FRED_API_KEY</code> in your environment.
Free key: <strong>fred.stlouisfed.org/docs/api/api_key.html</strong></p>

<h5>Required Environment Variables</h5>
<ul>
  <li><code>GROQ_API_KEY</code> — LLM insights (console.groq.com, free tier available)</li>
  <li><code>FRED_API_KEY</code> — Economic data (fred.stlouisfed.org, free)</li>
  <li>Polymarket — no key needed</li>
</ul>

<h5>Dashboard Tabs</h5>
<ul>
  <li><strong>Overview</strong> — summary metrics, probability histogram, top-20 market table</li>
  <li><strong>Markets</strong> — volume chart, scatter plot, category breakdown</li>
  <li><strong>Indicators</strong> — FRED metric cards and time-series charts</li>
  <li><strong>Clusters</strong> — PCA scatter, silhouette sweep, cluster comparison and detail cards</li>
  <li><strong>AI Insights</strong> — Groq-generated intelligence brief</li>
</ul>
"""

# ══════════════════════════════════════════════════════════════════
#  SHINY UI
# ══════════════════════════════════════════════════════════════════

app_ui = ui.page_fluid(
    ui.tags.style(CSS),

    # ── Header ──────────────────────────────────────────────────
    ui.div(
        ui.div(
            ui.div(
                HTML('<div class="brand-name">&#x2B21; PREDICTIQ</div>'
                     '<div class="brand-sub">Agentic Market Intelligence &middot; LangGraph Supervisor Architecture</div>'),
            ),
            ui.div(
                HTML('<div class="header-right">'
                     '<span class="hbadge hbadge-green">POLYMARKET</span>'
                     '<span class="hbadge hbadge-green">FRED</span>'
                     '<span class="hbadge hbadge-purple">GROQ LLAMA-3</span>'
                     '<span class="hbadge hbadge-yellow">K-MEANS + PCA</span>'
                     '</div>'),
                ui.input_action_button("help_btn", "?", class_="help-btn"),
                style="display:flex; align-items:center; gap:10px;",
            ),
            class_="header-inner",
        ),
        class_="app-header",
    ),

    ui.layout_sidebar(
        ui.sidebar(
            # Research query
            HTML('<span class="sidebar-label">Research Query</span>'
                 '<p class="query-hint">Edit below &mdash; shapes the AI brief</p>'),
            ui.input_text_area(
                "query", None,
                value="How do prediction markets reflect current macroeconomic uncertainty and Fed policy expectations?",
                rows=4,
            ),

            HTML('<hr class="sidebar-hr">'),
            HTML('<span class="sidebar-label">Market Category</span>'),
            ui.input_select("category", None, choices=POLY_CATS, selected=""),

            HTML('<hr class="sidebar-hr">'),
            HTML('<span class="sidebar-label">FRED Indicators</span>'),
            ui.input_checkbox_group(
                "fred_series", None,
                choices=FRED_SERIES,
                selected=["UNRATE", "CPIAUCSL", "FEDFUNDS", "VIXCLS", "UMCSENT", "T10Y2Y"],
            ),

            HTML('<hr class="sidebar-hr">'),
            ui.input_action_button("run_btn", "RUN ANALYSIS", class_="run-btn"),

            HTML('<hr class="sidebar-hr">'),
            HTML('<span class="sidebar-label">Agent Log</span>'),
            ui.output_ui("agent_log"),

            width=312,
            bg="#0b1425",
        ),

        ui.navset_tab(
            ui.nav_panel("Overview",    ui.div(ui.output_ui("tab_overview"),   style="padding:16px 22px;")),
            ui.nav_panel("Markets",     ui.div(ui.output_ui("tab_markets"),    style="padding:16px 22px;")),
            ui.nav_panel("Indicators",  ui.div(ui.output_ui("tab_indicators"), style="padding:16px 22px;")),
            ui.nav_panel("Clusters",    ui.div(ui.output_ui("tab_clusters"),   style="padding:16px 22px;")),
            ui.nav_panel("AI Insights", ui.div(ui.output_ui("tab_insights"),   style="padding:16px 22px;")),
            id="main_tabs",
        ),
    ),

    title="PredictIQ",
)


# ══════════════════════════════════════════════════════════════════
#  SHINY SERVER
# ══════════════════════════════════════════════════════════════════

def server(input, output, session):

    result_rv = reactive.Value(None)

    # ── Help modal ─────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.help_btn)
    def _show_help():
        ui.modal_show(ui.modal(
            HTML(HELP_HTML),
            title="How to use PredictIQ",
            easy_close=True,
            footer=ui.modal_button("Close", class_="modal-btn-close"),
            size="l",
        ))

    # ── Background pipeline ────────────────────────────────────
    @reactive.extended_task
    async def analysis_task(q: str, cat: str, series: List[str]):
        return await asyncio.to_thread(run_graph, q, cat, series)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _start():
        result_rv.set(None)
        analysis_task(input.query(), input.category(), list(input.fred_series()))

    @reactive.effect
    def _collect():
        if analysis_task.status() == "success":
            result_rv.set(analysis_task.result())

    # ── Agent log ──────────────────────────────────────────────
    @output
    @render.ui
    def agent_log():
        if analysis_task.status() == "running":
            return HTML('<div class="log-box"><div class="log-running">&#9679; RUNNING...</div></div>')
        s = result_rv.get()
        if s is None:
            return HTML('<div class="log-box"><div class="log-empty">Awaiting run...</div></div>')
        lines = s.get("log", [])[-32:]
        items = [
            f'<div class="log-line {"last" if i == len(lines)-1 else "ok"}">{ln}</div>'
            for i, ln in enumerate(lines)
        ]
        return HTML(f'<div class="log-box">{"".join(items)}</div>')

    # ── TAB: Overview ──────────────────────────────────────────
    @output
    @render.ui
    def tab_overview():
        if analysis_task.status() == "running":
            return loading_div("Executing supervisor pipeline...")
        s = result_rv.get()
        if s is None:
            return HTML("""
            <div class="welcome-wrap">
              <div class="welcome-title">&#x2B21; PREDICTIQ</div>
              <p class="welcome-sub">An agentic intelligence platform connecting live prediction
              market data to macroeconomic indicators, synthesised into an AI brief.<br><br>
              Click <strong style="color:#00d4aa">RUN ANALYSIS</strong> in the sidebar to start,
              or <strong style="color:#00d4aa">?</strong> in the header for instructions.</p>
              <div class="tech-badges">
                <span class="tbadge">LANGGRAPH</span><span class="tbadge">LANGCHAIN</span>
                <span class="tbadge">GROQ LLAMA-3</span><span class="tbadge">POLYMARKET API</span>
                <span class="tbadge">FRED API</span><span class="tbadge">K-MEANS</span>
                <span class="tbadge">PCA</span><span class="tbadge">SHINY</span>
              </div>
              <div class="pipeline-vis">
                <div class="pipe-node sup">SUPERVISOR<br><small>orchestrator</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node data">POLYMARKET<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node data">FRED<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node ml">ANALYSIS<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node llm">INSIGHTS<br><small>agent</small></div>
              </div>
            </div>""")

        cr  = s.get("cluster_results", {})
        fd  = s.get("fred_data", {})
        md  = s.get("market_data", [])
        sil = cr.get("sil_score")

        metrics = ui.div(
            metric_card("Markets",    f"{cr.get('total', len(md)):,}"),
            metric_card("Volume",     f"${cr.get('total_vol', 0)/1e6:.1f}M"),
            metric_card("Avg YES",    f"{cr.get('avg_prob', 0.5)*100:.1f}%"),
            metric_card("Clusters",   str(cr.get("n_clusters", "—"))),
            metric_card("Indicators", str(len(fd))),
            metric_card("Silhouette", f"{sil:.3f}" if sil else "—"),
            class_="metrics-row",
        )

        hist_html = ui.div()
        if md:
            dff = pd.DataFrame(md)
            dff = dff[dff["volume"] > 0]
            fig = px.histogram(dff, x="yes_prob", nbins=50,
                               color_discrete_sequence=[ACCENT],
                               labels={"yes_prob": "YES Probability", "count": "Markets"},
                               height=260)
            fig.update_traces(marker_line_width=0)
            fig.add_vline(x=0.5, line_dash="dash", line_color="#374151",
                          annotation_text="50/50", annotation_font_size=9,
                          annotation_font_color="#527090")
            apply_theme(fig, title="YES Probability Distribution")
            hist_html = plotly_html(fig, first=True)

        top  = sorted(md, key=lambda x: x["volume"], reverse=True)[:20]
        rows = "".join(
            f'<tr>'
            f'<td class="q-cell">{m["question"][:76]}{"..." if len(m["question"])>76 else ""}</td>'
            f'<td><span class="prob-badge {"prob-high" if m["yes_prob"]>0.65 else "prob-low" if m["yes_prob"]<0.35 else "prob-mid"}">{m["yes_prob"]*100:.1f}%</span></td>'
            f'<td style="font-family:var(--mono);font-size:0.68rem;">${m["volume"]:,.0f}</td>'
            f'<td style="font-family:var(--mono);font-size:0.64rem;color:var(--muted);">{m.get("category","")}</td>'
            f'<td style="font-family:var(--mono);font-size:0.64rem;color:var(--muted);">{m.get("end_date","")}</td>'
            f'</tr>'
            for m in top
        )
        return ui.div(
            metrics, hist_html,
            HTML('<div class="section-heading">Top Markets by Volume</div>'),
            HTML(f'<div class="table-wrap"><table class="data-table">'
                 f'<thead><tr><th>Question</th><th>YES%</th><th>Volume</th>'
                 f'<th>Category</th><th>Ends</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table></div>'),
        )

    # ── TAB: Markets ───────────────────────────────────────────
    @output
    @render.ui
    def tab_markets():
        if analysis_task.status() == "running":
            return loading_div("Fetching Polymarket data...")
        s = result_rv.get()
        if not s or not s.get("market_data"):
            return empty_div("No market data — run analysis first")

        md = s["market_data"]
        df = pd.DataFrame(md)
        df = df[df["volume"] > 0].copy()

        top20         = df.nlargest(20, "volume").copy()
        top20["qlbl"] = top20["question"].str[:65] + "..."

        # Top-20 bar — apply_theme handles title, then override yaxis separately
        fig1 = px.bar(
            top20, x="volume", y="qlbl", orientation="h",
            color="yes_prob",
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
            range_color=[0, 1],
            labels={"volume": "Volume ($)", "qlbl": "", "yes_prob": "YES"},
            height=600,
        )
        apply_theme(fig1, title="Top 20 Markets by Volume  (colour = YES probability)")
        fig1.update_yaxes(autorange="reversed", tickfont=dict(size=8))
        fig1.update_layout(
            coloraxis_colorbar=dict(title="YES", tickformat=".0%",
                                    tickfont=dict(size=8)),
        )

        # Scatter — apply_theme only, no xaxis/yaxis conflict
        fig2 = px.scatter(
            df.head(150), x="yes_prob", y="volume",
            color="category", size="liquidity",
            hover_name="question", log_y=True,
            labels={"yes_prob": "YES Probability", "volume": "Volume ($, log)"},
            height=420,
        )
        apply_theme(fig2, title="Probability vs Volume  (bubble size = liquidity)")
        fig2.update_traces(marker=dict(opacity=0.75))

        # Category bars
        cat_df = (df.groupby("category")
                  .agg(n=("id", "count"), vol=("volume", "sum"))
                  .reset_index().sort_values("vol", ascending=False).head(10))
        fig3 = go.Figure(go.Bar(
            x=cat_df["category"], y=cat_df["vol"],
            marker_color=ACCENT,
            text=cat_df["n"].astype(str),
            textposition="outside", textfont=dict(size=8),
        ))
        apply_theme(fig3, title="Total Volume by Category", height=310)
        fig3.update_xaxes(title="")
        fig3.update_yaxes(title="Total Volume ($)")

        return ui.div(plotly_html(fig1, first=True), plotly_html(fig2), plotly_html(fig3))

    # ── TAB: Indicators ────────────────────────────────────────
    @output
    @render.ui
    def tab_indicators():
        if analysis_task.status() == "running":
            return loading_div("Fetching FRED economic data...")
        s = result_rv.get()
        if not s or not s.get("fred_data"):
            return empty_div("No indicator data — run analysis first")

        fd    = s["fred_data"]
        items = list(fd.items())

        cards = [
            metric_card(
                d["name"][:26],
                f"{d['latest']:.2f}",
                delta=f"{d.get('pct_chg', 0):+.1f}% YoY  ({d['latest_date']})",
                trend="up" if d.get("pct_chg", 0) > 0 else "down",
            )
            for _, d in items
        ]

        COLORS = [ACCENT, "#7c6af7", "#f59e0b", "#ec4899",
                  "#22c55e", "#38bdf8", "#fb923c", "#a78bfa",
                  "#34d399", "#f472b6"]

        plot_rows = []
        first     = True
        for i in range(0, len(items), 2):
            row_figs = []
            for j, (sid, d) in enumerate(items[i:i+2]):
                col    = COLORS[(i + j) % len(COLORS)]
                r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                fill_c = f"rgba({r},{g},{b},0.07)"
                fig = go.Figure(go.Scatter(
                    x=d["dates"], y=d["values"],
                    mode="lines", line=dict(color=col, width=1.8),
                    fill="tozeroy", fillcolor=fill_c, name=sid,
                ))
                apply_theme(
                    fig,
                    title=f"{d['name']}  {d['latest']:.2f}  ({d.get('pct_chg', 0):+.1f}% YoY)",
                    height=235,
                )
                fig.update_layout(showlegend=False)
                row_figs.append(plotly_html(fig, first=first))
                first = False
            plot_rows.append(ui.div(*row_figs, class_="plot-row"))

        return ui.div(ui.div(*cards, class_="metrics-row"), ui.br(), *plot_rows)

    # ── TAB: Clusters ──────────────────────────────────────────
    @output
    @render.ui
    def tab_clusters():
        if analysis_task.status() == "running":
            return loading_div("Running K-Means clustering & PCA...")
        s = result_rv.get()
        if not s or not s.get("cluster_results"):
            return empty_div("No cluster results — run analysis first")

        cr  = s["cluster_results"]
        rec = cr.get("records", [])
        if not rec:
            return ui.p("Insufficient market data for clustering.")

        df = pd.DataFrame(rec)
        df["cluster_str"]   = df["cluster"].astype(str)
        lmap                = {str(su["cluster"]): su["label"] for su in cr.get("summaries", [])}
        df["cluster_label"] = df["cluster_str"].map(lmap).fillna(df["cluster_str"])
        PAL = ["#00d4aa", "#7c6af7", "#f59e0b", "#ec4899", "#38bdf8", "#fb923c", "#a78bfa"]
        pv  = cr.get("pca_var", [0, 0])

        # PCA scatter
        fig1 = px.scatter(
            df, x="pc1", y="pc2", color="cluster_label",
            hover_name="question",
            hover_data={"yes_prob": ":.1%", "volume": ":,.0f",
                        "cluster_label": False, "cluster_str": False,
                        "pc1": False, "pc2": False},
            labels={"pc1": f"PC1 ({pv[0]*100:.1f}% var)",
                    "pc2": f"PC2 ({pv[1]*100:.1f}% var)",
                    "cluster_label": "Cluster"},
            color_discrete_sequence=PAL, height=470,
        )
        apply_theme(fig1,
                    title=f"PCA Cluster Map — {cr['n_clusters']} clusters  "
                          f"(silhouette = {cr['sil_score']:.3f})")
        fig1.update_traces(marker=dict(size=7, opacity=0.8))

        # Silhouette sweep
        sil  = cr.get("sil_map", {})
        fig2 = ui.div()
        if sil:
            ks   = sorted(sil.keys())
            bcol = [ACCENT if k == cr["n_clusters"] else "#1b2c44" for k in ks]
            lcol = [ACCENT if k == cr["n_clusters"] else "#263550" for k in ks]
            f2   = go.Figure(go.Bar(
                x=ks, y=[sil[k] for k in ks],
                marker_color=bcol, marker_line_color=lcol, marker_line_width=2,
                text=[f"{sil[k]:.3f}" for k in ks], textposition="outside",
                textfont=dict(size=8, color="#6880a0"),
            ))
            apply_theme(f2,
                        title="Silhouette Score vs K  (selected K highlighted)",
                        height=265)
            f2.update_xaxes(title="K (number of clusters)")
            f2.update_yaxes(title="Silhouette Score")
            fig2 = plotly_html(f2)

        # Cluster comparison subplots
        sums   = cr.get("summaries", [])
        labels = [su["label"]        for su in sums]
        counts = [su["count"]        for su in sums]
        probs  = [su["avg_prob"]*100 for su in sums]
        vols   = [su["avg_volume"]   for su in sums]

        fig3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Market Count", "Avg YES Prob (%)", "Avg Volume ($)"],
            horizontal_spacing=0.07,
        )
        for ci, y in enumerate([counts, probs, vols], 1):
            fig3.add_trace(
                go.Bar(x=labels, y=y, marker_color=PAL[:len(sums)],
                       showlegend=False,
                       text=[f"{v:.0f}" for v in y], textposition="outside",
                       textfont=dict(size=7)),
                row=1, col=ci,
            )
        # apply_theme on a make_subplots figure — safe because no title/xaxis/yaxis in plotly_dark()
        apply_theme(fig3, height=305)
        for ann in fig3.layout.annotations:
            ann.update(font=dict(size=10, color="#99b0cc"))
        fig3.update_xaxes(tickfont=dict(size=7))

        # Cluster detail cards
        cards = []
        for su in sums:
            mkt_li = "".join(
                f'<li>{q[:66]}{"..." if len(q)>66 else ""}</li>'
                for q in su["top_markets"]
            )
            cards.append(HTML(
                f'<div class="cluster-card">'
                f'<div class="cc-header">{su["label"]}</div>'
                f'<div class="cc-stats">'
                f'<span>YES <span class="cc-val">{su["avg_prob"]*100:.1f}%</span></span>'
                f'<span>Vol <span class="cc-val">${su["avg_volume"]:,.0f}</span></span>'
                f'<span>N <span class="cc-val">{su["count"]}</span></span>'
                f'<span>Cert <span class="cc-val">{su["avg_certainty"]:.2f}</span></span>'
                f'</div><ul class="cc-markets">{mkt_li}</ul></div>'
            ))

        return ui.div(
            plotly_html(fig1, first=True), fig2, plotly_html(fig3),
            HTML('<div class="section-heading">Cluster Details</div>'),
            ui.div(*cards, class_="cluster-cards"),
        )

    # ── TAB: AI Insights ───────────────────────────────────────
    @output
    @render.ui
    def tab_insights():
        if analysis_task.status() == "running":
            return loading_div(f"Generating intelligence brief via Groq {MODEL}...")
        s = result_rv.get()
        if not s or not s.get("insights"):
            return empty_div("No insights yet — run analysis first")
        body = md_to_html(s["insights"])
        now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
        return HTML(
            f'<div class="insights-card">'
            f'<div class="insights-header">'
            f'<span class="insights-dot"></span>'
            f'GROQ &middot; {MODEL} &middot; Generated {now}'
            f'</div>{body}</div>'
        )


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

app = App(app_ui, server)