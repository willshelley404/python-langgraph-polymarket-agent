#!/usr/bin/env python3
"""
PredictIQ — Agentic Prediction Market Intelligence Platform
═══════════════════════════════════════════════════════════
Orchestration : LangGraph supervisor + 5 specialist agents
LLM           : Groq (llama-3.3-70b-versatile)
Market data   : Polymarket Gamma API (free, no key)
Economic data : US Census ACS 5-Year Estimates
Geography     : State-level market extraction + ACS overlay map
ML            : K-Means clustering · PCA · Silhouette sweep
UI            : Shiny for Python
"""

import os, json, re, asyncio
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional

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

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
MODEL          = "llama-3.3-70b-versatile"
ACCENT         = "#00d4aa"
ACS_YEAR       = "2022"          # most recent complete 5-year ACS

# ── ACS variables to pull at state level ──────────────────────
ACS_PULL_VARS: Dict[str, str] = {
    "B19013_001E": "med_income",     # Median household income ($)
    "B17001_002E": "poverty_count",  # Pop. below poverty level
    "B01003_001E": "population",     # Total population
    "B23025_005E": "unemployed",     # Civilian unemployed
    "B23025_003E": "labor_force",    # Civilian labor force
    "B25064_001E": "med_rent",       # Median gross rent ($)
    "B01002_001E": "median_age",     # Median age
    "B15003_022E": "bach_degree",    # Bachelor's degree count
    "B15003_001E": "educ_universe",  # Educational attainment universe
}

# Human-readable names for the UI checkboxes
ACS_METRIC_LABELS: Dict[str, str] = {
    "med_income":    "Median Household Income ($)",
    "poverty_rate":  "Poverty Rate (%)",
    "unemp_rate":    "Unemployment Rate (%)",
    "med_rent":      "Median Gross Rent ($)",
    "median_age":    "Median Age",
    "bach_rate":     "Bachelor's Degree Rate (%)",
    "population":    "Total Population",
}

POLY_CATS: Dict[str, str] = {
    "":            "All Categories",
    "politics":    "Politics",
    "crypto":      "Crypto",
    "economics":   "Economics",
    "sports":      "Sports",
    "science":     "Science & Tech",
    "technology":  "Technology",
    "pop-culture": "Pop Culture",
}

# ── US State reference data ────────────────────────────────────
US_STATE_NAMES: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}
# Reverse: abbrev -> full name
US_ABBREV: Dict[str, str] = {v: k for k, v in US_STATE_NAMES.items()}

# Abbreviations to also catch in raw text
_ABBREV_PATTERN = (
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|"
    r"UT|VT|VA|WA|WV|WI|WY|DC)\b"
)
_NAME_PATTERN = r"\b(" + "|".join(re.escape(n) for n in US_STATE_NAMES) + r")\b"


def extract_state_mentions(text: str) -> List[str]:
    """Return list of 2-letter state abbreviations mentioned in text."""
    found = set()
    for m in re.finditer(_NAME_PATTERN, text, re.IGNORECASE):
        name = m.group(0).title()
        # handle "New York", "West Virginia" etc. (multi-word)
        for full, abbr in US_STATE_NAMES.items():
            if full.lower() in text.lower():
                found.add(abbr)
    for m in re.finditer(_ABBREV_PATTERN, text):
        found.add(m.group(1))
    return list(found)


# ══════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    query:           str
    category:        str
    acs_metrics:     List[str]
    market_data:     Any           # None = not yet fetched
    acs_data:        Dict[str, Any]
    geo_data:        Dict[str, Any]
    cluster_results: Dict[str, Any]
    insights:        str
    log:             List[str]
    next:            str


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════
#  AGENT NODES
# ══════════════════════════════════════════════════════════════════

# ── Supervisor ───────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] SUPERVISOR  evaluating pipeline state...")

    nxt = (
        "polymarket" if state.get("market_data") is None
        else "acs"       if not state.get("acs_data")
        else "geography" if not state.get("geo_data")
        else "analysis"  if not state.get("cluster_results")
        else "insights"  if not state.get("insights")
        else "END"
    )
    log.append(f"[{_ts()}]             routing -> {nxt}")
    return {"next": nxt, "log": log}


# ── Polymarket Agent ─────────────────────────────────────────────

def polymarket_agent(state: AgentState) -> dict:
    """
    Fetches markets from Polymarket Gamma API.
    Category filtering: (1) passes tag_slug to API; (2) applies client-side
    fallback checking both the 'category' field AND the 'tags[*].slug' array
    because the API's category field is often null.
    """
    log  = list(state.get("log", []))
    cat  = state.get("category", "")
    log.append(f"[{_ts()}] POLYMARKET  fetching markets (category={cat or 'all'})...")

    try:
        params: Dict[str, Any] = {
            "limit":     500,
            "active":    "true",
            "closed":    "false",
            "order":     "volume",
            "ascending": "false",
        }
        if cat:
            params["tag_slug"] = cat

        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params=params, timeout=30,
        )
        r.raise_for_status()
        raw_markets = r.json()
        log.append(f"[{_ts()}] POLYMARKET  {len(raw_markets)} raw records received")

        out: List[Dict] = []
        for m in raw_markets:
            try:
                prices   = m.get("outcomePrices", [])
                prices   = json.loads(prices) if isinstance(prices, str) else (prices or [])
                outcomes = m.get("outcomes", [])
                outcomes = json.loads(outcomes) if isinstance(outcomes, str) else (outcomes or [])
                yes_p    = float(prices[0]) if prices else 0.5

                # ── Resolve category from multiple possible fields ──
                # 1. Direct 'category' field (often null on Gamma API)
                mcat = (m.get("category") or "").strip()
                # 2. 'tags' array — each tag is {id, label, slug, ...}
                raw_tags = m.get("tags", []) or []
                if isinstance(raw_tags, str):
                    try:
                        raw_tags = json.loads(raw_tags)
                    except Exception:
                        raw_tags = []
                tag_slugs  = [str(t.get("slug",  "")).lower() for t in raw_tags if isinstance(t, dict)]
                tag_labels = [str(t.get("label", "")).lower() for t in raw_tags if isinstance(t, dict)]
                # Use first tag label as category if category field is empty
                if not mcat and tag_labels:
                    mcat = tag_labels[0].title()

                out.append({
                    "id":         m.get("id", ""),
                    "question":   m.get("question", ""),
                    "yes_prob":   round(yes_p, 4),
                    "no_prob":    round(1 - yes_p, 4),
                    "volume":     float(m.get("volume",    0) or 0),
                    "liquidity":  float(m.get("liquidity", 0) or 0),
                    "category":   mcat or "General",
                    "tag_slugs":  tag_slugs,
                    "end_date":   (m.get("endDateIso", "") or "")[:10],
                })
            except Exception:
                continue

        # ── Client-side category filter (robust fallback) ──────────
        if cat and out:
            cat_lower = cat.lower()
            filtered  = [
                m for m in out
                if cat_lower in m["category"].lower()
                or cat_lower in m["tag_slugs"]
                or any(cat_lower in slug for slug in m["tag_slugs"])
            ]
            if filtered:
                log.append(f"[{_ts()}] POLYMARKET  category filter '{cat}' -> {len(filtered)} markets")
                out = filtered
            else:
                log.append(f"[{_ts()}] POLYMARKET  category filter '{cat}' matched 0; showing all")

        # Trim to top 200 by volume after filtering
        out = sorted(out, key=lambda x: x["volume"], reverse=True)[:200]
        log.append(f"[{_ts()}] POLYMARKET  {len(out)} markets after filter/sort")
        return {"market_data": out, "log": log}

    except Exception as e:
        log.append(f"[{_ts()}] POLYMARKET  error: {str(e)[:120]}")
        return {"market_data": [], "log": log}


# ── ACS Agent ────────────────────────────────────────────────────

def acs_agent(state: AgentState) -> dict:
    """
    Pulls ACS 5-Year Estimates for all 50 states + DC from the Census Bureau API.
    Derives poverty rate, unemployment rate, and bachelor's degree attainment rate.
    Falls back to an empty dict (gracefully) if the API key is missing or request fails.
    """
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] ACS         pulling Census ACS {ACS_YEAR} 5-year estimates...")

    api_key = CENSUS_API_KEY
    vars_to_get = ",".join(ACS_PULL_VARS.keys())
    url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

    params: Dict[str, Any] = {
        "get":  f"{vars_to_get},NAME",
        "for":  "state:*",
    }
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        rows  = r.json()
        hdrs  = rows[0]
        data_rows = rows[1:]
    except Exception as e:
        log.append(f"[{_ts()}] ACS         API error: {str(e)[:100]}")
        return {"acs_data": {}, "log": log}

    records: List[Dict] = []
    for row in data_rows:
        rec = dict(zip(hdrs, row))
        # Convert numeric strings
        numeric_keys = list(ACS_PULL_VARS.keys())
        for k in numeric_keys:
            try:
                rec[k] = float(rec[k]) if rec[k] not in (None, "", "-666666666") else np.nan
            except (ValueError, TypeError):
                rec[k] = np.nan

        # Build friendly record
        name_parts = rec.get("NAME", "").replace(", United States", "")
        # Match state abbreviation
        abbr = US_STATE_NAMES.get(name_parts, "??")

        pop     = rec.get("B01003_001E", np.nan)
        pov_cnt = rec.get("B17001_002E", np.nan)
        lf      = rec.get("B23025_003E", np.nan)
        unemp   = rec.get("B23025_005E", np.nan)
        educ_u  = rec.get("B15003_001E", np.nan)
        bach    = rec.get("B15003_022E", np.nan)

        records.append({
            "state_name":   name_parts,
            "abbr":         abbr,
            "fips":         rec.get("state", ""),
            "population":   int(pop)  if not np.isnan(pop)  else None,
            "med_income":   int(rec["B19013_001E"]) if not np.isnan(rec.get("B19013_001E", np.nan)) else None,
            "poverty_rate": round(pov_cnt / pop * 100, 2)  if pop and not np.isnan(pov_cnt) else None,
            "unemp_rate":   round(unemp  / lf   * 100, 2)  if lf  and not np.isnan(unemp)  else None,
            "med_rent":     int(rec["B25064_001E"]) if not np.isnan(rec.get("B25064_001E", np.nan)) else None,
            "median_age":   float(rec["B01002_001E"]) if not np.isnan(rec.get("B01002_001E", np.nan)) else None,
            "bach_rate":    round(bach / educ_u * 100, 2) if educ_u and not np.isnan(bach) else None,
        })

    log.append(f"[{_ts()}] ACS         {len(records)} state records loaded")
    return {"acs_data": {"records": records}, "log": log}


# ── Geography Agent ───────────────────────────────────────────────

def geography_agent(state: AgentState) -> dict:
    """
    Extracts US state references from prediction market questions.
    Builds per-state market counts, volume totals, and avg YES probability.
    Note: Polymarket / Kalshi do not expose bettor location (pseudonymous).
    This maps what geographic OUTCOMES markets are betting on, not bettor origin.
    """
    log     = list(state.get("log", []))
    markets = state.get("market_data", [])
    log.append(f"[{_ts()}] GEOGRAPHY   extracting state references from {len(markets)} markets...")

    state_stats: Dict[str, Dict] = {
        abbr: {"count": 0, "volume": 0.0, "yes_probs": [], "questions": []}
        for abbr in US_ABBREV
    }

    for m in markets:
        q      = m.get("question", "")
        states = extract_state_mentions(q)
        for abbr in states:
            if abbr in state_stats:
                state_stats[abbr]["count"]    += 1
                state_stats[abbr]["volume"]   += m.get("volume", 0)
                state_stats[abbr]["yes_probs"].append(m.get("yes_prob", 0.5))
                state_stats[abbr]["questions"].append(q)

    # Flatten to records
    geo_records: List[Dict] = []
    for abbr, s in state_stats.items():
        if s["count"] > 0:
            geo_records.append({
                "abbr":       abbr,
                "state_name": US_ABBREV.get(abbr, abbr),
                "count":      s["count"],
                "volume":     round(s["volume"], 0),
                "avg_prob":   round(np.mean(s["yes_probs"]), 3) if s["yes_probs"] else 0.5,
                "top_questions": s["questions"][:3],
            })

    geo_records.sort(key=lambda x: x["volume"], reverse=True)
    top_states = [r["abbr"] for r in geo_records[:5]]
    log.append(
        f"[{_ts()}] GEOGRAPHY   {len(geo_records)} states referenced. "
        f"Top: {', '.join(top_states)}"
    )
    return {"geo_data": {"records": geo_records}, "log": log}


# ── Analysis Agent ───────────────────────────────────────────────

def analysis_agent(state: AgentState) -> dict:
    """K-Means + PCA on prediction market data."""
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] ANALYSIS    feature engineering & clustering...")

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
    }
    log.append(f"[{_ts()}] ANALYSIS    {best_k} clusters (sil={best_s:.3f}), {len(df)} mkts")
    return {"cluster_results": result, "log": log}


# ── Insights Agent ───────────────────────────────────────────────

def insights_agent(state: AgentState) -> dict:
    """Synthesises all data into an intelligence brief via Groq LLM."""
    log = list(state.get("log", []))
    log.append(f"[{_ts()}] INSIGHTS    querying Groq {MODEL}...")

    markets  = state.get("market_data", [])
    acs_d    = state.get("acs_data", {})
    geo_d    = state.get("geo_data", {})
    cr       = state.get("cluster_results", {})
    query    = state.get("query", "general market intelligence")

    top_m = sorted(markets, key=lambda x: x["volume"], reverse=True)[:12]
    m_txt = "\n".join(
        f"  {m['question'][:100]}: {m['yes_prob']*100:.1f}% YES (${m['volume']:,.0f} vol)"
        for m in top_m
    )

    # ACS snapshot
    acs_recs = acs_d.get("records", [])
    acs_snap = sorted(acs_recs, key=lambda x: x.get("med_income") or 0, reverse=True)
    a_txt    = "\n".join(
        f"  {r['state_name']} ({r['abbr']}): income=${r['med_income']:,}, "
        f"poverty={r['poverty_rate']}%, unemp={r['unemp_rate']}%"
        for r in acs_snap[:10]
        if r.get("med_income")
    ) or "  No ACS data available."

    # Geo snapshot
    geo_recs = geo_d.get("records", [])[:8]
    g_txt    = "\n".join(
        f"  {r['state_name']} ({r['abbr']}): {r['count']} markets, ${r['volume']:,.0f} vol, "
        f"{r['avg_prob']*100:.0f}% avg YES"
        for r in geo_recs
    ) or "  No geographic data extracted."

    # Cluster snapshot
    c_txt = "".join(
        f"  [{s['label']}] {s['count']} markets | {s['avg_prob']*100:.0f}% avg YES | "
        f"${s['avg_volume']:,.0f} avg vol\n"
        f"    -> {'; '.join(s['top_markets'][:2])}\n"
        for s in cr.get("summaries", [])
    )

    prompt = f"""You are a senior quantitative analyst specialising in prediction markets, demographics, and economic geography.

USER RESEARCH FOCUS: {query}

=== POLYMARKET DATA (top markets by volume) ===
{m_txt}

=== US CENSUS ACS (top states by median income) ===
{a_txt}

=== GEOGRAPHIC MARKET FOCUS (states referenced in market questions) ===
{g_txt}

=== ML CLUSTERING ({cr.get('n_clusters', 0)} clusters, silhouette={cr.get('sil_score', 0):.3f}) ===
{c_txt}

Write a structured intelligence brief with these exact headers:

## Market Sentiment
What collective wisdom do prediction markets reveal? Highlight striking probabilities and consensus signals.

## Geographic & Demographic Context
Which US states are most prominent in market questions? How do ACS economic indicators for those states (income, poverty, unemployment) align with or explain market outcomes? Are poorer or wealthier states betting more on certain outcome types?

## Cluster Analysis
What market structure patterns emerge from the ML clustering? Which cluster is most actionable and why?

## Key Signals & Risks
Top 3 specific data-driven signals or risks an analyst should act on.

## Divergences & Anomalies
Where do prediction markets and Census economic data tell conflicting stories? Any informational edges?

Rules: cite actual numbers, be analytical and actionable, ~450 words total."""

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
    for name, fn in [
        ("supervisor",  supervisor_node),
        ("polymarket",  polymarket_agent),
        ("acs",         acs_agent),
        ("geography",   geography_agent),
        ("analysis",    analysis_agent),
        ("insights",    insights_agent),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "polymarket": "polymarket",
            "acs":        "acs",
            "geography":  "geography",
            "analysis":   "analysis",
            "insights":   "insights",
            "END":        END,
        },
    )
    for node in ["polymarket", "acs", "geography", "analysis", "insights"]:
        g.add_edge(node, "supervisor")
    return g.compile()


def run_graph(query: str, category: str, acs_metrics: List[str]) -> AgentState:
    graph = build_graph()
    return graph.invoke({
        "messages":        [],
        "query":           query,
        "category":        category,
        "acs_metrics":     acs_metrics,
        "market_data":     None,
        "acs_data":        {},
        "geo_data":        {},
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
  --bg:      #0a0e1a;
  --surface: #131c2e;
  --card:    #192438;
  --border:  #263550;
  --accent:  #00d4aa;
  --a2:      #7c6af7;
  --warn:    #f59e0b;
  --ok:      #22c55e;
  --danger:  #ef4444;
  --text:    #e4edf8;
  --dim:     #99b0cc;
  --muted:   #527090;
  --mono:    'Space Mono', monospace;
  --sans:    'DM Sans', sans-serif;
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
  padding: 13px 24px; gap: 12px;
}
.brand-name {
  font-family: var(--mono); font-size: 1.38rem; font-weight: 700;
  color: var(--accent); letter-spacing: -0.02em; margin: 0;
  text-shadow: 0 0 18px rgba(0,212,170,0.3);
}
.brand-sub {
  font-size: 0.59rem; color: var(--muted); letter-spacing: 0.1em;
  text-transform: uppercase; font-family: var(--mono); margin-top: 2px;
}
.header-right { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.hbadge {
  font-family: var(--mono); font-size: 0.57rem; padding: 3px 8px;
  border-radius: 3px; border: 1px solid; letter-spacing: 0.05em;
}
.hbadge-g  { border-color: var(--accent); color: var(--accent);  background: rgba(0,212,170,0.07); }
.hbadge-p  { border-color: var(--a2);     color: var(--a2);      background: rgba(124,106,247,0.07); }
.hbadge-y  { border-color: var(--warn);   color: var(--warn);    background: rgba(245,158,11,0.07); }
.help-btn {
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--dim) !important; border-radius: 50% !important;
  width: 28px !important; height: 28px !important; padding: 0 !important;
  font-family: var(--mono) !important; font-size: 0.75rem !important;
  font-weight: 700 !important; cursor: pointer !important;
  line-height: 26px !important; text-align: center !important;
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
  font-family: var(--mono); font-size: 0.56rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
  margin: 12px 0 4px 0; display: block;
}
.query-hint { font-size: 0.65rem; color: #507080; margin: -2px 0 5px 0; font-style: italic; }
.sidebar-hr { border: none; border-top: 1px solid var(--border); margin: 9px 0; }

/* ── Inputs ───────────────────────────────────────────────────── */
.form-control, .form-select, textarea.form-control {
  background: #152030 !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; font-family: var(--sans) !important;
  font-size: 0.81rem !important; border-radius: 6px !important;
}
.form-control:focus, .form-select:focus, textarea.form-control:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,170,0.1) !important;
  outline: none !important; background: #1a2840 !important;
}

/* ── Checkboxes — high contrast ────────────────────────────────── */
.form-check { padding: 2px 0 2px 24px !important; margin-bottom: 1px !important; }
.form-check-input {
  background-color: #152030 !important; border: 1px solid #4a6888 !important;
  border-radius: 3px !important; cursor: pointer !important;
}
.form-check-input:checked {
  background-color: var(--accent) !important; border-color: var(--accent) !important;
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'%3e%3cpath fill='none' stroke='%230a0e1a' stroke-linecap='round' stroke-linejoin='round' stroke-width='3' d='m6 10 3 3 6-6'/%3e%3c/svg%3e") !important;
}
.form-check-label,
.shiny-input-checkboxgroup label,
.shiny-input-container .form-check-label {
  color: #c0d4ec !important;
  font-size: 0.77rem !important;
  font-family: var(--sans) !important;
  line-height: 1.45 !important;
  cursor: pointer !important;
}
.form-check-label:hover { color: var(--text) !important; }

/* ── Run button ───────────────────────────────────────────────── */
.run-btn {
  width: 100% !important;
  background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
  color: #06101e !important; border: none !important;
  padding: 10px 14px !important; border-radius: 7px !important;
  font-family: var(--mono) !important; font-size: 0.79rem !important;
  font-weight: 700 !important; letter-spacing: 0.06em !important;
  cursor: pointer !important; transition: all 0.18s ease !important;
  text-transform: uppercase !important;
}
.run-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 16px rgba(0,212,170,0.28) !important;
}
.run-btn:active { transform: translateY(0) !important; }

/* ── Agent log ────────────────────────────────────────────────── */
.log-box {
  background: #060c18; border: 1px solid var(--border);
  border-radius: 6px; padding: 7px 9px; max-height: 220px;
  overflow-y: auto; font-family: var(--mono); font-size: 0.56rem; line-height: 1.65;
}
.log-line      { color: #3a5570; white-space: pre-wrap; word-break: break-all; }
.log-line.ok   { color: #4a6888; }
.log-line.last { color: var(--accent); }
.log-running   { color: var(--accent); font-family: var(--mono); font-size: 0.58rem; }
.log-empty     { color: var(--muted); font-style: italic; font-size: 0.66rem; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.nav-tabs {
  border-bottom: 1px solid var(--border) !important;
  background: #0b1425 !important; padding: 0 14px !important;
}
.nav-tabs .nav-link {
  font-family: var(--mono) !important; font-size: 0.65rem !important;
  letter-spacing: 0.04em !important; text-transform: uppercase !important;
  color: var(--muted) !important; border: none !important;
  padding: 9px 12px !important; border-bottom: 2px solid transparent !important;
  border-radius: 0 !important; transition: color 0.14s !important;
}
.nav-tabs .nav-link:hover  { color: var(--dim) !important; }
.nav-tabs .nav-link.active {
  color: var(--accent) !important; background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
}
.tab-content { background: var(--bg) !important; padding: 20px 0 !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
.metrics-row { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 18px; }
.metric-card {
  background: var(--card); border: 1px solid var(--border);
  border-top: 2px solid var(--accent); border-radius: 8px;
  padding: 11px 14px; flex: 1; min-width: 100px;
}
.metric-card.up   { border-top-color: var(--ok); }
.metric-card.down { border-top-color: var(--danger); }
.m-label {
  font-family: var(--mono); font-size: 0.53rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em;
}
.m-value {
  font-family: var(--mono); font-size: 1.28rem; font-weight: 700;
  color: var(--accent); line-height: 1.15; margin-top: 3px;
}
.m-delta { font-family: var(--mono); font-size: 0.59rem; color: var(--muted); margin-top: 2px; }
.m-delta.pos { color: var(--ok); }
.m-delta.neg { color: var(--danger); }

/* ── Data table ───────────────────────────────────────────────── */
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.74rem; }
.data-table thead tr { background: var(--card); border-bottom: 2px solid var(--border); }
.data-table th {
  padding: 8px 12px; text-align: left; font-family: var(--mono);
  font-size: 0.54rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--muted); white-space: nowrap;
}
.data-table td {
  padding: 7px 12px; border-bottom: 1px solid var(--border);
  color: var(--dim); background: var(--surface);
}
.data-table tr:hover td { background: var(--card); color: var(--text); }
.q-cell { max-width: 340px; }
.prob-badge {
  display: inline-block; padding: 2px 7px; border-radius: 20px;
  font-family: var(--mono); font-size: 0.64rem; font-weight: 700;
}
.prob-h { background: rgba(34,197,94,0.1);  color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.prob-m { background: rgba(245,158,11,0.1); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
.prob-l { background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

/* ── Loading / empty ──────────────────────────────────────────── */
.loading-wrap {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; min-height: 280px; gap: 14px;
}
.spinner {
  width: 36px; height: 36px; border: 2px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-msg { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); }
.empty-state {
  display: flex; flex-direction: column; align-items: center;
  min-height: 200px; justify-content: center; gap: 8px; color: var(--muted);
}
.empty-icon { font-size: 1.8rem; opacity: 0.3; font-family: var(--mono); }
.empty-msg  { font-family: var(--mono); font-size: 0.68rem; }

/* ── Welcome ──────────────────────────────────────────────────── */
.welcome-wrap { max-width: 640px; margin: 32px auto; text-align: center; padding: 0 18px; }
.welcome-title {
  font-family: var(--mono); font-size: 1.75rem; color: var(--accent);
  margin-bottom: 12px; text-shadow: 0 0 22px rgba(0,212,170,0.25);
}
.welcome-sub { color: var(--dim); font-size: 0.84rem; margin-bottom: 20px; line-height: 1.65; }
.tech-badges { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-bottom: 22px; }
.tbadge {
  font-family: var(--mono); font-size: 0.57rem; padding: 3px 10px;
  border-radius: 3px; border: 1px solid var(--border);
  color: var(--dim); background: var(--card); letter-spacing: 0.06em;
}
.pipeline-vis {
  display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 5px;
  background: var(--card); border: 1px solid var(--border); border-radius: 9px; padding: 16px;
}
.pipe-node {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 11px; font-family: var(--mono);
  font-size: 0.63rem; color: var(--dim); text-align: center; line-height: 1.4;
}
.pipe-node.sup  { border-color: var(--accent); color: var(--accent); }
.pipe-node.data { border-color: var(--a2);     color: var(--a2); }
.pipe-node.ml   { border-color: var(--warn);   color: var(--warn); }
.pipe-node.llm  { border-color: #ec4899;       color: #ec4899; }
.pipe-arrow { color: var(--muted); font-size: 0.8rem; }

/* ── Section heading ──────────────────────────────────────────── */
.section-heading {
  font-family: var(--mono); font-size: 0.59rem; text-transform: uppercase;
  letter-spacing: 0.12em; color: var(--muted); margin: 18px 0 9px 0;
  display: flex; align-items: center; gap: 8px;
}
.section-heading::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── 2-col plot grid ──────────────────────────────────────────── */
.plot-row { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; margin-bottom: 11px; }

/* ── Geo note ─────────────────────────────────────────────────── */
.geo-note {
  background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--warn);
  border-radius: 7px; padding: 10px 14px; margin-bottom: 14px;
  font-size: 0.76rem; color: var(--dim); line-height: 1.55;
}
.geo-note strong { color: var(--warn); }

/* ── Cluster cards ────────────────────────────────────────────── */
.cluster-cards { display: flex; flex-wrap: wrap; gap: 11px; margin-top: 14px; }
.cluster-card {
  background: var(--card); border: 1px solid var(--border);
  border-left: 3px solid var(--a2); border-radius: 8px;
  padding: 13px 16px; flex: 1; min-width: 195px; max-width: 330px;
}
.cc-header {
  font-family: var(--mono); font-weight: 700; font-size: 0.7rem;
  color: var(--a2); margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.04em;
}
.cc-stats {
  display: flex; gap: 11px; flex-wrap: wrap;
  font-family: var(--mono); font-size: 0.6rem; color: var(--muted); margin-bottom: 8px;
}
.cc-val { color: var(--dim); }
.cc-markets { font-size: 0.68rem; color: #5a7890; padding-left: 13px; margin: 0; line-height: 1.6; }

/* ── Insights ─────────────────────────────────────────────────── */
.insights-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 9px; padding: 22px 30px; max-width: 820px;
  line-height: 1.75; font-size: 0.85rem; color: var(--text);
}
.insights-card h3 {
  font-family: var(--mono); font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 5px; margin: 20px 0 9px 0;
}
.insights-card h3:first-child { margin-top: 0; }
.insights-card strong { color: #dce8f8; }
.insights-card li { margin-bottom: 5px; color: var(--dim); }
.insights-card p  { color: var(--dim); margin-bottom: 8px; }
.insights-hdr {
  font-family: var(--mono); font-size: 0.61rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em;
  margin-bottom: 14px; padding-bottom: 9px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 9px;
}
.insights-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0;
  animation: pdot 2s ease-in-out infinite;
}
@keyframes pdot {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,212,170,0.4); }
  50%      { box-shadow: 0 0 0 5px rgba(0,212,170,0); }
}

/* ── Modal ────────────────────────────────────────────────────── */
.modal-content { background: #0e1a2e !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
.modal-header  { border-bottom: 1px solid var(--border) !important; }
.modal-footer  { border-top:    1px solid var(--border) !important; }
.modal-title   { font-family: var(--mono) !important; font-size: 0.86rem !important; color: var(--accent) !important; letter-spacing: 0.05em !important; }
.btn-close     { filter: invert(1) opacity(0.6) !important; }
.modal-body h5 {
  font-family: var(--mono); font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent); margin: 16px 0 6px 0;
  border-bottom: 1px solid var(--border); padding-bottom: 4px;
}
.modal-body h5:first-child { margin-top: 0; }
.modal-body p, .modal-body li { font-size: 0.81rem; color: var(--dim); line-height: 1.65; }
.modal-body ul { padding-left: 17px; }
.modal-body li { margin-bottom: 4px; }
.modal-body code {
  background: var(--card); border: 1px solid var(--border);
  padding: 1px 5px; border-radius: 3px; font-family: var(--mono);
  font-size: 0.71rem; color: var(--accent);
}
.modal-close-btn {
  background: var(--accent) !important; color: #06101e !important;
  border: none !important; padding: 6px 18px !important; border-radius: 5px !important;
  font-family: var(--mono) !important; font-size: 0.71rem !important;
  font-weight: 700 !important; cursor: pointer !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
"""

# ══════════════════════════════════════════════════════════════════
#  PLOTLY THEME  (no title/xaxis/yaxis in base dict)
# ══════════════════════════════════════════════════════════════════

_AX   = dict(gridcolor="#1b2c44", zerolinecolor="#1b2c44",
             tickfont=dict(size=9, color="#527090"), linecolor="#263550")
_TFNT = dict(family="DM Sans, sans-serif", size=12, color="#c0d4ec")


def _base() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", size=10, color="#527090"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#263550",
                    font=dict(size=9, color="#99b0cc")),
        margin=dict(l=50, r=20, t=44, b=38),
    )


def theme(fig: go.Figure, title: str = "", height: int = None) -> go.Figure:
    """Apply dark theme safely — no keyword conflicts."""
    kw = _base()
    if title:
        kw["title"] = dict(text=title, font=_TFNT, x=0.01)
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)
    return fig


def fig_html(fig: go.Figure, first: bool = False) -> HTML:
    js = "cdn" if first else False
    return HTML(fig.to_html(include_plotlyjs=js, full_html=False))


# ══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════

def mcard(label: str, value: str, delta: str = "", trend: str = "") -> ui.Tag:
    dcls  = "pos" if trend == "up" else "neg" if trend == "down" else ""
    dhtml = f'<div class="m-delta {dcls}">{delta}</div>' if delta else ""
    ccls  = f"metric-card {trend}" if trend else "metric-card"
    return ui.div(
        HTML(f'<div class="m-label">{label}</div>'
             f'<div class="m-value">{value}</div>{dhtml}'),
        class_=ccls,
    )


def loading(msg: str = "Running...") -> ui.Tag:
    return ui.div(
        HTML(f'<div class="spinner"></div><p class="loading-msg">{msg}</p>'),
        class_="loading-wrap",
    )


def empty(msg: str) -> ui.Tag:
    return ui.div(
        HTML(f'<div class="empty-icon">[ ]</div><p class="empty-msg">{msg}</p>'),
        class_="empty-state",
    )


def md2html(text: str) -> str:
    text = re.sub(r"^## (.+)$",         r"<h3>\1</h3>",        text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$",        r"<h4>\1</h4>",        text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",         r"<em>\1</em>",         text)
    text = re.sub(r"^[\-\*\d+\.\#] (.+)$", r"<li>\1</li>",     text, flags=re.MULTILINE)
    text = text.replace("\n\n", "</p><p>")
    return f"<p>{text}</p>"


def prob_badge(p: float) -> str:
    cls = "prob-h" if p > 0.65 else "prob-l" if p < 0.35 else "prob-m"
    return f'<span class="prob-badge {cls}">{p*100:.1f}%</span>'


# ══════════════════════════════════════════════════════════════════
#  HELP MODAL
# ══════════════════════════════════════════════════════════════════

HELP_HTML = """
<h5>What is PredictIQ?</h5>
<p>A 5-agent intelligence platform connecting live prediction market data to US Census
demographic data, with geographic analysis and AI synthesis.</p>

<h5>Supervisor / Agent Architecture</h5>
<ul>
  <li><strong>Supervisor</strong> — LangGraph orchestrator. Routes to whichever agent stage has not yet completed.</li>
  <li><strong>Polymarket Agent</strong> — Fetches up to 500 markets from the free Gamma API. Applies server-side tag filtering AND client-side slug matching, so the category selector reliably filters results.</li>
  <li><strong>ACS Agent</strong> — Pulls ACS 5-Year Estimates for all 50 states + DC from the Census Bureau. Derives poverty rate, unemployment rate, and bachelor's degree attainment. Requires <code>CENSUS_API_KEY</code>.</li>
  <li><strong>Geography Agent</strong> — Parses market questions for US state references using regex. Aggregates market count, volume, and avg YES probability per state. <em>Note: Polymarket/Kalshi do not expose bettor location — this maps what geographic outcomes are being bet on, not bettor origin.</em></li>
  <li><strong>Analysis Agent</strong> — Silhouette sweep + K-Means + PCA on market features.</li>
  <li><strong>Insights Agent</strong> — Sends all data to Groq LLaMA-3.3-70B for a structured brief with a geographic/demographic lens.</li>
</ul>

<h5>Research Query</h5>
<p>Click into the text area at the top of the sidebar and type any question before running.
The AI Insights agent uses it to shape the brief. Examples:</p>
<ul>
  <li><em>Are markets in low-income states betting more on economic hardship outcomes?</em></li>
  <li><em>How are prediction markets pricing US state-level political outcomes vs ACS income data?</em></li>
  <li><em>Which states have the most market activity and what does their poverty rate look like?</em></li>
</ul>

<h5>Geography Tab</h5>
<p>Shows a US choropleth of market activity by state (extracted from question text), then an
ACS overlay comparing that activity to median income and poverty rates. Because prediction
market platforms are pseudonymous, we map <strong>what is being bet on</strong>, not who is
betting where.</p>

<h5>ACS Metrics</h5>
<p>Select which Census indicators to pull and display. Requires <code>CENSUS_API_KEY</code>
(free at <strong>api.census.gov/data/key_signup.html</strong>).</p>

<h5>Required Environment Variables</h5>
<ul>
  <li><code>GROQ_API_KEY</code> — LLM insights (console.groq.com)</li>
  <li><code>CENSUS_API_KEY</code> — ACS data (api.census.gov)</li>
  <li>Polymarket — no key needed</li>
</ul>

<h5>Dashboard Tabs</h5>
<ul>
  <li><strong>Overview</strong> — summary metrics, probability histogram, top-20 markets</li>
  <li><strong>Markets</strong> — volume chart, scatter, category breakdown</li>
  <li><strong>Geography</strong> — US choropleth + ACS overlay + state market table</li>
  <li><strong>Demographics</strong> — ACS state-level charts and comparisons</li>
  <li><strong>Clusters</strong> — PCA scatter, silhouette sweep, cluster details</li>
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
                     '<div class="brand-sub">Agentic Market Intelligence &middot; '
                     'LangGraph Supervisor &middot; Census ACS</div>'),
            ),
            ui.div(
                HTML('<div class="header-right">'
                     '<span class="hbadge hbadge-g">POLYMARKET</span>'
                     '<span class="hbadge hbadge-g">CENSUS ACS</span>'
                     '<span class="hbadge hbadge-p">GROQ LLAMA-3</span>'
                     '<span class="hbadge hbadge-y">K-MEANS + PCA</span>'
                     '</div>'),
                ui.input_action_button("help_btn", "?", class_="help-btn"),
                style="display:flex;align-items:center;gap:9px;",
            ),
            class_="header-inner",
        ),
        class_="app-header",
    ),

    ui.layout_sidebar(
        ui.sidebar(
            # 1. Research Query
            HTML('<span class="sidebar-label">Research Query</span>'
                 '<p class="query-hint">Edit — shapes the AI brief</p>'),
            ui.input_text_area(
                "query", None,
                value="How do prediction markets reflect US demographic and economic divides? Which states show the most market activity and how does that align with Census poverty and income data?",
                rows=4,
            ),

            HTML('<hr class="sidebar-hr">'),
            # 2. Market Category
            HTML('<span class="sidebar-label">Market Category</span>'),
            ui.input_select("category", None, choices=POLY_CATS, selected=""),

            HTML('<hr class="sidebar-hr">'),
            # 3. RUN BUTTON — positioned above the long checklist
            ui.input_action_button("run_btn", "RUN ANALYSIS", class_="run-btn"),

            HTML('<hr class="sidebar-hr">'),
            # 4. ACS Metrics
            HTML('<span class="sidebar-label">ACS Metrics</span>'),
            ui.input_checkbox_group(
                "acs_metrics", None,
                choices=ACS_METRIC_LABELS,
                selected=["med_income", "poverty_rate", "unemp_rate",
                          "med_rent", "bach_rate"],
            ),

            HTML('<hr class="sidebar-hr">'),
            HTML('<span class="sidebar-label">Agent Log</span>'),
            ui.output_ui("agent_log"),

            width=308,
            bg="#0b1425",
        ),

        ui.navset_tab(
            ui.nav_panel("Overview",    ui.div(ui.output_ui("tab_overview"),    style="padding:15px 20px;")),
            ui.nav_panel("Markets",     ui.div(ui.output_ui("tab_markets"),     style="padding:15px 20px;")),
            ui.nav_panel("Geography",   ui.div(ui.output_ui("tab_geo"),         style="padding:15px 20px;")),
            ui.nav_panel("Demographics",ui.div(ui.output_ui("tab_demo"),        style="padding:15px 20px;")),
            ui.nav_panel("Clusters",    ui.div(ui.output_ui("tab_clusters"),    style="padding:15px 20px;")),
            ui.nav_panel("AI Insights", ui.div(ui.output_ui("tab_insights"),    style="padding:15px 20px;")),
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
    def _help():
        ui.modal_show(ui.modal(
            HTML(HELP_HTML),
            title="How to use PredictIQ",
            easy_close=True,
            footer=ui.modal_button("Close", class_="modal-close-btn"),
            size="l",
        ))

    # ── Pipeline ───────────────────────────────────────────────
    @reactive.extended_task
    async def pipeline(q: str, cat: str, metrics: List[str]):
        return await asyncio.to_thread(run_graph, q, cat, metrics)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        result_rv.set(None)
        pipeline(input.query(), input.category(), list(input.acs_metrics()))

    @reactive.effect
    def _collect():
        if pipeline.status() == "success":
            result_rv.set(pipeline.result())

    # ── Agent log ──────────────────────────────────────────────
    @output
    @render.ui
    def agent_log():
        if pipeline.status() == "running":
            return HTML('<div class="log-box"><div class="log-running">&#9679; RUNNING...</div></div>')
        s = result_rv.get()
        if s is None:
            return HTML('<div class="log-box"><div class="log-empty">Awaiting run...</div></div>')
        lines = s.get("log", [])[-30:]
        items = [
            f'<div class="log-line {"last" if i==len(lines)-1 else "ok"}">{ln}</div>'
            for i, ln in enumerate(lines)
        ]
        return HTML(f'<div class="log-box">{"".join(items)}</div>')

    # ── TAB: Overview ──────────────────────────────────────────
    @output
    @render.ui
    def tab_overview():
        if pipeline.status() == "running":
            return loading("Executing 5-agent pipeline...")
        s = result_rv.get()
        if s is None:
            return HTML("""
            <div class="welcome-wrap">
              <div class="welcome-title">&#x2B21; PREDICTIQ</div>
              <p class="welcome-sub">An agentic intelligence platform connecting live prediction
              markets to US Census ACS demographic data, with geographic analysis and AI synthesis.<br><br>
              Click <strong style="color:#00d4aa">RUN ANALYSIS</strong> to start, or
              <strong style="color:#00d4aa">?</strong> in the header for instructions.</p>
              <div class="tech-badges">
                <span class="tbadge">LANGGRAPH</span><span class="tbadge">LANGCHAIN</span>
                <span class="tbadge">GROQ LLAMA-3</span><span class="tbadge">POLYMARKET</span>
                <span class="tbadge">CENSUS ACS</span><span class="tbadge">K-MEANS</span>
                <span class="tbadge">PCA</span><span class="tbadge">SHINY</span>
              </div>
              <div class="pipeline-vis">
                <div class="pipe-node sup">SUPERVISOR<br><small>orchestrator</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node data">POLYMARKET<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node data">ACS<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node data">GEOGRAPHY<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node ml">ANALYSIS<br><small>agent</small></div>
                <div class="pipe-arrow">&#8594;</div>
                <div class="pipe-node llm">INSIGHTS<br><small>agent</small></div>
              </div>
            </div>""")

        cr  = s.get("cluster_results", {})
        md  = s.get("market_data", [])
        gd  = s.get("geo_data", {})
        ad  = s.get("acs_data", {})
        sil = cr.get("sil_score")

        geo_recs   = gd.get("records", [])
        states_hit = len(geo_recs)

        rows = ui.div(
            mcard("Markets",    f"{cr.get('total', len(md)):,}"),
            mcard("Volume",     f"${cr.get('total_vol', 0)/1e6:.1f}M"),
            mcard("Avg YES",    f"{cr.get('avg_prob', 0.5)*100:.1f}%"),
            mcard("Clusters",   str(cr.get("n_clusters", "—"))),
            mcard("States Hit", str(states_hit)),
            mcard("Silhouette", f"{sil:.3f}" if sil else "—"),
            mcard("ACS States", str(len(ad.get("records", [])))),
            class_="metrics-row",
        )

        # Probability histogram
        hist = ui.div()
        if md:
            dff = pd.DataFrame(md)
            dff = dff[dff["volume"] > 0]
            fig = px.histogram(dff, x="yes_prob", nbins=50,
                               color_discrete_sequence=[ACCENT],
                               labels={"yes_prob": "YES Probability", "count": "Markets"},
                               height=250)
            fig.update_traces(marker_line_width=0)
            fig.add_vline(x=0.5, line_dash="dash", line_color="#374151",
                          annotation_text="50/50", annotation_font_size=9,
                          annotation_font_color="#527090")
            theme(fig, title="YES Probability Distribution")
            hist = fig_html(fig, first=True)

        # Top markets table
        top  = sorted(md, key=lambda x: x["volume"], reverse=True)[:20]
        rows_html = "".join(
            f'<tr>'
            f'<td class="q-cell">{m["question"][:75]}{"..." if len(m["question"])>75 else ""}</td>'
            f'<td>{prob_badge(m["yes_prob"])}</td>'
            f'<td style="font-family:var(--mono);font-size:0.67rem;">${m["volume"]:,.0f}</td>'
            f'<td style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);">{m.get("category","")}</td>'
            f'<td style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);">{m.get("end_date","")}</td>'
            f'</tr>'
            for m in top
        )
        table = HTML(
            f'<div class="table-wrap"><table class="data-table">'
            f'<thead><tr><th>Question</th><th>YES%</th><th>Volume</th>'
            f'<th>Category</th><th>Ends</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )
        return ui.div(
            rows, hist,
            HTML('<div class="section-heading">Top Markets by Volume</div>'),
            table,
        )

    # ── TAB: Markets ───────────────────────────────────────────
    @output
    @render.ui
    def tab_markets():
        if pipeline.status() == "running":
            return loading("Fetching & filtering Polymarket data...")
        s = result_rv.get()
        if not s or not s.get("market_data"):
            return empty("No market data — run analysis first")

        md = s["market_data"]
        df = pd.DataFrame(md)
        df = df[df["volume"] > 0].copy()

        top20         = df.nlargest(20, "volume").copy()
        top20["qlbl"] = top20["question"].str[:65] + "..."

        fig1 = px.bar(
            top20, x="volume", y="qlbl", orientation="h",
            color="yes_prob",
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
            range_color=[0, 1],
            labels={"volume": "Volume ($)", "qlbl": "", "yes_prob": "YES"},
            height=590,
        )
        theme(fig1, title="Top 20 Markets by Volume  (colour = YES probability)")
        fig1.update_yaxes(autorange="reversed", tickfont=dict(size=8))
        fig1.update_layout(
            coloraxis_colorbar=dict(title="YES", tickformat=".0%", tickfont=dict(size=8)),
        )

        fig2 = px.scatter(
            df.head(150), x="yes_prob", y="volume",
            color="category", size="liquidity",
            hover_name="question", log_y=True,
            labels={"yes_prob": "YES Probability", "volume": "Volume ($, log)"},
            height=410,
        )
        theme(fig2, title="Probability vs Volume  (bubble size = liquidity)")
        fig2.update_traces(marker=dict(opacity=0.75))

        cat_df = (df.groupby("category")
                  .agg(n=("id", "count"), vol=("volume", "sum"))
                  .reset_index().sort_values("vol", ascending=False).head(10))
        fig3 = go.Figure(go.Bar(
            x=cat_df["category"], y=cat_df["vol"],
            marker_color=ACCENT,
            text=cat_df["n"].astype(str), textposition="outside",
            textfont=dict(size=8),
        ))
        theme(fig3, title="Total Volume by Category", height=300)
        fig3.update_xaxes(title="")
        fig3.update_yaxes(title="Volume ($)")

        return ui.div(fig_html(fig1, first=True), fig_html(fig2), fig_html(fig3))

    # ── TAB: Geography ─────────────────────────────────────────
    @output
    @render.ui
    def tab_geo():
        if pipeline.status() == "running":
            return loading("Extracting geographic signals & loading ACS data...")
        s = result_rv.get()
        if not s:
            return empty("Run analysis first")

        geo_d = s.get("geo_data", {})
        acs_d = s.get("acs_data", {})
        geo_recs = geo_d.get("records", [])
        acs_recs = acs_d.get("records", [])

        note = HTML(
            '<div class="geo-note">'
            '<strong>Geographic interpretation:</strong> Polymarket and Kalshi are pseudonymous — '
            'bettor location is not exposed. The map below shows <em>what geographic outcomes '
            'markets are betting on</em> (extracted from question text via regex), not where '
            'bettors are located. The ACS overlay adds Census economic context per state.'
            '</div>'
        )

        if not geo_recs:
            return ui.div(note, empty("No US state references found in current market questions"))

        geo_df = pd.DataFrame(geo_recs)

        # ── Choropleth 1: Market volume by state ──────────────
        fig1 = go.Figure(go.Choropleth(
            locations=geo_df["abbr"],
            z=np.log1p(geo_df["volume"]),
            locationmode="USA-states",
            colorscale=[[0, "#0d1a2a"], [0.5, "#1a4060"], [1, ACCENT]],
            showscale=True,
            colorbar=dict(title="log(Vol)", tickfont=dict(size=8), len=0.6),
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Markets: %{customdata[0]}<br>"
                "Volume: $%{customdata[1]:,.0f}<br>"
                "Avg YES: %{customdata[2]:.1%}<extra></extra>"
            ),
            customdata=geo_df[["count", "volume", "avg_prob"]].values,
        ))
        fig1.update_layout(
            geo=dict(
                scope="usa",
                showlakes=True,
                lakecolor="rgba(10,14,26,0.8)",
                bgcolor="rgba(0,0,0,0)",
                landcolor="#152030",
                coastlinecolor="#263550",
                showcoastlines=True,
                projection_type="albers usa",
            ),
            **_base(),
            title=dict(text="Market Focus by State  (log-scale volume)", font=_TFNT, x=0.01),
            height=440,
        )

        # ── Choropleth 2: ACS poverty rate overlay ─────────────
        fig2_html = ui.div()
        fig3_html = ui.div()
        fig4_html = ui.div()

        if acs_recs:
            acs_df = pd.DataFrame(acs_recs).dropna(subset=["abbr"])

            # Poverty rate choropleth
            fig2 = go.Figure(go.Choropleth(
                locations=acs_df["abbr"],
                z=acs_df["poverty_rate"],
                locationmode="USA-states",
                colorscale=[[0, "#162a1e"], [0.5, "#854d0e"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(title="Pov %", tickfont=dict(size=8), len=0.6),
                hovertemplate=(
                    "<b>%{location} — %{customdata[0]}</b><br>"
                    "Poverty Rate: %{z:.1f}%<br>"
                    "Median Income: $%{customdata[1]:,}<br>"
                    "Unemp Rate: %{customdata[2]:.1f}%<extra></extra>"
                ),
                customdata=acs_df[["state_name", "med_income", "unemp_rate"]].fillna(0).values,
            ))
            fig2.update_layout(
                geo=dict(
                    scope="usa", showlakes=True, lakecolor="rgba(10,14,26,0.8)",
                    bgcolor="rgba(0,0,0,0)", landcolor="#152030",
                    coastlinecolor="#263550", projection_type="albers usa",
                ),
                **_base(),
                title=dict(text="ACS Poverty Rate by State  (Census 2022)", font=_TFNT, x=0.01),
                height=400,
            )
            fig2_html = fig_html(fig2)

            # Scatter: market volume vs poverty rate
            merged = acs_df.merge(geo_df[["abbr", "volume", "count"]], on="abbr", how="left")
            merged["volume"]   = merged["volume"].fillna(0)
            merged["count"]    = merged["count"].fillna(0)
            merged["has_mkts"] = merged["volume"] > 0

            fig3 = px.scatter(
                merged, x="poverty_rate", y="med_income",
                size=np.log1p(merged["volume"]) + 1,
                color="unemp_rate",
                color_continuous_scale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                hover_name="state_name",
                hover_data={
                    "abbr": True, "poverty_rate": ":.1f",
                    "med_income": ":,", "unemp_rate": ":.1f",
                    "count": True, "volume": ":,.0f",
                },
                labels={
                    "poverty_rate": "Poverty Rate (%)",
                    "med_income":   "Median Household Income ($)",
                    "unemp_rate":   "Unemp Rate",
                },
                height=440,
            )
            theme(fig3,
                  title="State Economics vs Market Activity  "
                        "(bubble size = market volume, colour = unemployment)")
            fig3.update_traces(marker=dict(opacity=0.85))
            # Annotate top market states
            for _, row in merged.nlargest(8, "volume").iterrows():
                fig3.add_annotation(
                    x=row["poverty_rate"], y=row["med_income"],
                    text=row["abbr"], showarrow=False,
                    font=dict(size=8, color="#99b0cc"), yshift=12,
                )
            fig3_html = fig_html(fig3)

            # Bar: top 15 states by market count
            top15 = geo_df.head(15).merge(acs_df[["abbr", "med_income", "poverty_rate"]],
                                          on="abbr", how="left")
            fig4 = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Market Volume by State", "Poverty Rate of Market-Active States"],
                horizontal_spacing=0.1,
            )
            fig4.add_trace(
                go.Bar(x=top15["abbr"], y=top15["volume"],
                       marker_color=ACCENT, showlegend=False,
                       hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"),
                row=1, col=1,
            )
            fig4.add_trace(
                go.Bar(x=top15["abbr"],
                       y=top15["poverty_rate"].fillna(0),
                       marker_color="#ef4444",
                       showlegend=False,
                       hovertemplate="%{x}: %{y:.1f}%<extra></extra>"),
                row=1, col=2,
            )
            theme(fig4, height=310)
            for ann in fig4.layout.annotations:
                ann.update(font=dict(size=10, color="#99b0cc"))
            fig4_html = fig_html(fig4)

        # State reference table
        tbl_rows = "".join(
            f'<tr>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">{r["abbr"]}</td>'
            f'<td>{r["state_name"]}</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">{r["count"]}</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">${r["volume"]:,.0f}</td>'
            f'<td>{prob_badge(r["avg_prob"])}</td>'
            f'<td style="font-size:0.68rem;color:var(--muted);">'
            f'{"; ".join(q[:55] + "..." for q in r["top_questions"][:2])}'
            f'</td>'
            f'</tr>'
            for r in geo_recs[:20]
        )
        state_table = HTML(
            '<div class="table-wrap"><table class="data-table">'
            '<thead><tr><th>Abbr</th><th>State</th><th>Mkt Count</th>'
            '<th>Volume</th><th>Avg YES</th><th>Sample Questions</th></tr></thead>'
            f'<tbody>{tbl_rows}</tbody></table></div>'
        )

        return ui.div(
            note,
            fig_html(fig1, first=True),
            fig2_html,
            fig3_html,
            fig4_html,
            HTML('<div class="section-heading">States Referenced in Market Questions</div>'),
            state_table,
        )

    # ── TAB: Demographics (ACS) ────────────────────────────────
    @output
    @render.ui
    def tab_demo():
        if pipeline.status() == "running":
            return loading("Pulling Census ACS 5-year estimates...")
        s = result_rv.get()
        if not s or not s.get("acs_data"):
            return empty("No ACS data — run analysis first (requires CENSUS_API_KEY)")

        acs_d = s["acs_data"]
        recs  = acs_d.get("records", [])
        if not recs:
            return empty("ACS returned no records — check CENSUS_API_KEY")

        df = pd.DataFrame(recs).dropna(subset=["abbr"])
        selected = list(input.acs_metrics()) if pipeline.status() != "running" else []

        COLORS = [ACCENT, "#7c6af7", "#f59e0b", "#ec4899",
                  "#22c55e", "#38bdf8", "#fb923c", "#a78bfa"]

        # Metric summary cards (national averages)
        avg_income  = df["med_income"].median()
        avg_poverty = df["poverty_rate"].median()
        avg_unemp   = df["unemp_rate"].median()
        avg_rent    = df["med_rent"].median()
        avg_bach    = df["bach_rate"].median()
        avg_age     = df["median_age"].median()

        cards = ui.div(
            mcard("Median Income",  f"${avg_income:,.0f}" if not np.isnan(avg_income) else "—",
                  delta="national median", trend=""),
            mcard("Poverty Rate",   f"{avg_poverty:.1f}%" if not np.isnan(avg_poverty) else "—",
                  delta="median across states"),
            mcard("Unemp Rate",     f"{avg_unemp:.1f}%"  if not np.isnan(avg_unemp)  else "—"),
            mcard("Median Rent",    f"${avg_rent:,.0f}"  if not np.isnan(avg_rent)   else "—"),
            mcard("Bach Degree",    f"{avg_bach:.1f}%"   if not np.isnan(avg_bach)   else "—"),
            mcard("Median Age",     f"{avg_age:.1f}"     if not np.isnan(avg_age)    else "—"),
            class_="metrics-row",
        )

        METRIC_COLS: Dict[str, str] = {
            "med_income":   "Median Household Income ($)",
            "poverty_rate": "Poverty Rate (%)",
            "unemp_rate":   "Unemployment Rate (%)",
            "med_rent":     "Median Gross Rent ($)",
            "bach_rate":    "Bachelor's Degree Rate (%)",
            "median_age":   "Median Age",
            "population":   "Population",
        }

        active_metrics = [m for m in selected if m in METRIC_COLS]
        if not active_metrics:
            active_metrics = ["med_income", "poverty_rate", "unemp_rate"]

        plot_rows = []
        first     = True
        for i in range(0, len(active_metrics), 2):
            chunk = active_metrics[i:i+2]
            figs  = []
            for j, col in enumerate(chunk):
                col_label = METRIC_COLS[col]
                col_color = COLORS[(i + j) % len(COLORS)]
                sub_df    = df.dropna(subset=[col]).sort_values(col, ascending=False)

                fig = go.Figure(go.Bar(
                    x=sub_df["abbr"], y=sub_df[col],
                    marker_color=col_color,
                    hovertemplate=f"%{{x}}: %{{y}}<br>{col_label}<extra></extra>",
                ))
                theme(fig, title=col_label, height=280)
                fig.update_xaxes(tickfont=dict(size=7))
                fig.update_layout(showlegend=False)
                figs.append(fig_html(fig, first=first))
                first = False
            plot_rows.append(ui.div(*figs, class_="plot-row"))

        # Correlation matrix scatter: income vs poverty
        fig_corr = px.scatter(
            df.dropna(subset=["med_income", "poverty_rate", "unemp_rate"]),
            x="med_income", y="poverty_rate",
            color="unemp_rate",
            size="population",
            hover_name="state_name",
            color_continuous_scale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
            labels={
                "med_income":   "Median Household Income ($)",
                "poverty_rate": "Poverty Rate (%)",
                "unemp_rate":   "Unemployment Rate (%)",
            },
            height=420,
        )
        theme(fig_corr,
              title="Income vs Poverty by State  (bubble = population, colour = unemployment)")
        # Annotate a few outliers
        outliers = pd.concat([
            df.nlargest(4, "med_income"), df.nlargest(4, "poverty_rate")
        ]).drop_duplicates("abbr").dropna(subset=["med_income", "poverty_rate"])
        for _, row in outliers.iterrows():
            fig_corr.add_annotation(
                x=row["med_income"], y=row["poverty_rate"],
                text=row["abbr"], showarrow=False,
                font=dict(size=8, color="#99b0cc"), yshift=11,
            )

        # ACS data table
        tbl_df = df.sort_values("med_income", ascending=False)
        tbl_rows_html = "".join(
            f'<tr>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">{row["abbr"]}</td>'
            f'<td>{row["state_name"]}</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">'
            f'${int(row["med_income"]):,}' if row["med_income"] else '<td>—</td>'
            f'</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">'
            f'{row["poverty_rate"]:.1f}%' if row["poverty_rate"] else '—'
            f'</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">'
            f'{row["unemp_rate"]:.1f}%' if row["unemp_rate"] else '—'
            f'</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">'
            f'${int(row["med_rent"]):,}' if row["med_rent"] else '—'
            f'</td>'
            f'<td style="font-family:var(--mono);font-size:0.7rem;">'
            f'{row["bach_rate"]:.1f}%' if row["bach_rate"] else '—'
            f'</td>'
            f'</tr>'
            for _, row in tbl_df.iterrows()
        )

        return ui.div(
            cards,
            *plot_rows,
            fig_html(fig_corr),
            HTML('<div class="section-heading">ACS State Data Table (ranked by median income)</div>'),
            HTML(
                '<div class="table-wrap"><table class="data-table">'
                '<thead><tr><th>State</th><th>Name</th><th>Med Income</th>'
                '<th>Poverty%</th><th>Unemp%</th><th>Med Rent</th><th>Bach%</th></tr></thead>'
                f'<tbody>{tbl_rows_html}</tbody></table></div>'
            ),
        )

    # ── TAB: Clusters ──────────────────────────────────────────
    @output
    @render.ui
    def tab_clusters():
        if pipeline.status() == "running":
            return loading("Running K-Means clustering & PCA...")
        s = result_rv.get()
        if not s or not s.get("cluster_results"):
            return empty("No cluster results — run analysis first")

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

        fig1 = px.scatter(
            df, x="pc1", y="pc2", color="cluster_label",
            hover_name="question",
            hover_data={"yes_prob": ":.1%", "volume": ":,.0f",
                        "cluster_label": False, "cluster_str": False,
                        "pc1": False, "pc2": False},
            labels={"pc1": f"PC1 ({pv[0]*100:.1f}% var)",
                    "pc2": f"PC2 ({pv[1]*100:.1f}% var)",
                    "cluster_label": "Cluster"},
            color_discrete_sequence=PAL, height=460,
        )
        theme(fig1, title=f"PCA Cluster Map — {cr['n_clusters']} clusters  "
                          f"(silhouette = {cr['sil_score']:.3f})")
        fig1.update_traces(marker=dict(size=7, opacity=0.8))

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
            theme(f2, title="Silhouette Score vs K  (selected K highlighted)", height=255)
            f2.update_xaxes(title="K (clusters)")
            f2.update_yaxes(title="Silhouette Score")
            fig2 = fig_html(f2)

        sums = cr.get("summaries", [])
        fig3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Market Count", "Avg YES Prob (%)", "Avg Volume ($)"],
            horizontal_spacing=0.07,
        )
        for ci, y in enumerate([[s["count"] for s in sums],
                                 [s["avg_prob"]*100 for s in sums],
                                 [s["avg_volume"] for s in sums]], 1):
            fig3.add_trace(
                go.Bar(x=[s["label"] for s in sums], y=y,
                       marker_color=PAL[:len(sums)], showlegend=False,
                       text=[f"{v:.0f}" for v in y], textposition="outside",
                       textfont=dict(size=7)),
                row=1, col=ci,
            )
        theme(fig3, height=295)
        for ann in fig3.layout.annotations:
            ann.update(font=dict(size=9, color="#99b0cc"))
        fig3.update_xaxes(tickfont=dict(size=7))

        cards = []
        for su in sums:
            mkt_li = "".join(
                f'<li>{q[:65]}{"..." if len(q)>65 else ""}</li>'
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
            fig_html(fig1, first=True), fig2, fig_html(fig3),
            HTML('<div class="section-heading">Cluster Details</div>'),
            ui.div(*cards, style="display:flex;flex-wrap:wrap;gap:11px;margin-top:12px;"),
        )

    # ── TAB: AI Insights ───────────────────────────────────────
    @output
    @render.ui
    def tab_insights():
        if pipeline.status() == "running":
            return loading(f"Generating intelligence brief via Groq {MODEL}...")
        s = result_rv.get()
        if not s or not s.get("insights"):
            return empty("No insights yet — run analysis first")
        body = md2html(s["insights"])
        now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
        return HTML(
            f'<div class="insights-card">'
            f'<div class="insights-hdr">'
            f'<span class="insights-dot"></span>'
            f'GROQ &middot; {MODEL} &middot; {now}'
            f'</div>{body}</div>'
        )


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

app = App(app_ui, server)
