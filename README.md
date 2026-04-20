# ⬡ PredictIQ — Agentic Prediction Market Intelligence Platform

A production-grade agentic workflow built with **LangGraph**, **LangChain**, and **Groq**,
delivering a live intelligence dashboard via **Shiny for Python**.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       LangGraph Workflow                          │
│                                                                   │
│  ┌─────────────┐                                                  │
│  │  SUPERVISOR  │  ← reads state, routes to next uncompleted agent│
│  └──────┬───────┘                                                  │
│         │ conditional edges                                        │
│  ┌──────▼──────┐ ┌──────┐ ┌───────────┐ ┌──────────┐ ┌────────┐  │
│  │ POLYMARKET  │ │ ACS  │ │ GEOGRAPHY │ │ ANALYSIS │ │INSIGHTS│  │
│  │   agent     │ │agent │ │  agent    │ │  agent   │ │ agent  │  │
│  └─────────────┘ └──────┘ └───────────┘ └──────────┘ └────────┘  │
│  Free Gamma API  Census   regex + state  K-Means+PCA  Groq LLaMA  │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Shiny for Python Dashboard (6 tabs)
```

### Agents

| Agent | Role | Data Source |
|-------|------|-------------|
| **Supervisor** | Reads pipeline state; routes to next incomplete agent | Internal state |
| **Polymarket Agent** | Fetches live prediction markets with dual-layer filtering | `gamma-api.polymarket.com` (free) |
| **ACS Agent** | Pulls ACS 5-Year Estimates for 50 states + DC | `api.census.gov` (free key) |
| **Geography Agent** | Extracts US state references from market text; aggregates by state | Regex on market questions |
| **Analysis Agent** | Feature engineering, silhouette sweep, K-Means, PCA | scikit-learn |
| **Insights Agent** | Synthesises all data into a geographic intelligence brief | Groq LLaMA-3.3-70B |

---

## Geographic Analysis

> **Important note on bettor location data**: Polymarket and Kalshi are pseudonymous platforms — neither exposes the geographic origin of bettors. The **Geography tab** maps *what geographic outcomes are being bet on* (extracted from market question text via regex), not where bettors are located.

### What the Geography tab shows
- **US choropleth** — market volume by state extracted from question text (e.g. "Will Texas…", "New York Governor…")
- **ACS poverty rate overlay** — Census poverty rates per state for context
- **Economics scatter** — State median income vs poverty rate, with bubble size = market activity and colour = unemployment
- **Side-by-side comparison** — market volume vs poverty rate for the top market-active states
- **State reference table** — ranked list of states with sample market questions

### Correlation hypothesis
States that are economically stressed (high poverty, low income) may be referenced more in prediction markets about economic outcomes, social programs, or political leadership. The ACS overlay lets you visually test this.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export GROQ_API_KEY="gsk_..."
export CENSUS_API_KEY="your_census_key"
```

Or create a `.env` file (add `python-dotenv` to requirements and call `load_dotenv()` at startup):

```
GROQ_API_KEY=gsk_...
CENSUS_API_KEY=your_census_key
```

**Getting API keys:**
- **Groq**: https://console.groq.com (free tier available)
- **Census ACS**: https://api.census.gov/data/key_signup.html (free, instant approval)
- **Polymarket**: No key required — free Gamma API

### 3. Run the app

```bash
shiny run app.py --reload
```

Then open `http://localhost:8000` in your browser.

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | Metric cards, YES probability histogram, top-20 market table |
| **Markets** | Volume bar chart (colour = YES%), scatter, category breakdown |
| **Geography** | US choropleth of market topic focus, ACS poverty overlay, income vs market activity scatter |
| **Demographics** | ACS state charts for selected metrics, income/poverty correlation scatter, full data table |
| **Clusters** | PCA scatter, silhouette sweep, cluster comparison subplots, detail cards |
| **AI Insights** | Groq-generated brief with geographic/demographic lens |

---

## Category Filtering Fix

The Polymarket Gamma API returns markets where the `category` field is frequently `null`.
PredictIQ now uses a **dual-layer filter**:
1. **Server-side**: passes `tag_slug=<category>` as a query parameter to the API
2. **Client-side**: checks both the `category` string field AND each item in the `tags[*].slug` array

This ensures the category selector reliably filters results even when the category field is null.

---

## Census ACS Variables

PredictIQ pulls the following ACS 5-Year Estimate variables for all states:

| ACS Variable | Raw Name | Derived Metric |
|---|---|---|
| `B19013_001E` | Median household income | Direct |
| `B17001_002E` + `B01003_001E` | Poverty count + population | Poverty Rate (%) |
| `B23025_005E` + `B23025_003E` | Unemployed + labor force | Unemployment Rate (%) |
| `B25064_001E` | Median gross rent | Direct |
| `B01002_001E` | Median age | Direct |
| `B15003_022E` + `B15003_001E` | Bachelor's degree + universe | Bachelor's Rate (%) |

---

## Machine Learning Details

The **Analysis Agent** performs:

1. **Feature engineering** — log-transforms for volume/liquidity; certainty score `(|YES% − 0.5| × 2)`; Shannon entropy
2. **Silhouette sweep** — tests K = 2 to min(8, N/4), selects optimal K
3. **K-Means clustering** — fits at optimal K; auto-labels clusters (High-Confidence YES/NO, Toss-Up, High-Volume Active)
4. **PCA** — reduces to 2D; reports explained variance per component

---

## Extending the App

- **Add Kalshi data**: Register for Kalshi's free API and add a `kalshi_agent` node
- **County-level ACS**: Change `for=state:*` to `for=county:*&in=state:*` for more granular geography
- **Bettor proxy via on-chain data**: For Polymarket (Polygon blockchain), on-chain wallet analysis can partially infer geographic clusters — add a `onchain_agent` node
- **Add DBSCAN**: Swap KMeans for DBSCAN in `analysis_agent` for density-based clustering
