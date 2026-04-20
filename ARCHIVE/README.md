# ⬡ PredictIQ — Agentic Prediction Market Intelligence Platform

A production-grade agentic workflow built with **LangGraph**, **LangChain**, and **Groq**,
delivering a live intelligence dashboard via **Shiny for Python**.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                     │
│                                                          │
│  ┌─────────────┐                                         │
│  │  SUPERVISOR  │  ← orchestrates all routing            │
│  │   (node)     │                                         │
│  └──────┬───────┘                                         │
│         │ conditional edges                               │
│    ┌────▼────┐  ┌──────┐  ┌──────────┐  ┌──────────┐    │
│    │POLYMARKET│→│ FRED │→│ ANALYSIS │→│ INSIGHTS │    │
│    │  agent   │  │agent │  │  agent   │  │  agent   │    │
│    └─────────┘  └──────┘  └──────────┘  └──────────┘    │
│    Free API     FRED API   K-Means+PCA   Groq LLaMA-3    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Shiny for Python Dashboard (5 tabs)
```

### Agents

| Agent | Role | Data Source |
|-------|------|-------------|
| **Supervisor** | Reads pipeline state, routes to next agent | Internal state |
| **Polymarket Agent** | Fetches live prediction markets | `gamma-api.polymarket.com` (free) |
| **FRED Agent** | Pulls economic indicator time-series | `api.stlouisfed.org` |
| **Analysis Agent** | Feature engineering, K-Means, PCA, silhouette sweep | scikit-learn |
| **Insights Agent** | Synthesizes all data into intelligence brief | Groq LLaMA-3.3-70B |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export GROQ_API_KEY="gsk_..."
export FRED_API_KEY="your_fred_key"
```

Or create a `.env` file:
```
GROQ_API_KEY=gsk_...
FRED_API_KEY=your_fred_key
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html  
Get a free Groq API key at: https://console.groq.com  
Polymarket requires no API key.

### 3. Run the app

```bash
shiny run app.py --reload
```

Then open `http://localhost:8000` in your browser.

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | Metric cards + probability histogram + top markets table |
| **Markets** | Top-20 volume bar chart, prob vs volume scatter, category breakdown |
| **Indicators** | FRED metric cards + time-series plots for each selected indicator |
| **Clusters** | PCA scatter, silhouette sweep, cluster comparison bars, cluster detail cards |
| **AI Insights** | Groq-generated intelligence brief: sentiment, macro context, cluster insights, signals |

---

## Machine Learning Details

The **Analysis Agent** performs:

1. **Feature Engineering** — log-transforms volume & liquidity, computes certainty score (`|YES% - 50%| × 2`)
2. **Silhouette Sweep** — tests K from 2 to min(8, N/4), picks optimal K
3. **K-Means Clustering** — clusters markets by probability, volume, liquidity, certainty
4. **PCA** — reduces to 2D for visualization; reports explained variance
5. **Auto-labeling** — assigns semantic labels (High-Confidence YES, Toss-Up, High-Volume Active, etc.)

---

## Extending the App

- **Add more agents**: Add a new node to `build_graph()` and update `supervisor_node` routing
- **Add Kalshi data**: Replace or supplement `polymarket_agent` with Kalshi's free API
- **Add Census data**: Add a `census_agent` node using `api.census.gov`
- **Add DBSCAN**: Swap KMeans for DBSCAN in `analysis_agent` for density-based clustering
- **Persistent state**: Wrap `run_graph` result in a database or file cache