# LLM Cost Optimizer

An intelligent middleware system that optimizes LLM API costs through a multi-stage pipeline: **Prompt Optimization → Semantic Caching → Model Selection → Request Batching → LLM Execution**.

## How It Works

```
User Query → [Prompt Optimizer] → [Semantic Cache] → [Model Selector] → [Batcher] → [LLM] → Response
                  ↓                     ↓                  ↓                ↓           ↓
              Clean &              FAISS lookup         Pick cheapest     Group by    Gemini API
              shorten              for similar          model meeting     model &     or simulate
              + analyze            past answers         requirements      thresholds
```

### Pipeline Stages

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1. Prompt Optimization | `prompt_optimizer/` | Cleans filler words, shortens via rules/LLM, classifies intent & complexity |
| 2. Semantic Cache | `cache/` | FAISS vector search for similar past queries — returns cached response if match found |
| 3. Model Selection | `batching/model_selector.py` | Picks the cheapest LLM model that meets the query's requirements |
| 4. Request Batching | `batching/batcher.py` | Groups requests by model with adaptive time/size/token thresholds |
| 5. LLM Execution | `llm/service.py` | Calls Gemini API (or simulates) and extracts metrics |
| 6. Cache Store | `cache/manager.py` | Saves the response for future cache hits |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (simulation mode works by default)
```

### 3. Start the API Server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Send a Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning in simple terms"}'
```

### 5. Launch Dashboard (Optional)

```bash
streamlit run streamlit_app.py
```

## Project Structure

```
├── main.py                 ← FastAPI entry point
├── config.py               ← Unified configuration
├── models.py               ← Pydantic data models
├── streamlit_app.py        ← Monitoring dashboard
├── requirements.txt        ← Dependencies
├── .env.example            ← Environment template
│
├── prompt_optimizer/       ← Stage 1: Clean, shorten, analyze prompts
│   ├── cleaner.py          ← Remove filler words, normalize whitespace
│   ├── shortener.py        ← Rule-based or LLM-powered shortening
│   ├── analyzer.py         ← Intent, complexity, latency classification
│   └── tokenizer.py        ← Token counting via tiktoken
│
├── cache/                  ← Stage 2: FAISS semantic cache
│   ├── manager.py          ← Cache manager with FAISS index
│   ├── embedding_service.py← Gemini embeddings or simulation
│   └── policy.py           ← Cache decision & eviction scoring
│
├── batching/               ← Stage 3+4: Model selection & request batching
│   ├── model_catalog.py    ← 15 models, 6 providers, real pricing
│   ├── model_selector.py   ← Cost-optimized model selection
│   ├── batcher.py          ← Model-wise batch lifecycle
│   └── policy.py           ← Adaptive batching thresholds
│
├── llm/                    ← Stage 5: LLM execution
│   └── service.py          ← Gemini API + simulation fallback
│
└── pipeline/               ← Orchestration
    ├── orchestrator.py      ← Central pipeline (connects everything)
    └── tracker.py           ← Per-query tracking & metrics
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/query` | Process query through pipeline |
| `GET` | `/metrics` | System-wide metrics |
| `GET` | `/recent-queries` | Recent query history |
| `GET` | `/cache/stats` | Cache statistics |
| `GET` | `/cache/entries` | View cached entries |
| `GET` | `/cache/evictions` | Eviction history |
| `POST` | `/cache/clear` | Clear cache |
| `GET` | `/batching/stats` | Batching statistics |
| `GET` | `/config` | Current configuration |
| `POST` | `/clear-all` | Reset everything |

## Simulation Mode

By default, both `SIMULATE_LLM` and `SIMULATE_EMBEDDINGS` are `true`. This means:
- No API keys required
- Deterministic embeddings (hash-based) for cache testing
- Template responses for LLM calls
- Full pipeline works end-to-end for development and demos

To use real APIs, set in `.env`:
```
SIMULATE_LLM=false
SIMULATE_EMBEDDINGS=false
GEMINI_API_KEY=your_key_here
```

## Model Catalog

15 models across 6 providers with real pricing:

| Model | Provider | Input $/1M | Output $/1M | Best For |
|-------|----------|-----------|------------|----------|
| deepseek-chat | DeepSeek | $0.14 | $0.28 | Budget general |
| gemini-2.0-flash | Google | $0.10 | $0.40 | Fast & cheap |
| gpt-4o-mini | OpenAI | $0.15 | $0.60 | Balanced budget |
| gemini-2.5-flash | Google | $0.15 | $0.60 | Quality + speed |
| llama-4-scout | Meta | $0.11 | $0.34 | Ultra-fast inference |
| gpt-4o | OpenAI | $2.50 | $10.00 | High quality |
| claude-3.5-sonnet | Anthropic | $3.00 | $15.00 | Premium coding |
| gemini-2.5-pro | Google | $1.25 | $10.00 | Top reasoning |
| gpt-4.1 | OpenAI | $2.00 | $8.00 | Premium general |

## Tech Stack

- **FastAPI** — API server
- **FAISS** — Vector similarity search
- **Gemini API** — LLM + Embeddings
- **Pydantic** — Data validation
- **Streamlit + Plotly** — Dashboard
- **tiktoken** — Token counting