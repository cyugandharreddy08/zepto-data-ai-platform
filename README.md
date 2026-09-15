# Zepto Data & AI Platform

**Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning**  
*Submitted as a unified end-to-end platform comprising Data Engineering, Machine Learning Analytics, and Grounded GenAI.*

---

## 1. Project Overview & Architecture

As an incoming AI/ML engineer on Zepto’s analytics guild, this capstone delivers a single connected platform spanning the full engineering spectrum:
1. **Data Pipeline (`/data_pipeline`) [25 Marks]**: Automated catalog scraping, cleaning, currency normalization (1 GBP = 105.50 INR), relational 3NF SQLite database storage, analytical SQL queries, and relational algebra parity verification (`pd.read_sql` vs `pd.merge`).
2. **Analytics Pipeline (`/analytics`) [50 Marks]**: End-to-end demographic profiling, missing value threshold enforcement, IQR outlier detection, skewness determination, strict 6x6 correlation matrix, multivariate visual data storytelling, stratified train-only preprocessing pipeline, 3-way classifier benchmarking, SMOTE imbalance analysis, GridSearchCV tuning with Out-of-Bag (OOB) score, fare regression with heteroscedasticity diagnosis, and serialized pipeline deployment.
3. **Support Assistant (`/support_assistant`) [25 Marks]**: Grounded GenAI policy assistant powered by LangGraph, ChromaDB, and FastAPI. Ingests 8 official Zepto policy documents, routes query intents via graph state, retrieves context with local dense embeddings (`all-MiniLM-L6-v2`), validates JSON responses via Pydantic schemas, and serves via a containerized Docker application.

---

## 2. Repository Structure

```text
d:/capstone/
├── .gitignore
├── requirements.txt                    # Consolidated project dependencies
├── README.md                           # Root system documentation
├── data_pipeline/
│   ├── README.md                       # Module 1 documentation
│   ├── pipeline.py                     # Scraping, ETL, SQLite loading & SQL queries
│   ├── data_pipeline.ipynb             # Interactive ETL notebook
│   ├── queries.sql                     # Analytical SQL queries
│   └── zepto_catalog.db                # Persisted normalized SQLite database
├── analytics/
│   ├── README.md                       # Module 2 documentation & interpretation
│   ├── titanic.csv                     # Committed single-load offline fallback dataset
│   ├── 01_eda.py                       # Profiling, cleaning, IQR, skewness, 6x6 corr, & charts
│   ├── 01_eda.ipynb                    # Interactive EDA notebook
│   ├── 02_modeling.py                  # Modeling, imbalance, tuning, regression, & serialization
│   ├── 02_modeling.ipynb               # Interactive modeling notebook
│   ├── best_pipeline.joblib            # Serialized end-to-end fitted scikit-learn pipeline
│   └── figures/                        # Supporting visualization artifacts
│       ├── age_fare_distribution.png
│       ├── correlation_heatmap.png
│       ├── multivariate_story.png
│       ├── standardization_check.png
│       ├── decision_tree.png
│       ├── roc_curves.png
│       └── residual_plot.png
└── support_assistant/
    ├── README.md                       # Module 3 documentation & API transcripts
    ├── Dockerfile                      # Containerization recipe (port 7860)
    ├── main.py                         # FastAPI service with POST /ask
    ├── graph.py                        # 3-node LangGraph StateGraph & intent router
    ├── rag.py                          # ChromaDB ingestion, prompt template & retrieval
    ├── models.py                       # Pydantic request/response schemas
    ├── test_api.py                     # Automated endpoint test suite
    ├── docs/                           # 8 official Zepto policy documents
    │   ├── doc_01.txt ... doc_08.txt
    └── chroma_db/                      # Persisted local ChromaDB vector store
```

---

## 3. Environment Setup

A single consolidated `requirements.txt` is provided at the repository root.

```bash
# 1. Clone repository and navigate to root
git clone <YOUR_REPO_URL>
cd capstone

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Install consolidated dependencies
pip install -r requirements.txt
```

---

## 4. How to Run Each Module End to End

### Module 1: Data Pipeline
```bash
# Execute ETL, database creation, SQL queries, and pandas equivalence check
python data_pipeline/pipeline.py

# Or launch Jupyter Notebook:
jupyter notebook data_pipeline/data_pipeline.ipynb
```

### Module 2: Analytics Pipeline
```bash
# Step 1: Run Profiling, Cleaning, and Visual Data Story (Part A)
python analytics/01_eda.py

# Step 2: Run Predictive Modeling, Tuning, Regression, and Pipeline Export (Part B)
python analytics/02_modeling.py

# Or launch notebooks in order:
# 1. analytics/01_eda.ipynb
# 2. analytics/02_modeling.ipynb
```

### Module 3: Support Assistant
```bash
# Ingest policy corpus into local ChromaDB
python -m support_assistant.rag

# Start the FastAPI service locally (MOCK_LLM=1 default graded baseline)
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860

# Run automated endpoint tests (demonstrating policy & general queries)
python -m support_assistant.test_api

# Run containerized via Docker:
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```

---

## 5. Summary of Design Decisions by Module

### Module 1 — Data Pipeline (`/data_pipeline`)
- **Scraping Scope**: Scraped 144 items across 4 categories (`Travel`, `Mystery`, `Historical Fiction`, `Sequential Art`) from `http://books.toscrape.com/`, comfortably exceeding the $\ge 60$ products requirement.
- **Cleaning & Imputation**: Stripped currency glyphs via regex `[^\d.]` to create `price_gbp`. If a price is corrupted or missing, it is imputed with the **category median price** to avoid skewing price distributions. Text star ratings (`One`..`Five`) are deterministically mapped to integers 1–5. Availability is parsed to boolean `in_stock`.
- **Fixed Currency Baseline**: Converted `price_gbp` to `price_inr` using the project's exact constant baseline: **1 GBP = 105.50 INR**.
- **Relational Normalization**: Implemented a 3NF two-table schema with primary/foreign keys (`categories` and `books`).
- **Relational Algebra Parity**: Wrote and executed 5 analytical SQL queries covering `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and an inner `JOIN`. Proved that `pd.read_sql` and in-memory `pd.merge` yield identical data structures down to floating-point precision.

### Module 2 — Analytics Pipeline (`/analytics`)
- **Offline Fallback Architecture**: Loaded `sns.load_dataset('titanic')` **exactly once** and saved `analytics/titanic.csv`. Both EDA and modeling consume this single committed CSV, preventing network failure during evaluation.
- **Percentage-Based Missing Value Threshold Rule**:
  - `< 5%` (`embarked` at 0.22%): Dropped the 2 rows without loss of statistical power.
  - `5% - 30%` (`age` at 19.87%): Imputed using median age (28.0 years) to maintain central tendency.
  - `> 30%` (`deck` at 77.22%): Dropped column entirely to prevent massive synthetic bias.
- **Central Tendency & Outliers**: Calculated IQR outlier counts for `age` (65 outliers) and `fare` (114 outliers). Proved that `fare` is **right-skewed** because $\text{Mean (32.10)} > \text{Median (14.45)} > \text{Mode (8.05)}$.
- **Correlation Matrix**: Strict 6-feature subset (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), strictly excluding derived boolean columns (`adult_male`, `alone`). Identified top correlations: `pclass` & `fare` ($r = -0.5482$) and `sibsp` & `parch` ($r = 0.4145$).
- **Visual Data Story**: Created 4 distinct multi-panel figures exploring class privilege, fare distribution, age demographics, and family size survival peaks.
- **Train-Only Preprocessing**: Enforced zero data leakage using `ColumnTransformer` (median imputer + `StandardScaler` for numerics; mode imputer + `OneHotEncoder` for categoricals) fit strictly on the 80% stratified training split.
- **Classifier Benchmark & Tuning**: Trained Logistic Regression, Decision Tree (visualized with `plot_tree`), and Random Forest. Tuned Random Forest via `GridSearchCV` achieving an **OOB score of 0.8272** and test ROC AUC of **0.8420**. Recommended Random Forest for production deployment.
- **Imbalance & Regression Side-Tasks**: Compared baseline, `class_weight='balanced'`, and SMOTE on training fold. Fitted multivariate linear regression for ticket fare ($R^2 = 0.3999$) and identified distinct **heteroscedasticity** via residual funneling.
- **Serialization**: Exported full pipeline as `best_pipeline.joblib` and confirmed end-to-end inference on raw input dictionaries.

### Module 3 — Support Assistant (`/support_assistant`)
- **Corpus & Local Embeddings**: Stored 8 official Zepto policy documents in `docs/` and generated dense vectors locally using `sentence-transformers/all-MiniLM-L6-v2` into a persistent ChromaDB vector store.
- **Structured Prompt Template**: Formulated following the Role–Context–Task–Format–Length skeleton with explicit negative constraints (*"do not answer using information not present in the provided context"*) and few-shot JSON exemplars.
- **LangGraph StateGraph**: Orchestrated via 3 nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) with a conditional routing edge.
- **Dual Execution Modes (`MOCK_LLM`)**:
  - `MOCK_LLM=1` (Default Graded Baseline): Runs 100% offline without API keys. Keyword-based intent classification, real cosine similarity retrieval from ChromaDB, and deterministic templated responses.
  - `MOCK_LLM=0` (Optional Extension): Dispatches to external LLMs (e.g. Groq free tier) with 2-stage JSON validation retry logic.
- **Production Delivery**: Validated output using Pydantic (`QueryResponse`), served via FastAPI `POST /ask`, and packaged with a production-ready `Dockerfile`.

---

## 6. Git Workflow Verification

As required by the capstone grading rubric, the repository history demonstrates feature branch development (`feature/zepto-platform`), multiple independent commits per module, and a formal merge commit back into `main`:

```text
*   ecef18e Merge branch 'feature/zepto-platform' into main - Zepto Data & AI Platform
|\  
| * 0a99d94 feat(support_assistant): implement RAG support service with LangGraph, ChromaDB, FastAPI, and Dockerfile
| * d9085a1 feat(analytics): complete EDA, visual data story, ML modeling pipeline, and serialized model
| * 4426249 feat(data_pipeline): complete raw-to-relational ETL catalog pipeline with SQLite and SQL queries
|/  
* b527215 chore: initial project scaffolding and dependencies
```
*(Verified via `git log --graph --all --oneline`)*.

---

## 7. Compliance Checklist

- [x] Exactly ONE public repository containing `/data_pipeline`, `/analytics`, `/support_assistant`, and root `README.md`.
- [x] Git feature branch created, committed to at least twice, and merged back into `main`.
- [x] Module 1: Scraped $\ge 60$ books (144 items) across 4 categories, cleaned types, applied fixed rate (1 GBP = 105.50 INR), 3NF SQLite schema, 5 SQL queries with JOIN, and Pandas merge equivalence.
- [x] Module 2: Offline fallback `titanic.csv`, threshold missing value handling, IQR outlier counts, skewness ordering, 6x6 correlation matrix, 4 visual charts with interpretations, stratified train-only Pipeline, 3 classifiers, SMOTE imbalance analysis, GridSearchCV tuning with OOB score, fare regression with heteroscedasticity plot, comparison table, and saved pipeline.
- [x] Module 3: 8 policy documents, ChromaDB vector store, structured prompt skeleton, 3-node LangGraph StateGraph, deterministic mock baseline (`MOCK_LLM=1`), Pydantic validation & retry, FastAPI `POST /ask` endpoint, Dockerfile, and architecture documentation with live call transcripts.
