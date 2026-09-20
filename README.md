# Zepto Data & AI Platform — Capstone Submission

**Author:** Yugandhar Reddy (`cyugandharreddy08`)  
**Certificate Program in Artificial Intelligence & Machine Learning**  
**Repository:** [https://github.com/cyugandharreddy08/zepto-data-ai-platform](https://github.com/cyugandharreddy08/zepto-data-ai-platform)

---

## Project Overview

This repository is my submission for the Zepto Data & AI Platform capstone. It combines three core areas of practical AI/ML engineering into one unified project:
1. **Data Pipeline (`/data_pipeline`)**: A web scraper and ETL script that collects product catalog data, cleans and normalizes types, applies currency conversion, loads it into a SQLite database with foreign keys, and runs analytical SQL queries.
2. **Analytics Pipeline (`/analytics`)**: An exploratory data analysis and predictive modeling workflow using customer demographic data. Covers missing value threshold handling, outlier detection, statistical skewness, a 6x6 correlation matrix, visual data storytelling, stratified train-only preprocessing, classifier comparisons, hyperparameter tuning with Out-of-Bag (OOB) scoring, and fare regression.
3. **Support Assistant (`/support_assistant`)**: A policy support service built on LangGraph, ChromaDB, and FastAPI. Ingests 8 official Zepto policy documents, uses a 3-node state machine to classify intent and retrieve relevant policy excerpts using local dense embeddings, validates outputs using Pydantic, and serves requests via a Docker container and FastAPI REST endpoint.

---

## Directory Structure

```text
.
├── .gitignore
├── requirements.txt                    # Single consolidated dependencies file
├── README.md                           # Main submission documentation
│
├── data_pipeline/
│   ├── README.md                       # Detailed pipeline documentation
│   ├── pipeline.py                     # Main scraper, ETL, and SQL query script
│   ├── data_pipeline.ipynb             # Interactive Jupyter notebook
│   ├── queries.sql                     # SQL query scripts
│   └── zepto_catalog.db                # SQLite database (auto-generated)
│
├── analytics/
│   ├── README.md                       # Full EDA writeup, story & model comparison
│   ├── titanic.csv                     # Committed offline fallback dataset
│   ├── 01_eda.py                       # Profiling, cleaning, IQR, & charts
│   ├── 01_eda.ipynb                    # Step-by-step EDA notebook with charts
│   ├── 02_modeling.py                  # Classifiers, tuning, regression, & joblib export
│   ├── 02_modeling.ipynb               # Step-by-step modeling notebook
│   ├── best_pipeline.joblib            # Serialized fitted pipeline
│   └── figures/                        # Saved PNG plots
│       ├── age_fare_distribution.png
│       ├── correlation_heatmap.png
│       ├── multivariate_story.png
│       ├── standardization_check.png
│       ├── decision_tree.png
│       ├── roc_curves.png
│       └── residual_plot.png
│
└── support_assistant/
    ├── README.md                       # RAG architecture & sample call transcripts
    ├── Dockerfile                      # Production container build
    ├── main.py                         # FastAPI web app & interactive dashboard
    ├── graph.py                        # LangGraph StateGraph intent router
    ├── rag.py                          # ChromaDB vector store & prompt template
    ├── models.py                       # Pydantic schemas (QueryRequest, QueryResponse)
    ├── test_api.py                     # Automated endpoint verification
    ├── docs/                           # 8 policy documents (doc_01 to doc_08)
    └── chroma_db/                      # Local ChromaDB vector database
```

---

## Setup & Installation

I created a single consolidated `requirements.txt` file in the root directory. To set up the environment:

```bash
# 1. Clone repository
git clone https://github.com/cyugandharreddy08/zepto-data-ai-platform.git
cd zepto-data-ai-platform

# 2. (Optional) Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run Each Module

### Module 1: Data Pipeline
To run the web scraper, database loader, and SQL queries:
```bash
python data_pipeline/pipeline.py
```
Or open and execute `data_pipeline/data_pipeline.ipynb` in Jupyter.

### Module 2: Analytics Pipeline
Run the EDA and modeling scripts in order:
```bash
# Part A: Profiling, cleaning, IQR outliers, skewness, and charts
python analytics/01_eda.py

# Part B: Modeling, imbalance checks, tuning, regression, and pipeline export
python analytics/02_modeling.py
```
Or open and run `analytics/01_eda.ipynb` followed by `analytics/02_modeling.ipynb`.

### Module 3: Support Assistant
To run the automated API verification:
```bash
python -m support_assistant.test_api
```
To start the local web application and dashboard on port 7860:
```bash
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
Then open `http://localhost:7860` in your browser to view the interactive chat and dashboard, or `http://localhost:7860/docs` for the Swagger API docs.

To run with Docker:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```

---

## Design Decisions by Module

### 1. Data Pipeline (`/data_pipeline`)
- **Scraping Scope**: I targeted `books.toscrape.com` across 4 categories (`Travel`, `Mystery`, `Historical Fiction`, `Sequential Art`), scraping 144 items to exceed the required 60 items.
- **Cleaning & Imputation**: 
  - Stripped non-numeric characters from prices to create `price_gbp`. If any price was missing or unparseable, I used category median imputation to keep realistic price distributions.
  - Converted star ratings (`One` to `Five`) to integers 1 through 5.
  - Parsed availability text to integer/boolean `in_stock` (1 if "in stock" was present, 0 otherwise).
  - Dropped any rows lacking a title since title is a primary item identifier.
- **Fixed Currency Baseline**: Calculated `price_inr = round(price_gbp * 105.50, 2)` using the required fixed project baseline: **1 GBP = 105.50 INR**.
- **Relational Schema**: Implemented two normalized tables in `zepto_catalog.db`:
  - `categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE)`
  - `books(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))`
- **SQL & Pandas Equivalence**: Wrote 5 SQL queries covering `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and an inner `JOIN`. I verified that querying with `pd.read_sql` and joining in memory with `pd.merge` produced identical results using `pd.testing.assert_frame_equal`.

### 2. Analytics Pipeline (`/analytics`)
- **Single-Load Offline Fallback**: Loaded the dataset once via `sns.load_dataset('titanic')` and saved it to `analytics/titanic.csv`. Both `01_eda` and `02_modeling` load from this saved file so grading runs offline without network dependence.
- **Missing-Value Handling**:
  - `deck` (77.22% missing): Dropped the column entirely. Imputing over 77% missing data would inject too much synthetic noise.
  - `age` (19.87% missing): Imputed with the median age (28.0 years) because 19.87% falls between 5% and 30%.
  - `embarked` and `embark_town` (0.22% missing, 2 rows): Dropped these 2 rows because the missingness is well under 5%.
- **Univariate Analysis & Outliers**:
  - Using the IQR rule, `age` had 65 outliers (bounds [2.50, 54.50]) and `fare` had 114 outliers (bounds [-26.76, 65.66]).
  - Fare central tendencies: Mean (32.10) > Median (14.45) > Mode (8.05). Because Mean > Median > Mode, the fare distribution is positively right-skewed with a long tail of luxury tickets.
- **Bivariate & Correlation**:
  - Survival rates by boolean masking: Females (74.04%) vs Males (18.89%); 1st class (62.62%), 2nd class (47.28%), 3rd class (24.24%).
  - Computed a 6x6 correlation matrix restricted to numeric features: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (excluding derived booleans `adult_male` and `alone`).
  - Top 2 off-diagonal correlations:
    1. `pclass` & `fare` ($r = -0.5482$): Inversely coded class numbers mean higher classes had much higher fares.
    2. `sibsp` & `parch` ($r = 0.4145$): Passengers traveling with siblings/spouses also tended to travel with parents/children.
- **Visual Data Story**: Created 4 charts showing survival by sex/class, fare distributions, age vs fare, and family size survival trends.
- **Preprocessing & Model Evaluation**:
  - Used an 80/20 stratified split because the target has a 61.6% to 38.4% imbalance, ensuring both train and test partitions maintain identical proportions.
  - Built a `ColumnTransformer` with median imputation and `StandardScaler` for numeric features, and mode imputation with `OneHotEncoder` for categoricals. Fit strictly on the training partition to eliminate data leakage.
  - Trained Logistic Regression, Decision Tree (visualized via `plot_tree`), and Random Forest.
  - Compared baseline vs `class_weight='balanced'` vs SMOTE (applied to train fold only). Balanced weights gave the best precision-recall balance without creating synthetic points.
  - Tuned Random Forest via `GridSearchCV` (5-fold CV) finding best parameters `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 200}`, giving a test ROC AUC of 0.8420 and an Out-of-Bag (OOB) score of 0.8272.
  - Linear regression on fare yielded $R^2 = 0.3999$ with a residual plot showing clear heteroscedasticity (residual variance widening as predicted fare increases).
  - Recommended Random Forest for deployment due to high ensemble stability and strong OOB generalization.
  - Saved the complete fitted pipeline to `best_pipeline.joblib` and confirmed predictions work directly on raw dictionary inputs.

### 3. Support Assistant (`/support_assistant`)
- **Corpus & Vector Storage**: Loaded the 8 Zepto policy text documents from `docs/` and indexed them locally in ChromaDB (`chroma_db/`) using `sentence-transformers/all-MiniLM-L6-v2`. Runs entirely offline on CPU.
- **Structured Prompt**: Formatted with Role-Context-Task-Format-Length, an explicit negative constraint ("do not answer using information not present in the provided context"), and a few-shot JSON example.
- **LangGraph StateGraph**:
  - `classify_intent`: Checks for keywords (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`) to route to `policy_question` or `general_question`.
  - `retrieve_and_answer`: Runs real cosine retrieval from ChromaDB (top 3 chunks). In default mock mode (`MOCK_LLM=1`), returns a deterministic templated excerpt of the top chunk. If `MOCK_LLM=0`, sends grounded context to the LLM.
  - `direct_answer`: In mock mode, returns a polite canned refusal for non-policy questions.
- **Schema & API**: Defined Pydantic schema `QueryResponse(answer, sources, confidence)`. Packaged with FastAPI (`POST /ask`), verified with test client, and containerized via `Dockerfile`.

---

## Git Workflow

My git commit history reflects feature branch development, multiple commits, and a merge commit into `main`:

```text
* 6d21404 docs: add comprehensive root README with setup, execution guides, and design decisions
*   ecef18e Merge branch 'feature/zepto-platform' into main - Zepto Data & AI Platform
|\  
| * 0a99d94 feat(support_assistant): implement RAG support service with LangGraph, ChromaDB, FastAPI, and Dockerfile
| * d9085a1 feat(analytics): complete EDA, visual data story, ML modeling pipeline, and serialized model
| * 4426249 feat(data_pipeline): complete raw-to-relational ETL catalog pipeline with SQLite and SQL queries
|/  
* b527215 chore: initial project scaffolding and dependencies
```
*(Verified using `git log --graph --all --oneline`)*.
