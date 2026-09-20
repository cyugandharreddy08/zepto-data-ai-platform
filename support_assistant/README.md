# Support Assistant (`/support_assistant`)

## Overview
This module implements a grounded GenAI customer support service for Zepto. It reads 8 official company policy documents, stores dense vector embeddings in a local ChromaDB collection, routes incoming user questions using a 3-node LangGraph StateGraph, and validates JSON responses using Pydantic models.

---

## 1. RAG Pipeline Architecture

```text
[User Query via POST /ask]
             │
             ▼
   [Node 1: classify_intent]
   (Keyword Heuristic / LLM Router)
             │
     ┌───────┴────────────────┐
     ▼                        ▼
[policy_question]       [general_question]
     │                        │
     ▼                        ▼
[Node 2: retrieve_and_answer]  [Node 3: direct_answer]
┌─────────────────────────┐   ┌───────────────────────────┐
│ 1. Local Dense Embed    │   │ Polite Refusal:           │
│    (all-MiniLM-L6-v2)   │   │ "I can only answer        │
│ 2. Cosine Vector Search │   │ questions about Zepto     │
│    (ChromaDB Top-3)     │   │ policies right now."      │
│ 3. Grounded Generation  │   │ sources: [], conf: 1.0    │
│    (Templated / LLM)    │   └─────────────┬─────────────┘
└────────────┬────────────┘                 │
             │                              │
             └──────────────┬───────────────┘
                            ▼
           [Pydantic JSON Validation & Schema]
             {answer: str, sources: list, confidence: float}
                            │
                            ▼
             [HTTP 200 JSON Response]
```

### Stage-by-Stage Implementation

1. **Ingestion (`support_assistant/rag.py`)**:
   - Reads all 8 policy files in `docs/` (`doc_01.txt` through `doc_08.txt`) covering delivery timelines, returns and refunds, membership tiers, rider tracking, order cancellation, damaged items, gift cards, and support hours.
2. **Embedding (`sentence-transformers/all-MiniLM-L6-v2`)**:
   - Encodes text chunks into 384-dimensional dense vectors on the CPU locally. No external embedding API or network access is needed.
3. **Storage & Retrieval (`ChromaDB`)**:
   - Persisted at `support_assistant/chroma_db` under collection `zepto_policies`.
   - `retrieve_top_k()` uses cosine similarity to return the top 3 matching chunks with distances and metadata.
4. **Graph Orchestration (`support_assistant/graph.py`)**:
   - Built with LangGraph `StateGraph`.
   - `classify_intent`: Determines whether the query asks about a company policy or is a general query.
   - `retrieve_and_answer`: Retrieves top policy chunks from ChromaDB and builds the answer.
   - `direct_answer`: Handles general inquiries politely without retrieving documents.
5. **Output Validation (`support_assistant/models.py`)**:
   - Uses Pydantic to enforce `{ "answer": str, "sources": list, "confidence": float }`.

---

## 2. The `MOCK_LLM` Toggle

Every LLM generation step is controlled by the `MOCK_LLM` environment variable:

- **Default Mock Mode (`MOCK_LLM=1` or unset)**:
  - This is the graded baseline and requires no API key or external service.
  - `classify_intent` uses a keyword rule: if the query contains `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, or `support hours`, it routes to `policy_question`; otherwise, it routes to `general_question`.
  - ChromaDB retrieval runs for real in both modes.
  - `retrieve_and_answer` returns a deterministic templated snippet of the top chunk: `f"Based on the retrieved context: {top_chunk_snippet}"` with source document IDs and confidence `1.0`.
  - `direct_answer` returns: `"I can only answer questions about Zepto policies right now."` with sources `[]`.
- **Real LLM Mode (`MOCK_LLM=0`)**:
  - Routes and generates answers using an LLM (e.g., Groq free tier).
  - Uses the structured prompt template in `rag.py`.
  - If output fails JSON schema validation, it retries up to 2 times before falling back cleanly.

---

## 3. Structured Prompt Template
Located in `support_assistant/rag.py`:

```text
You are Zepto's official AI Policy Assistant (ROLE).
Your objective is to assist customers with accurate answers regarding Zepto delivery, returns, membership, cancellation, and support guidelines (TASK).

Here is the retrieved policy context:
{context}

STRICT CONSTRAINTS (NEGATIVE CONSTRAINTS):
1. Do NOT answer using information not present in the provided context.
2. Do NOT invent, extrapolate, or guess policy terms, fees, or timelines that are not explicitly documented.
3. If the context does not contain enough information to answer the question, state clearly: "I am sorry, but that information is not available in Zepto's documented policies."

OUTPUT FORMAT (FORMAT):
Respond with a single valid JSON object containing exactly three keys:
- "answer": string containing the policy answer.
- "sources": list of string document IDs (e.g., ["doc_01", "doc_03"]).
- "confidence": float between 0.0 and 1.0 representing answer certainty.

RESPONSE LENGTH (LENGTH):
Keep the answer concise, direct, and factual—between 2 and 4 sentences.

FEW-SHOT EXAMPLE:
User Question: What are the perks of Zepto Pass+?
Retrieved Context: [doc_03] Zepto offers three account tiers: Basic (free), Zepto Pass (INR 49 per month), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live).
Response:
{
  "answer": "Zepto Pass+ costs INR 99 per month and includes free priority delivery, 10% off select categories, and 24-hour early access to limited-time deals.",
  "sources": ["doc_03"],
  "confidence": 0.98
}
```

---

## 4. Live API Test Transcripts (`MOCK_LLM=1`)

These example calls were run against the local FastAPI test suite:

### Call 1: Policy Question (Triggers Retrieval)
```json
// POST /ask
{
  "query": "What is the delivery fee for orders under INR 149?"
}
```
**Response (200 OK):**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01"
  ],
  "confidence": 1.0
}
```

### Call 2: General Question (Routes to Direct Answer, No Retrieval)
```json
// POST /ask
{
  "query": "Can you explain quantum computing?"
}
```
**Response (200 OK):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

### Call 3: Perishable Goods Return Window
```json
// POST /ask
{
  "query": "How many days do I have to report a return for perishable grocery items?"
}
```
**Response (200 OK):**
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02"
  ],
  "confidence": 1.0
}
```

---

## 5. How to Run

### Run Local API & Web Dashboard:
```bash
# Ingest documents and start server
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
Open `http://localhost:7860` in a browser for the interactive chat interface, or `http://localhost:7860/docs` for the Swagger API docs.

### Run Automated Tests:
```bash
python -m support_assistant.test_api
```

### Run with Docker:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
