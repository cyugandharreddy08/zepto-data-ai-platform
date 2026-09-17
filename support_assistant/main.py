import os
import sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from support_assistant.models import QueryRequest, QueryResponse
from support_assistant.graph import run_query
from support_assistant.rag import ingest_corpus, get_chroma_client, COLLECTION_NAME

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analytics", "figures"))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "zepto_catalog.db"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ingest ChromaDB on startup if needed
    try:
        client = get_chroma_client()
        col = client.get_or_create_collection(COLLECTION_NAME)
        if col.count() == 0:
            print("ChromaDB collection empty, ingesting docs...")
            ingest_corpus()
        else:
            print(f"ChromaDB ready with {col.count()} documents.")
    except Exception as e:
        print(f"Startup ChromaDB check: {e}")
    yield

app = FastAPI(
    title="Zepto Data & AI Platform",
    description="Unified platform: Data Pipeline, Analytics Pipeline, and Grounded GenAI Support Assistant.",
    version="1.0.0",
    lifespan=lifespan
)

if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
def ask_policy_question(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        response = run_query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal graph error: {str(e)}")

@app.get("/api/catalog")
def get_catalog():
    if not os.path.exists(DB_PATH):
        return {"books": []}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.price_inr DESC
        LIMIT 15
    """).fetchall()
    conn.close()
    return {
        "books": [
            {
                "id": r[0], "title": r[1], "category": r[2],
                "price_gbp": r[3], "price_inr": r[4],
                "rating": r[5], "in_stock": bool(r[6])
            }
            for r in rows
        ]
    }

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zepto Data & AI Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6b21a8;
            --primary-light: #9333ea;
            --accent: #ff2e93;
            --bg: #0d0c15;
            --surface: #171526;
            --surface-card: #201c35;
            --border: #2d284a;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: linear-gradient(90deg, #3b0764, #581c87, #701a75);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 18px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .brand-badge {
            background: var(--accent);
            color: white;
            font-weight: 800;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 1.1rem;
            letter-spacing: -0.5px;
        }
        .brand h1 {
            font-size: 1.3rem;
            font-weight: 700;
        }
        .nav-links {
            display: flex;
            gap: 16px;
        }
        .nav-link {
            color: #cbd5e1;
            text-decoration: none;
            font-size: 0.9rem;
            padding: 8px 14px;
            border-radius: 6px;
            background: rgba(255,255,255,0.06);
            transition: all 0.2s;
        }
        .nav-link:hover {
            background: rgba(255,255,255,0.15);
            color: white;
        }
        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 28px 24px;
            width: 100%;
            flex: 1;
        }
        .tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }
        .tab-btn {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 10px 20px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: var(--primary-light);
            border-color: var(--primary-light);
            color: white;
            box-shadow: 0 4px 14px rgba(147, 51, 234, 0.4);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Chat UI */
        .chat-container {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
        }
        @media (max-width: 900px) { .chat-container { grid-template-columns: 1fr; } }
        .chat-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            height: 600px;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
        }
        .message.user {
            align-self: flex-end;
        }
        .message.assistant {
            align-self: flex-start;
        }
        .message-bubble {
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .message.user .message-bubble {
            background: var(--primary-light);
            color: white;
            border-bottom-right-radius: 2px;
        }
        .message.assistant .message-bubble {
            background: var(--surface-card);
            border: 1px solid var(--border);
            color: var(--text);
            border-bottom-left-radius: 2px;
        }
        .meta-badges {
            display: flex;
            gap: 8px;
            margin-top: 6px;
            font-size: 0.75rem;
        }
        .badge {
            background: rgba(255,255,255,0.08);
            padding: 3px 8px;
            border-radius: 4px;
            color: var(--text-muted);
        }
        .badge.source { color: #38bdf8; background: rgba(56, 189, 248, 0.12); }
        .badge.conf { color: var(--success); background: rgba(16, 185, 129, 0.12); }
        .chat-input-area {
            padding: 16px;
            border-top: 1px solid var(--border);
            background: rgba(0,0,0,0.2);
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: white;
            font-size: 0.95rem;
            outline: none;
        }
        .chat-input:focus { border-color: var(--primary-light); }
        .send-btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .send-btn:hover { opacity: 0.9; }
        
        .sidebar-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        .sidebar-card h3 {
            font-size: 1.05rem;
            margin-bottom: 14px;
            color: #f1f5f9;
        }
        .suggested-query {
            background: var(--surface-card);
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            color: #cbd5e1;
        }
        .suggested-query:hover {
            border-color: var(--primary-light);
            color: white;
            transform: translateX(4px);
        }
        
        /* Table & Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .card h2 { font-size: 1.25rem; margin-bottom: 16px; color: #f8fafc; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th, td {
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
        }
        th { background: rgba(255,255,255,0.03); color: var(--text-muted); font-weight: 600; }
        tr:hover td { background: rgba(255,255,255,0.02); }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
        .figure-img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-badge">ZEPTO</span>
            <h1>Data & AI Platform</h1>
        </div>
        <div class="nav-links">
            <a href="/docs" target="_blank" class="nav-link">Swagger Docs</a>
            <a href="https://github.com/cyugandharreddy08/zepto-data-ai-platform" target="_blank" class="nav-link">GitHub Repository</a>
        </div>
    </header>

    <div class="container">
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-assistant')">Module 3: GenAI Support Assistant</button>
            <button class="tab-btn" onclick="switchTab('tab-data')">Module 1: Data Pipeline</button>
            <button class="tab-btn" onclick="switchTab('tab-analytics')">Module 2: Analytics & ML</button>
        </div>

        <!-- TAB 1: Support Assistant -->
        <div id="tab-assistant" class="tab-content active">
            <div class="chat-container">
                <div class="chat-box">
                    <div class="chat-messages" id="chat-messages">
                        <div class="message assistant">
                            <div class="message-bubble">
                                Hello! I am Zepto's AI Policy Assistant grounded in official company documents. How can I help you regarding deliveries, returns, cancellations, or membership tiers today?
                            </div>
                            <div class="meta-badges">
                                <span class="badge source">System Initialized</span>
                                <span class="badge conf">Confidence 1.0</span>
                            </div>
                        </div>
                    </div>
                    <div class="chat-input-area">
                        <input type="text" id="query-input" class="chat-input" placeholder="Ask about delivery fees, cancellation window, Zepto Pass+..." onkeydown="if(event.key==='Enter') sendQuery()">
                        <button class="send-btn" onclick="sendQuery()">Ask Policy</button>
                    </div>
                </div>

                <div class="sidebar-card">
                    <h3>💡 Quick Test Queries</h3>
                    <div class="suggested-query" onclick="setQuery('What is the delivery fee for orders under INR 149?')">
                        📦 Delivery fee under INR 149?
                    </div>
                    <div class="suggested-query" onclick="setQuery('How long do I have to return perishable grocery items?')">
                        🥦 Perishable return window?
                    </div>
                    <div class="suggested-query" onclick="setQuery('What perks do I get with Zepto Pass+?')">
                        ⭐ Zepto Pass+ benefits?
                    </div>
                    <div class="suggested-query" onclick="setQuery('Can I cancel my order after it has been packed?')">
                        ❌ Cancellation after packing?
                    </div>
                    <div class="suggested-query" onclick="setQuery('What is the capital of France?')">
                        ❓ General chitchat (tests refusal router)
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Data Pipeline -->
        <div id="tab-data" class="tab-content">
            <div class="card">
                <h2>📦 Scraped Catalog Items (Normalized SQLite Database)</h2>
                <p style="color: var(--text-muted); margin-bottom: 16px;">
                    144 live books scraped from books.toscrape.com across 4 categories, normalized into 3NF SQLite (<code>zepto_catalog.db</code>) with fixed conversion: <strong>1 GBP = 105.50 INR</strong>.
                </p>
                <div style="overflow-x: auto;">
                    <table id="catalog-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Title</th>
                                <th>Category</th>
                                <th>Price (GBP)</th>
                                <th>Price (INR)</th>
                                <th>Rating</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="catalog-body">
                            <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading catalog data...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: Analytics & ML -->
        <div id="tab-analytics" class="tab-content">
            <div class="card">
                <h2>📊 Machine Learning Model Comparison</h2>
                <p style="color: var(--text-muted); margin-bottom: 16px;">
                    Benchmarked across identical 80/20 stratified splits with train-only ColumnTransformer preprocessing.
                </p>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Group</th>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>ROC AUC</th>
                                <th>MAE</th>
                                <th>R²</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Classification</td>
                                <td><strong>Logistic Regression</strong></td>
                                <td>0.8045</td>
                                <td>0.7931</td>
                                <td>0.6667</td>
                                <td>0.7244</td>
                                <td>0.8435</td>
                                <td>N/A</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Classification</td>
                                <td><strong>Decision Tree</strong></td>
                                <td>0.7877</td>
                                <td>0.8444</td>
                                <td>0.5507</td>
                                <td>0.6667</td>
                                <td>0.8210</td>
                                <td>N/A</td>
                                <td>N/A</td>
                            </tr>
                            <tr style="background: rgba(147, 51, 234, 0.1);">
                                <td>Classification</td>
                                <td><strong style="color: #c084fc;">Random Forest (Tuned) ⭐</strong></td>
                                <td><strong>0.7933</strong></td>
                                <td><strong>0.8200</strong></td>
                                <td><strong>0.5942</strong></td>
                                <td><strong>0.6891</strong></td>
                                <td><strong>0.8420</strong></td>
                                <td>N/A</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Regression</td>
                                <td><strong>Multivariate Linear Regression</strong></td>
                                <td>N/A</td>
                                <td>N/A</td>
                                <td>N/A</td>
                                <td>N/A</td>
                                <td>N/A</td>
                                <td>20.8094</td>
                                <td>0.3999</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>ROC Curve Benchmark</h2>
                    <img src="/figures/roc_curves.png" alt="ROC Curves" class="figure-img">
                </div>
                <div class="card">
                    <h2>Multivariate Story</h2>
                    <img src="/figures/multivariate_story.png" alt="Multivariate Story" class="figure-img">
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'tab-data') loadCatalog();
        }

        function setQuery(text) {
            document.getElementById('query-input').value = text;
            sendQuery();
        }

        async function sendQuery() {
            const input = document.getElementById('query-input');
            const q = input.value.trim();
            if (!q) return;

            const chat = document.getElementById('chat-messages');
            
            // Add User message
            chat.innerHTML += `
                <div class="message user">
                    <div class="message-bubble">${q}</div>
                </div>
            `;
            input.value = '';
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q })
                });
                const data = await res.json();
                
                const sourcesHtml = data.sources && data.sources.length 
                    ? data.sources.map(s => `<span class="badge source">${s}</span>`).join(' ')
                    : '<span class="badge">General Intent</span>';
                
                chat.innerHTML += `
                    <div class="message assistant">
                        <div class="message-bubble">${data.answer}</div>
                        <div class="meta-badges">
                            ${sourcesHtml}
                            <span class="badge conf">Confidence ${(data.confidence * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                `;
            } catch(e) {
                chat.innerHTML += `
                    <div class="message assistant">
                        <div class="message-bubble" style="color: #ef4444;">Error connecting to LangGraph service: ${e.message}</div>
                    </div>
                `;
            }
            chat.scrollTop = chat.scrollHeight;
        }

        async function loadCatalog() {
            try {
                const res = await fetch('/api/catalog');
                const data = await res.json();
                const tbody = document.getElementById('catalog-body');
                if (!data.books.length) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No books in database.</td></tr>';
                    return;
                }
                tbody.innerHTML = data.books.map(b => `
                    <tr>
                        <td>${b.id}</td>
                        <td>${b.title}</td>
                        <td>${b.category}</td>
                        <td>£${b.price_gbp.toFixed(2)}</td>
                        <td>₹${b.price_inr.toFixed(2)}</td>
                        <td>⭐ ${b.rating}/5</td>
                        <td><span style="color: #10b981; font-weight: 600;">In Stock</span></td>
                    </tr>
                `).join('');
            } catch(e) {
                console.error(e);
            }
        }
    </script>
</body>
</html>
"""
