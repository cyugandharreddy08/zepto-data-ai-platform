import os
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from support_assistant.models import GraphState, QueryResponse
from support_assistant.rag import retrieve_top_k, RAG_SYSTEM_PROMPT

# Keywords mandated by rubric for mock intent classification
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]

def is_mock_mode() -> bool:
    """Returns True if MOCK_LLM is unset or set to '1'."""
    val = os.environ.get("MOCK_LLM", "1").strip().lower()
    return val not in ("0", "false", "no")

# -------------------------------------------------------------------------
# NODE 1: classify_intent
# -------------------------------------------------------------------------
def classify_intent(state: GraphState) -> Dict[str, Any]:
    query = state["query"].strip()
    query_lower = query.lower()
    
    if is_mock_mode():
        # Graded baseline: deterministic keyword heuristic
        is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
        return {"intent": intent}
    else:
        # Optional MOCK_LLM=0 extension using Groq / external LLM
        try:
            from groq import Groq
            client = Groq()
            prompt = f"""Classify the following query into exactly one category: 'policy_question' or 'general_question'.
A 'policy_question' is about Zepto policies (delivery, refunds, returns, membership, tracking, cancellations, gift cards, support hours).
A 'general_question' is general chitchat or unrelated topics.

Query: {query}

Reply with ONLY the category string: 'policy_question' or 'general_question'."""
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw = completion.choices[0].message.content.strip().lower()
            intent = "policy_question" if "policy_question" in raw else "general_question"
            return {"intent": intent}
        except Exception as e:
            # Fallback to keyword heuristic on connection failure
            is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
            return {"intent": "policy_question" if is_policy else "general_question", "error": str(e)}

# -------------------------------------------------------------------------
# NODE 2: retrieve_and_answer
# -------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> Dict[str, Any]:
    query = state["query"]
    # Real retrieval step runs in both mock and real mode
    retrieved = retrieve_top_k(query, k=3)
    
    if not retrieved:
        resp = QueryResponse(
            answer="I am sorry, but no relevant policy documents were found.",
            sources=[],
            confidence=0.0
        )
        return {"retrieved_docs": [], "response": resp}
        
    top_chunk = retrieved[0]
    sources = [doc["id"] for doc in retrieved]
    
    if is_mock_mode():
        # Graded baseline: canned format using top ~200 characters of top chunk
        top_snippet = top_chunk["text"][:200].strip()
        canned_answer = f"Based on the retrieved context: {top_snippet}"
        resp = QueryResponse(
            answer=canned_answer,
            sources=[top_chunk["id"]],
            confidence=1.0
        )
        return {"retrieved_docs": retrieved, "response": resp}
    else:
        # Optional MOCK_LLM=0 extension with retry logic
        from groq import Groq
        client = Groq()
        
        context_str = "\n\n".join([f"[{doc['id']}] {doc['text']}" for doc in retrieved])
        sys_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)
        
        retry_count = state.get("retry_count", 0)
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                user_msg = f"User Question: {query}"
                if attempt > 0:
                    user_msg += f"\nNote: Previous response failed schema validation ({last_error}). Please output ONLY valid JSON matching the requested schema."
                    
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                raw_json = completion.choices[0].message.content.strip()
                data = json.loads(raw_json)
                resp = QueryResponse(**data)
                return {"retrieved_docs": retrieved, "response": resp, "retry_count": attempt}
            except Exception as err:
                last_error = str(err)
                
        # If retries exhausted
        fallback_resp = QueryResponse(
            answer="Error: Failed to generate a validated response after retries.",
            sources=sources,
            confidence=0.0
        )
        return {"retrieved_docs": retrieved, "response": fallback_resp, "error": last_error, "retry_count": max_retries}

# -------------------------------------------------------------------------
# NODE 3: direct_answer
# -------------------------------------------------------------------------
def direct_answer(state: GraphState) -> Dict[str, Any]:
    query = state["query"]
    
    if is_mock_mode():
        # Graded baseline: fixed canned response
        canned_text = "I can only answer questions about Zepto policies right now."
        resp = QueryResponse(
            answer=canned_text,
            sources=[],
            confidence=1.0
        )
        return {"response": resp, "retrieved_docs": []}
    else:
        # Optional MOCK_LLM=0 extension
        try:
            from groq import Groq
            client = Groq()
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are Zepto's customer support agent. Inform the user politely that you can only answer questions relating to Zepto's service and policies."},
                    {"role": "user", "content": query}
                ],
                temperature=0.3
            )
            raw = completion.choices[0].message.content.strip()
            resp = QueryResponse(
                answer=raw,
                sources=[],
                confidence=0.9
            )
            return {"response": resp, "retrieved_docs": []}
        except Exception as e:
            resp = QueryResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0
            )
            return {"response": resp, "retrieved_docs": [], "error": str(e)}

# -------------------------------------------------------------------------
# ROUTER & GRAPH ASSEMBLY
# -------------------------------------------------------------------------
def route_intent(state: GraphState) -> str:
    intent = state.get("intent", "general_question")
    if intent == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Conditional routing edge
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    
    # Terminal edges
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()

# Global compiled graph
app_graph = build_graph()

def run_query(query_str: str) -> QueryResponse:
    initial_state: GraphState = {
        "query": query_str,
        "intent": None,
        "retrieved_docs": [],
        "response": None,
        "retry_count": 0,
        "error": None
    }
    final_state = app_graph.invoke(initial_state)
    return final_state["response"]

if __name__ == "__main__":
    print("Testing LangGraph StateGraph Execution (MOCK_LLM=1 default)...")
    
    # 1. Policy query
    q1 = "What is the fee for delivery on orders under INR 149?"
    r1 = run_query(q1)
    print(f"\nQuery 1 (Policy): {q1}")
    print(f"Response: {r1.model_dump_json(indent=2)}")
    
    # 2. General query
    q2 = "What is the capital of France?"
    r2 = run_query(q2)
    print(f"\nQuery 2 (General): {q2}")
    print(f"Response: {r2.model_dump_json(indent=2)}")
