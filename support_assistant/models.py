from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question to be answered", example="What is the delivery fee for orders under INR 149?")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The grounded response or refusal message")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs used (empty for general questions)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    response: Optional[QueryResponse]
    retry_count: int
    error: Optional[str]
