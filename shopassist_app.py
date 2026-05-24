# =============================================================
# shopassist_app.py -- ShopAssist AI FastAPI Application
# Week 16 -- Deploying & Monitoring Agentic AI Systems
# =============================================================

import os
import time
import json
import logging
from dotenv import load_dotenv

# Load .env before any other module reads environment variables
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# =============================================================
# Structured JSON Logger
# JSON logs are parseable by Datadog, CloudWatch, GCP Logging.
# Never use print() for production logging.
# =============================================================

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in log_entry:
                log_entry[key] = value
        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = get_logger("shopassist")

# =============================================================
# Tools
# In production replace these with real DB / API calls.
# The function signature stays identical -- agent never knows.
# =============================================================

@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of a customer order by order ID."""
    mock_orders = {
        "ORD001": "Shipped -- Out for delivery, expected by end of day tomorrow.",
        "ORD002": "Processing -- Payment confirmed, being packed at warehouse.",
        "ORD003": "Delivered -- Delivered on 20th March 2026 at 2:34 PM.",
        "ORD004": "Cancelled -- Refund of Rs 2,340 initiated, 3-5 business days.",
        "ORD005": "Return Initiated -- Pickup scheduled for 25th March 2026.",
    }
    result = mock_orders.get(order_id.strip().upper())
    if result:
        return result
    return "Order ID '" + order_id + "' not found. Please verify the order ID."


@tool
def get_product_info(product_name: str) -> str:
    """Get product details including price, availability, and warranty."""
    mock_catalog = {
        "laptop":     "UltraBook Pro 14 -- Rs 75,000 -- In Stock -- 1 year warranty",
        "phone":      "SmartPhone X12 -- Rs 25,000 -- In Stock -- 6 months warranty",
        "headphones": "SoundMax Pro -- Rs 5,000 -- Out of Stock -- Restock in 1 week",
        "tablet":     "TabMax 10 -- Rs 35,000 -- In Stock -- 1 year warranty",
        "charger":    "FastCharge 65W -- Rs 1,200 -- In Stock -- 3 month warranty",
        "mouse":      "ErgoClick Wireless -- Rs 2,500 -- In Stock -- 1 year warranty",
    }
    query = product_name.strip().lower()
    for key, value in mock_catalog.items():
        if key in query:
            return value
    return "No match for '" + product_name + "'. Available: laptop, phone, headphones, tablet, charger, mouse."


# =============================================================
# Agent
# =============================================================

SYSTEM_PROMPT = (
    "You are ShopAssist, a professional customer support agent for ShopEasy,"
    " an Indian e-commerce platform selling electronics.\n\n"
    "Your responsibilities:\n"
    "- Help customers track orders using the get_order_status tool\n"
    "- Provide product information using the get_product_info tool\n"
    "- Answer general queries about shipping, returns, and warranties\n\n"
    "Your boundaries:\n"
    "- Do not discuss topics unrelated to ShopEasy products and orders\n"
    "- If you cannot help, direct the customer to call 1800-SHOPEASY\n"
    "- Always respond in the same language the customer uses\n"
    "- Be concise, professional, and empathetic"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[get_order_status, get_product_info],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

logger.info("ShopAssist agent initialised", extra={"model": "gpt-4o-mini"})

# =============================================================
# FastAPI Application
# =============================================================

app = FastAPI(
    title="ShopAssist AI",
    description="LangGraph-powered customer support agent for ShopEasy.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# Pydantic Schemas
# =============================================================

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique session identifier per customer.",
        example="customer_12345"
    )
    message: str = Field(
        ...,
        description="The customer query.",
        example="What is the status of my order ORD001?"
    )

class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model: str
    tools_available: list

SERVER_START_TIME = time.time()

# =============================================================
# Endpoints
# =============================================================

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """
    Health check endpoint used by load balancers, Render, and Cloud Run.
    Always returns 200 if the server is running.
    """
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
        model="gpt-4o-mini",
        tools_available=["get_order_status", "get_product_info"],
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Main chat endpoint. Pass the same session_id across turns
    to maintain conversation context within a session.
    """
    logger.info(
        "Incoming request",
        extra={"session_id": request.session_id, "message_length": len(request.message)}
    )
    start_time = time.time()
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        response_text = result["messages"][-1].content
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Request completed",
            extra={"session_id": request.session_id, "latency_ms": latency_ms}
        )
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(
            "Agent invocation failed",
            extra={"session_id": request.session_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail="ShopAssist is temporarily unavailable. Please try again in a moment."
        )
