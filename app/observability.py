"""
observability.py — Langfuse client initialization.
Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from environment (.env).
Import `langfuse` from this module anywhere you need to create spans/generations.
"""
from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client

langfuse = get_client()