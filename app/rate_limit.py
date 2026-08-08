"""
rate_limit.py — Shared rate limiter instance.

A single Limiter is created here so app/api.py (for wiring) and each
router (for the @limiter.limit(...) decorator) reference the same
object, keyed by client IP by default.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
