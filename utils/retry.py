"""
Implementation for a reusable backoff decorator
"""

import time
import random
import logging
from functools import wraps
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)


def with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    verbose: bool = False,
):
    """
    Decorator that retries a function with exponential backoff.
    
    Delay formula:  min(base_delay * 2^attempt, max_delay)
    With jitter:    delay *= uniform(0.5, 1.5)   # spreads burst retries
    """
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted as exc:
                    if attempt == max_retries:
                        raise   # give up after final attempt

                    delay = min(base_delay * (2 ** attempt), max_delay)

                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    msg = (
                        f"[backoff] {func.__name__} rate_limited "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    if verbose:
                        print(msg)
                    else:
                        logger.warning(msg)

                    time.sleep(delay)
            return wrapper
        return decorator