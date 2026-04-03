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
):
    """
    Decorator that retries a function with exponential backoff.
    
    Delay formula:  min(base_delay * 2^attempt, max_delay)
    With jitter:    delay *= uniform(0.5, 1.5)   # spreads burst retries
    """
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # effective_delay = kwargs.pop("_base_delay", base_delay) # runtime override    # This was an option to make base_delay modifiable
            for attempt in range(max_retries + 1):                                          # from decorated functions. I elected not to, to
                try:                                                                        # avoid surprises to future callers and reviewers.
                    return func(*args, **kwargs)                                            # This would be good at scale, if well-documented,
                except ResourceExhausted as exc:                                            # instead of reconstructing a decorated function 
                    if attempt == max_retries:                                              # per-call. Currently the complexity doesn't pay
                        raise   # give up after final attempt                               # for itself.
                                                                                            #
                    delay = min(base_delay * (2 ** attempt), max_delay)                     # delay -> effective_delay

                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    logger.warning(
                        "[backoff] %s rate limited (attempt %d/%d). Retrying in %.1fs...",
                        func.__name__, attempt + 1, max_retries, delay,
                    )

                    time.sleep(delay)
        return wrapper
    return decorator