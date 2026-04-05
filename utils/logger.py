import os
import json
import threading
from datetime import datetime, timezone


#
# Structured run logger
#

class RunLogger:
    """
    Writes a structured JSON log for each agent run.
    
    Each log entry captures:
        - timestamp, prompt, total tool calls, tokens used,
        per-step tool names and their results (truncated),
        and whether the run completed or hit the iteration cap.
        
    This makes task-completion reliability and tool-use patterns
    verifiable from the log files alone.
    """

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = os.path.join(log_dir, f"run_{ts}.json")
        self._lock = threading.Lock()
        self._record: dict = {
            "timestamp": ts,
            "prompt": "",
            "completed": False,
            "iterations": 0,
            "total_tool_calls": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "steps": [],
        }

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._record["prompt"] = prompt

    def log_step(self, iteration: int, tool_calls: list[dict]) -> None:
        with self._lock:
            self._record["steps"].append({
                "iteration": iteration,
                "tool_calls": tool_calls,
            })
            self._record["iterations"] = iteration
            self._record["total_tool_calls"] += len(tool_calls)

    def log_tokens(self, prompt_tokens: int, response_tokens: int) -> None:
        with self._lock:
            self._record["total_prompt_tokens"] += prompt_tokens or 0
            self._record["total_response_tokens"] += response_tokens or 0

    def finish(self, completed: bool) -> None:
        with self._lock:
            self._record["completed"] = completed
            with open(self._path, "w") as f:
                json.dump(self._record, f, indent=2)

    @property
    def path(self) -> str:
        return self._path