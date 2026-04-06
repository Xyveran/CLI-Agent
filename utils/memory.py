import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from config import MEMORY_DIR, MEMORY_TOP_K

# need to store:
#  task outcomes, one per completed run
#   query vector -> embedding of user prompt
#   stored text -> what the agent did and concluded
#   metadata -> timestamp, tool_calls, run_log_path, type="outcome"
#
#  preference signals, accumulated, de-duplicated
#   query vector -> embedding of the inferred preference
#   stored text -> "User prefers --verbose output"
#   metadata -> first_seen, last_reinforced, count, type="preference"

# @dataclass
# class MemoryRecord:
#     prompt: str         # original user prompt
#     summary: str        # what the agent did or found
#     outcome: str        # final text answer (truncated)
#     tool_calls: int     # how many tools were used
#     timestamp: str      # ISO UTC
#     run_log_path: str   # link back to the full JSON log for auditability

@dataclass
class OutcomeRecord:
    prompt: str
    summary: str
    tool_calls: str
    timestamp: str
    run_log_path: str

@dataclass
class PreferenceRecord:
    preference: str         # e.g. "prfers -verbose output"
    first_seen: str
    last_reinforced: str
    count: int              # how many runs have reinforced this

# implement with Chromadb
class MemoryStore:
    """
    Wraps ChromaDB to provide outcome and preference memory
    across agent runs. Two collections are maintained:
        - 'outcomes'    : one record per completed run
        - 'preferences' : inferred user behavior patterns
    """

    def __init__(self, api_key: str):
        self._client
        self._outcomes
        self._preferences

    #
    # Write
    #

    def write_outcome(self, record: OutcomeRecord) -> None:
        pass

    def write_preference(self, preference: str) -> None:
        """
        Insert a preference signal. If a near-duplicate already exists
        (cosine distance < 0.15), increment its count instead of inserting.
        """
        pass

    #
    # Retrieve
    #

    def retrieve_outcome(self, query: str) -> list[dict]:
        pass

    def retrieve_preferences(self) -> list[dict]:
        """Returns all preferences sorted by reinforcement count."""
        pass

    #
    # Format for injection
    #
    
    def format_context_message(
            self,
            outcomes: list[dict],
            preferences: list[dict]
    ) -> Optional[str]:
        """
        Builds the text block prepended to the conversation as a user
        message before the agent loop starts. Returns None if there is
        nothing to inject.
        """
        pass