import os
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from config import MEMORY_DIR

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
# https://docs.trychroma.com/integrations/embedding-models/google-gemini
class MemoryStore:
    """
    Wraps ChromaDB to provide outcome and preference memory
    across agent runs. Two collections are maintained:
        - 'outcomes'    : one record per completed run
        - 'preferences' : inferred user behavior patterns
    """

    def __init__(self): # api_key arg not needed
        self._client = chromadb.PersistentClient(path=MEMORY_DIR)
        embed_fn = embedding_functions.GoogleGeminiEmbeddingFunction(
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            dimension=768,  # range of 128-3072 supported. Lower dimension is sufficient here.
        )
        self._outcomes = self._client.get_or_create_collection(
            name="outcomes",
            embedding_function=embed_fn,
        )
        self._preferences = self._client.get_or_create_collection(
            name="preferences",
            embedding_function=embed_fn
        )

    #
    # Write
    #

    def write_outcome(self, record: OutcomeRecord) -> None:
        doc_id = f"outcome_{record.timestamp}"
        self._outcomes.upsert(
            ids=[doc_id],
            documents=[record.summary], # what gets embedded and searched
            metadatas=[asdict(record)],
        )

    def write_preference(self, preference: str) -> None:
        """
        Upsert a preference signal. If a near-duplicate already exists
        (cosine distance < 0.15), increment its count instead of inserting.
        """
        results = self._preferences.query(
            query_texts=[preference],
            n_results=1,
            include=["distances","metadatas"],
        )

        now = datetime.now(timezone.utc).isoformat()

        if results["ids"][0] and results["distances"][0][0] < 0.15:
            # reinforce existing preference
            existing = results["metadatas"][0][0]
            existing_id = results["ids"][0][0]
            existing["count"] += 1
            existing["last_reinforced"] = now
            self._preferences.upsert(
                ids=[existing_id],
                documents=[preference],
                metadatas=[existing],
            )
        else:
            # new preference
            doc_id = f"pref_{now}"
            self._preferences.upsert(
                ids=[doc_id],
                documents=[preference],
                metadatas=[{
                    "preference": preference,
                    "first_seen": now,
                    "last_reinforced": now,
                    "count": 1,
                }],
            )

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

    def extract_preferences(client, prompt: str, outcome: str) -> list[str]:
        """One extra Gemini call per completed run to mine preference signals."""
        pass