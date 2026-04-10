import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from config import MEMORY_DIR, MEMORY_TOP_K
from prompts import preference_extraction_prompt, summarize_run_prompt

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
        if self._outcomes.count() == 0:
            return []
        
        results = self._outcomes.query(
            query_texts=[query],
            n_results=min(MEMORY_TOP_K, self._outcomes.count()),
            include=["documents", "metadatas", "distances"],
        )

        return [
            {"summary": doc, "meta": meta, "distance": dist}
            for doc, meta, dist
            in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
            if dist < 0.7   # filter out low-relevance results
        ]

    def retrieve_preferences(self) -> list[dict]:
        """Returns all preferences sorted by reinforcement count."""
        if self._preferences.count() == 0:
            return []
        
        results = self._preferences.get(include=["metadatas"])

        return sorted(
            results["metadatas"],
            key=lambda x: x["count"],
            reverse=True,
        )

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
        parts = []

        if preferences:
            pref_lines = "\n".join(
                f"- {p['preference']} (seen {p['count']} time(s))"
                for p in preferences[:5]    # cap at 5 most reinforced
            )
            parts.append(f"[USER PREFERENCES - learned from past sessions]\n{pref_lines}")

        if outcomes:
            outcome_lines = "\n".join(
                f"Task ({o['meta']['timestamp'][:10]}): {o['summary']}"
                for o in outcomes
            )
            parts.append(f"[RELEVANT PAST RUNS]\n{outcome_lines}")
        
        if not parts:
            return None
        
        return (
            "[MEMORY - retrieved from past sessions."
            "Use this to avoid repeating known work and to respect user preferences.]\n\n"
            + "\n\n".join(parts)
            + "\n\n[END MEMORY]"
        )

def extract_preferences(client, prompt: str, outcome: str) -> list[str]:
    """One extra Gemini call per completed run to mine preference signals."""
    from google.genai import types as gtypes
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=gtypes.GenerateContentConfig(
            system_instruction=preference_extraction_prompt,
        ),
        contents=[
            gtypes.Content(role="user", parts=[gtypes.Part(text=(
                f"User prompt: {prompt}\n\nAgent outcome: {outcome}"
            ))])
        ],
    )

    try:
        raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        return []
    
def generate_run_summary(client, prompt: str, outcome: str) -> str:
    """Summarize what the agent did in a completed run, for memory storage."""
    from google.genai import types as gtypes
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=gtypes.GenerateContentConfig(
            system_instruction=summarize_run_prompt,
        ),
        contents=[
            gtypes.Content(role="user", parts=[gtypes.Part(text=(
                f"User prompt: {prompt}\n\nAgent final answer: {outcome}"
            ))])
        ],
    )

    # prompt[:200] fallback to get a non-empty string if the API call fails
    # ChromaDB requires a non-empty document for embedding
    return response.text.strip() if response.text else prompt[:200]