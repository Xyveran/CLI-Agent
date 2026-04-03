"""
Agent runner that wraps main.generate_content agentic loop
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.api import call_api, call_and_validate


class AgentRunner:
    """Thin wrapper around the project's main.py agentic loop."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._client = None
        self._available_functions = None
        self._system_prompt = None
    
    def _ensure_client(self):
        if self._client is not None:
            return
        
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it before running evals."
            )
    
        from google import genai
        from google.genai import types
        import prompts
        from functions.call_function import available_functions

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._available_functions = available_functions
        self._system_prompt = prompts.system_prompt

    def run(self, prompt: str) -> dict:
        """
        Run the agent on `prompt`.
        
        Returns a dict with keys:
            output (str), tool_calls (int), prompt_tokens (int), response_tokens (int)
        """
        self._ensure_client()

        types = self._types
        client = self._client

        messages = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        total_tool_calls = 0
        total_prompt_tokens = 0
        total_response_tokens = 0
        final_output = ""

        for _ in range(10):
            response = call_api(
                client,
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    tools=[self._available_functions],
                    system_instruction=self._system_prompt,
                ),
                contents=messages,
                base_delay=2.0
            )

            if response.usage_metadata:
                total_prompt_tokens += response.usage_metadata.prompt_token_count or 0

                total_response_tokens += (
                    response.usage_metadata.candidates_token_count or 0
                )
            
            if not response.function_calls:
                final_output = response.text or ""
                break

            parallel_calls, sequential_calls = _partition_calls(
                response.function_calls
            )

            all_indexed = []    # (original_index, result)

            # --- parallel group ---
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(call_and_validate, fc, self.verbose): i
                    for i, fc in parallel_calls
                }
                for future in as_completed(futures):
                    try:
                        all_indexed.append((futures[future], future.result()))
                    except Exception:
                        for f in futures:
                            f.cancel()
                        raise

            # --- sequential group ---
            for i, fc in sequential_calls:
                all_indexed.append((i, call_and_validate(fc, self.verbose)))

            # Reassemble in original order, then count
            function_results = [r for _, r in sorted(all_indexed)]
            total_tool_calls += len(function_results)

            all_parts = [part for result in function_results for part in result.parts]
 
            messages.append(
                types.Content(role="user", parts=all_parts)
            )

        return {
            "output": final_output,
            "tool_calls": total_tool_calls,
            "prompt_tokens": total_prompt_tokens,
            "response_tokens": total_response_tokens,
        }

#
# Path-overlap classifier
#

def _partition_calls(function_calls):
    """
    Splits function_calls into (parallel, sequential) groups.
    
    Sequential group contains:
        - Any run_python_file call (opaque filesystem access) 
        - Any write where the target path was already seen as a write target
        - Any read where the target path is already a pending write target
        
    Returns a list of (original_indrx, fc) tuples for both groups so that
    results can be reassembled in original order after execution.
    """
    write_paths = set()
    parallel, sequential = [], []
 
    for i, fc in enumerate(function_calls):
        if fc.name == "run_python_file":
            sequential.append((i, fc))
            continue
 
        args = dict(fc.args or {})
        path = args.get("file_path") or args.get("directory")
        is_write = fc.name == "write_file"
 
        if path in write_paths:
            sequential.append((i, fc))
        else:
            parallel.append((i, fc))
            if is_write and path:
                write_paths.add(path)
 
    return parallel, sequential