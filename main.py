import os
import json
import argparse
import prompts
import logging
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.call_function import available_functions, call_function, partition_calls
from utils.retry import with_backoff
from concurrent.futures import ThreadPoolExecutor, as_completed

_subprocess_semaphore = threading.Semaphore(2)

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
        self._record["prompt"] = prompt

    def log_step(self, iteration: int, tool_calls: list[dict]) -> None:
        self._record["steps"].append({
            "iteration": iteration,
            "tool_calls": tool_calls,
        })
        self._record["iterations"] = iteration
        self._record["total_tool_calls"] += len(tool_calls)

    def log_tokens(self, prompt_tokens: int, response_tokens: int) -> None:
        self._record["total_prompt_tokens"] += prompt_tokens or 0
        self._record["total_response_tokens"] += response_tokens or 0

    def finish(self, completed: bool) -> None:
        self._record["completed"] = completed
        with open(self._path, "w") as f:
            json.dump(self._record, f, indent=2)

    @property
    def path(self) -> str:
        return self._path
    
#
# Main
#

def main():
    args = parse_arguments()
    logging.basicConfig(            # no-op if the root logger already has handlers, potential issue
        filename="logs/agent.log",  # if a testing framework or library configures logging
        level=logging.WARNING,      # before main() runs
        format="%(asctime)s %(name)s %(levelname)s %(messages)s",
    )

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API key was not found.")
    
    client = genai.Client(api_key=api_key)

    logger = RunLogger()
    logger.set_prompt(args.user_prompt)

    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")

    messages = [
        types.Content(
            role="user",
                parts=[types.Part(text=args.user_prompt)]
        )
    ]

    completed = False
    
    for iteration in range(1, 11):

        function_responses, done = generate_content(
            client, messages, args.verbose, logger, iteration
            )
        
        if done:
            completed = True
            break

        if function_responses:
            messages.append(types.Content(role="user", parts=function_responses))

    logger.finish(completed)

    if args.verbose:
        print(f"\nRun log saved to: {logger.path}")
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

@with_backoff(max_retries=5, base_delay=1.0)
def _call_api(client, model, config, contents):
    return client.models.generate_content(
        model=model,
        config=config,
        contents=contents,
    )

# wrapper for error handling with the ThreadPoolExecutor
def _call_and_validate(fc, verbose):
    if fc.name == "run_python_file":
        with _subprocess_semaphore:
            response = call_function(fc, verbose)
    else:
        response = call_function(fc, verbose)

    if not response.parts:
        raise Exception("Call function parts list is empty")
    if not response.parts[0].function_response:
        raise Exception("No function response object returned")
    if not response.parts[0].function_response.response:
        raise Exception("No result from function call")
    
    return response

def generate_content(client, messages, verbose, logger: RunLogger, iteration: int):
    """
    Returns (function_response_parts | None, is_done).
    is_done=True when the model produces a text-only final answer.
    """

    response = _call_api(
        client,
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[available_functions],
            system_instruction=prompts.system_prompt
        ),
        contents=messages,
    )
    
    if not response.usage_metadata:
        raise RuntimeError("Gemini API response appears to be malformed")
    
    logger.log_tokens(
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count,
    )
    
    if verbose:
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    
    if not response.function_calls:
        print("Response:")
        print(response.text)
        return None, True # done, model gave a final text answer
    
    parallel_calls, sequential_calls = partition_calls(response.function_calls)

    all_indexed = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_call_and_validate, fc, verbose): i
            for i, fc in parallel_calls
        }
        
        for future in as_completed(futures):
            try:
                all_indexed.append((futures[future], future.result()))   # errors in threads re-raised at
            except Exception:                                            # future.result()
                for f in future:
                    f.cancel()  # explicitly cancel in-flight future when exceptions raise
                raise

    for i, fc in sequential_calls:
        all_indexed.append((i, _call_and_validate(fc, verbose)))

    function_results = [r for _, r in sorted(all_indexed)]

    # Build logging data after sorting. Single thread, correct order, no rc
    step_tool_calls = []
    for i, fc in enumerate(response.function_calls):
        raw_result = function_results[i].parts[0].function_response.response

        step_tool_calls.append({
            "name": fc.name,
            "args": dict(fc.args or {}),
            "result_preview": str(raw_result)[:300],
        })

        if verbose:
            print(f"-> {raw_result}")
    
    logger.log_step(iteration, step_tool_calls)
    return function_results, False # not done yet

    # for function_call in response.function_calls:
        
    #     function_response = call_function(function_call, verbose)

    #     if not function_response.parts:
    #         raise Exception("Call function parts list is empty")

    #     if not function_response.parts[0].function_response:
    #         raise Exception("No function response object returned")
        
    #     if not function_response.parts[0].function_response.response:
    #         raise Exception("No result from function call")
        
    #     raw_result = function_response.parts[0].function_response.response
    #     result_preview = str(raw_result)[:300]

    #     step_tool_calls.append({
    #         "name": function_call.name,
    #         "args": dict(function_call.args or {}),
    #         "result_preview": result_preview,
    #     })

    #     if verbose:
    #         print(f"-> {raw_result}")

    #     function_results.append(function_response)

    # logger.log_step(iteration, step_tool_calls)
    # return function_results, False # not done yet


if __name__ == "__main__":
    main()