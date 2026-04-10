import os
import argparse
import prompts
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.call_function import available_functions, partition_calls
from utils.api import call_api, call_and_validate
from utils.logger import RunLogger
from utils.memory import MemoryStore, OutcomeRecord, extract_preferences, generate_run_summary
from concurrent.futures import ThreadPoolExecutor, as_completed

    
#
# Main
#

def main():
    args = parse_arguments()
    logging.basicConfig(            # no-op if the root logger already has handlers, potential issue
        filename="logs/agent.log",  # if a testing framework or library configures logging
        level=logging.WARNING,      # before main() runs
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
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

    # should implement a messages deque for better performance when prepending from vector store
    # ^ no longer needed: no longer prepending messages. May have also caused Gemini SDK issues.

    memory = MemoryStore()      # always instantiating MemoryStore() prevents future UnboundLocalErrors
                                # should any memory-related logic be added that runs regardless of the
                                # --no-memory flag
    if not args.no_memory:
        past_outcomes = memory.retrieve_outcome(args.user_prompt)
        past_prefs = memory.retrieve_preferences()
        context_msg = memory.format_context_message(past_outcomes, past_prefs)
    else:
        context_msg = None

    messages = []
    if context_msg:
        messages.append(types.Content(      # append first as user message
            role="user",
            parts=[types.Part(text=context_msg)]
        ))
    messages.append(types.Content(
        role="user",
        parts=[types.Part(text=args.user_prompt)]
    ))

    final_answer = ""
    total_tool_calls = 0
    completed = False
    
    for iteration in range(1, 11):

        function_responses, done, final_answer = generate_content(
            client, messages, args.verbose, logger, iteration
            )
        
        if done:
            completed = True
            break

        if function_responses is not None:
            total_tool_calls += len(function_responses)
            messages.append(types.Content(role="user", parts=function_responses))

    logger.finish(completed)

    if completed and not args.no_memory:
        summary = generate_run_summary(client, args.user_prompt, final_answer)
        memory.write_outcome(OutcomeRecord(
            prompt=args.user_prompt,
            summary=summary,
            tool_calls=total_tool_calls,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_log_path=logger.path,
        ))
        preferences = extract_preferences(client, args.user_prompt, final_answer)
        for pref in preferences:
            memory.write_preference(pref)

    if args.verbose:
        print(f"\nRun log saved to: {logger.path}")
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory retrieval and writing for this run")
    return parser.parse_args()

def generate_content(client, messages, verbose, logger: RunLogger, iteration: int):
    """
    Returns (function_response_parts | None, is_done, final_answer | None).
    is_done=True when the model produces a text-only final answer.
    """

    response = call_api(
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
        return None, True, response.text or "" # done, model gave a final text answer
    
    parallel_calls, sequential_calls = partition_calls(response.function_calls)

    all_indexed = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(call_and_validate, fc, verbose): i
            for i, fc in parallel_calls
        }
        
        for future in as_completed(futures):
            try:
                all_indexed.append((futures[future], future.result()))   # errors in threads re-raised at
            except Exception:                                            # future.result()
                for f in futures:
                    f.cancel()  # explicitly cancel in-flight future when exceptions raise
                raise

    for i, fc in sequential_calls:
        all_indexed.append((i, call_and_validate(fc, verbose)))

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
    return function_results, False, None # not done yet

if __name__ == "__main__":
    main()