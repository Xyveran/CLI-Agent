import os
import argparse
import prompts
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.call_function import available_functions, partition_calls
from utils.api import call_api, call_and_validate
from utils.logger import RunLogger
from utils.memory import MemoryStore, MemoryRecord
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
    messages = [
        types.Content(
            role="user",
                parts=[types.Part(text=args.user_prompt)]
        )
    ]

    memory = MemoryStore()
    past_context = memory.retrieve(args.user_prompt)
    if past_context:
        # prepend as a system-level context message
        messages.insert(0, types.Content(           
            role="user",
            parts=[types.Part(text=memory.format_for_prompt(past_context))]
        ))

    completed = False
    
    for iteration in range(1, 11):

        function_responses, done = generate_content(
            client, messages, args.verbose, logger, iteration
            )
        
        if done:
            completed = True
            break

        if function_responses is not None:
            messages.append(types.Content(role="user", parts=function_responses))

    logger.finish(completed)

    # add summary generation for outcome data

    if completed:
        memory.write(MemoryRecord(
            prompt=args.user_prompt,
            summary=...,
            outcome=...,
        ))

    if args.verbose:
        print(f"\nRun log saved to: {logger.path}")
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # add --no-memory to opt out of vector store mem during testing
    return parser.parse_args()

def generate_content(client, messages, verbose, logger: RunLogger, iteration: int):
    """
    Returns (function_response_parts | None, is_done).
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
        return None, True # done, model gave a final text answer
    
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
    return function_results, False # not done yet

if __name__ == "__main__":
    main()