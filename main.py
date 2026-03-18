import os
import argparse
import prompts
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.call_function import available_functions, call_function


def main():
    args = parse_arguments()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API key was not found.")
    
    client = genai.Client(api_key=api_key)

    messages = [
        types.Content(
            role="user",
                parts=[types.Part(text=args.user_prompt)]
        )
    ]
    
    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")

    messages = [
        types.Content(
            role="user",
            parts=[types.Part(text=args.user_prompt)],
        )
    ]

    for _ in range(10):

        function_responses = generate_content(client, messages, args.verbose)

        if not function_responses:
            break

        messages.append(types.Content(role="user",parts=function_responses))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def generate_content(client, messages, verbose):
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        config=types.GenerateContentConfig(
            tools=[available_functions],
            system_instruction=prompts.system_prompt
            ),
        contents=messages,
    )
    
    if not response.usage_metadata:
        raise RuntimeError("Gemini API response appears to be malformed")
    
    if verbose:
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    
    if not response.function_calls:
        print("Response:")
        print(response.text)
        return None
    
    function_results = []

    for function_call in response.function_calls:
        
        function_response = call_function(function_call, verbose)

        if not function_response.parts:
            raise Exception("Call function parts list is empty")

        if not function_response.parts[0].function_response:
            raise Exception("No function response object returned")
        
        if not function_response.parts[0].function_response.response:
            raise Exception("No result from function call")         

        if verbose:
            print(f"-> {function_response.parts[0].function_response.response}")

        function_results.append(function_response)

    return function_results


if __name__ == "__main__":
    main()
