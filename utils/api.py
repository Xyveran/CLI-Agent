import threading
from google.genai import types
from utils.retry import with_backoff
from functions.call_function import call_function


#
# Subprocess concurrency cap, shared across all calls within a run
#

_subprocess_semaphore = threading.Semaphore(2)

# This pattern applies a fresh decorator on each call_api for a modifiable base_delay param
def call_api(client, model, config, contents, base_delay: float = 1.0):
    @with_backoff(max_retries=5, base_delay=base_delay)
    def _inner():
        return client.models.generate_content(
            model=model, config=config, contents=contents,
        )
    return _inner()

#
# Validated call wrapper
#

def call_and_validate(fc, verbose) -> types.Content:
    """Call a single function and validate the response structure."""
    if fc.name == "run_python_file":
        with _subprocess_semaphore:
            response = call_function(fc, verbose)
    else:
        response = call_function(fc, verbose)
 
    if not response.parts:
        raise Exception(f"Call function parts list is empty for {fc.name}")
    if not response.parts[0].function_response:
        raise Exception(f"No function response object returned for {fc.name}")
    if not response.parts[0].function_response.response:
        raise Exception(f"No result from function call for {fc.name}")
 
    return response