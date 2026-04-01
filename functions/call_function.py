from .get_files_info import schema_get_files_info, get_files_info
from .get_file_content import schema_get_file_content, get_file_content
from .run_python_file import schema_run_python_file, run_python_file
from .write_file import schema_write_file, write_file
from config import WORKING_DIR
from google import genai
from google.genai import types


available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_get_file_content,
        schema_run_python_file,
        schema_write_file,
    ],
)

def call_function(function_call: types.FunctionCall, verbose=False):
    
    if verbose:
        print(f"Calling function: {function_call.name}({function_call.args})")
    else:
        print(f" - Calling function: {function_call.name}")

    function_map = {
        "get_file_content": get_file_content,
        "get_files_info": get_files_info,
        "run_python_file": run_python_file,
        "write_file": write_file,
    }

    function_name = function_call.name or ""

    if function_name not in function_map:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Unknown function: {function_name}"},
                )
            ],
        )
    
    args = dict(function_call.args) if function_call.args else {}

    args["working_directory"] = WORKING_DIR
    
    try:
        function_result = function_map[function_name](**args)
    except Exception as e:
        function_result = f"Error executing {function_name}: {e}" 

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
        ],
    )

# path-overlap classifier, determine if function calls are safe to parallelize
def partition_calls(function_calls):
    """
    Returns (parallel_group, must_sequence) based on file path overlap.
    Each list holds tuples of (call_order, function_call).
    """
    write_paths = set()
    parallel, sequential = [], []

    for i, fc in function_calls:
        if fc.name == "run_python_file":    # Special case of running files.
            sequential.append((i, fc))      # Conservative choice to delegate them to sequential to
            continue                        # avoid non-deterministic and context-dependent bugs.
                                            # e.g. cascading script dependencies, or shared output files
        args = dict(fc.args or {})
        path = args.get("file_path") or args.get("directory")
        is_write = fc.name == "write_file"

        if is_write and path in write_paths:
            sequential.append((i, fc))   # same file written twice
        elif path in write_paths:
            sequential.append((i, fc))   # reading a file being written
        else:
            parallel.append((i, fc))
            if is_write and path:
                write_paths.add(path)

    return parallel, sequential