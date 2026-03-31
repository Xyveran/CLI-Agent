import os
import subprocess
from google.genai import types


schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Runs a python file at a specified file path, relative to the working directory. Returns std output and error",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        description="Arguments for executing a Python file",
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="File path of the file to access relative to the working directory",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.STRING,
                ),
                description="Array of arguments to pass",
            ),
        },
        required=["file_path"]
    ),
)


def run_python_file(working_directory, file_path, args=None):
    try:
        abs_working = os.path.realpath(working_directory)
        target = os.path.normpath(os.path.join(abs_working, file_path))

        valid_target = os.path.commonpath([abs_working, target]) == abs_working

        if not valid_target:
            return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory.'
        
        if not os.path.isfile(target):
            return f'Error: "{file_path}" does not exist or is not a regular file'

        if not target.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'

        command = ["python", target]
        if args:
            command.extend(args)

        process = subprocess.run(
                        args=command,
                        cwd=abs_working,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,                                           
                        timeout=30,
                    )
        
        string_content = []

        if process.returncode != 0:
            string_content.append(f'Process exited with code {process.returncode}')

        if not process.stdout and not process.stderr:
            string_content.append('No output produced')
        else:
            if process.stdout:
                string_content.append(f'STDOUT:\n{process.stdout}')

            if process.stderr:
                string_content.append(f'STDERR:\n{process.stderr}')
        
        return "\n".join(string_content)
    except subprocess.TimeoutExpired:
        return 'Error: Execution timed out (30 second limit)'
    except Exception as e:
        return f'Error: executing Python file: {e}'