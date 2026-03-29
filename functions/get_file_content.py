import os
from google.genai import types
from config import MAX_CHARS


schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description=f"Gets the content of a specified file relative to the working directory, up to {MAX_CHARS} characters.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="File path of the file to access relative to the working directory",
            ),
        },
        required=["file_path"],
    ),
)


def get_file_content(working_directory, file_path):
    try: 
        abs_working = os.path.realpath(working_directory)
        target = os.path.normpath(os.path.join(abs_working, file_path))

        valid_target = os.path.commonpath([abs_working, target]) == abs_working

        if not valid_target:
            return f"Error: Cannot read '{file_path}' as it is outside the permitted working directory."
        
        if not os.path.isfile(target):
            return f"Error: File not found or is not a regular file: '{file_path}'"
        
        with open(target, 'r') as file:
            file_content = file.read(MAX_CHARS)

            if file.read(1):
                file_content += f"[...File '{file_path}' truncated at {MAX_CHARS} characters]"
            
            return file_content

    except Exception as e:
        return f"Error: {str(e)}"