import os
from google.genai import types


schema_write_file = types.FunctionDeclaration(
    name="write_file",
    description="Write a file at a specified path relative to the working directory. Overwrites a file if it exists at the path",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        description="Arguments for writing and writing to a file",
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="File path of the file to access relative to the working directory",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Text content to write to the file",
            ),
        },
        required=["file_path", "content"]
    ),
)


def write_file(working_directory, file_path, content):
    try:
        abs_working = os.path.realpath(working_directory)
        target = os.path.normpath(os.path.join(abs_working, file_path))

        valid_target = os.path.commonpath([abs_working, target]) == abs_working

        if not valid_target:
            return f"Error: Cannot write '{file_path}' as it is outside the permitted working directory."
        
        if os.path.isdir(target):
            return f"Error: Cannot write to '{file_path}' as it is a directory"  
      
        os.makedirs(os.path.dirname(target), exist_ok=True)

        with open(target, 'w', encoding="utf-8") as file:
            file.write(content)

        return f"Successfully wrote to '{file_path}' ({len(content)} characters written)"

    except Exception as e:
        return f"Error: {str(e)}"