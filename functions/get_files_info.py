import os
from google.genai import types


schema_get_files_info = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in a specified directory relative to the working directory, providing file size and directory status",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description="Directory path to list files from, relative to the working directory (default is the working directory itself)",
            ),
        },
    ),
)


def get_files_info(working_directory, directory="."):
    try:
        abs_working = os.path.realpath(working_directory)
        target = os.path.normpath(os.path.join(abs_working, directory))

        valid_target = os.path.commonpath([abs_working, target]) == abs_working

        if not valid_target:
            return f"Error: Cannot list '{directory}' as it is outside the permitted working directory."
        
        if not os.path.isdir(target):
            return f"Error: '{directory}' is not a directory"

        dir_contents = []

        for item in os.listdir(target):        
            item_path = os.path.join(target, item)

            file_size = os.path.getsize(item_path)
            is_dir = os.path.isdir(item_path)

            line = f"{item}: file_size={file_size} bytes, is_dir={is_dir}"
            dir_contents.append(line)

        return "\n - ".join(dir_contents)

    except Exception as e:
        return f"Error: {str(e)}"