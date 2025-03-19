import re
import os
from pathlib import Path
from llm import api_client
from prompt import template_prompt


parent_dir = Path.cwd() / os.environ.get("DIRECTORY_NAME")
print(f"Parent directory: {parent_dir}")

def get_node_path(node):

    path = parent_dir.joinpath(*node.split('.'))
    if path.is_dir():
        path = path.joinpath("__init__.py")
    else:
        path = path.with_suffix(".py")

    if path.exists():
        stem = path.stem
        ws_path = path.parent / f"{stem}_ws{path.suffix}"
        if ws_path.exists():
            path = ws_path
        return path
    
    else:
        print(f"Path does not exist: {path}")
        return None


def annotate_file(node, children=[]):

    main_file_path = get_node_path(node)
    dependent_file_paths = [(child, get_node_path(child)) for child in children]

    try:
        with open(main_file_path, 'r', encoding='utf-8') as file:
            main_file_content = file.read()
    except Exception as e:
        return f"Error reading main file: {str(e)}"

    dependent_files_section = ""
    if dependent_file_paths:
        for i, tup in enumerate(dependent_file_paths, 1):
            dep_name, dep_path = tup
            try:
                with open(dep_path, 'r', encoding='utf-8') as file:
                    dep_content = file.read()

                dependent_files_section += f"""- **Dependent File {i}:**  
  **File Name:** `{dep_name}`
  **Content:** `{dep_content}`

"""
            except Exception as e:
                raise Exception(
                    f"Error reading file {os.path.basename(dep_path)}: {e}") from e

    else:
        dependent_files_section = "There are no dependent files for this main file."

    formatted_prompt = template_prompt.format(
        main_file_name=node,
        main_file_content=main_file_content,
        dependent_files_section=dependent_files_section
    )

    output = api_client(prompt=formatted_prompt)

    pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    matches = pattern.findall(output)

    if matches:
        python_code = matches[0].strip()
    else:
        raise ValueError("No Python code block found.")

    stem = main_file_path.stem
    suffix = main_file_path.suffix

    new_file_name = f"{stem}_ws{suffix}"
    new_file_path = main_file_path.parent / new_file_name

    if not new_file_path.parent.exists():
        raise FileNotFoundError(
            f"Directory {new_file_path.parent} does not exist.")

    with open(new_file_path, "w") as file:
        file.write(python_code)
