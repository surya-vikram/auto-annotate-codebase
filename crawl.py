import os


def count_python_files_and_lines(directory):
    result = {}
    total_files = 0
    total_lines = 0

    for root, _, files in os.walk(directory):
        # Filter only Python files
        py_files = [file for file in files if file.endswith('.py')]
        file_count = len(py_files)
        result[root] = file_count
        total_files += file_count

        # Count lines in each Python file
        for file in py_files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
            except Exception as e:
                print(f"Could not read file: {file_path}. Error: {e}")

    return result, total_files, total_lines


def display_results(results, total_files, total_lines):
    print("Directory Name                         | No. of Python Files")
    print("------------------------------------------------------------")
    for directory, count in results.items():
        print(f"{directory:<40} | {count}")
    print("------------------------------------------------------------")
    print(f"Total Python Files: {total_files}")
    print(f"Total Lines in Python Files: {total_lines}")


if __name__ == "__main__":
    base_dir = "./astropy"  # Change this to your target directory
    results, total_files, total_lines = count_python_files_and_lines(base_dir)
    display_results(results, total_files, total_lines)
