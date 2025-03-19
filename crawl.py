import os

def count_python_files(directory):
    result = {}
    total_files = 0

    for root, _, files in os.walk(directory):
        py_files = [file for file in files if file.endswith('.py')]
        file_count = len(py_files)
        result[root] = file_count
        total_files += file_count

    return result, total_files


def display_results(results, total_files):
    print("Directory Name                         | No. of Python Files")
    print("------------------------------------------------------------")
    for directory, count in results.items():
        print(f"{directory:<40} | {count}")

    print("------------------------------------------------------------")
    print(f"Total Python Files: {total_files}")


if __name__ == "__main__":
    base_dir = "./astropy"
    results, total_files = count_python_files(base_dir)
    display_results(results, total_files)
