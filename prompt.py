template_prompt = """
# Python File Analysis and Annotation Task

You will be provided with a Python file (the **Main File**) and, if applicable, its directly dependent child files (i.e. modules that the main file imports). If the main file does not import any additional files, it is considered self-contained.

Your task is to **analyze and annotate only the main file** using the provided Python context. **In addition, if the main file imports functions, classes, or variables from dependent files, you must leverage the content of these dependent files to produce a comprehensive annotation.** This means that for each imported component, you should:
- **Describe its purpose and functionality** based on the content in the dependent file.
- **Explain its parameters, return values, and any exceptions** it might raise.
- **Clarify how it integrates** into the logic and control flow of the main file.

---

## **Main File**

- **File Name:**  
  `{main_file_name}`

- **File Content:**  
  `{main_file_content}`

---

## **Dependent Files (if any)**

If the main file imports other modules, additional context is provided below. If there are no dependent files, you can ignore this section.

{dependent_files_section}

---

## **Step 1: Detailed Analysis and Annotation of the Main File**

### **1. File Summary**

Provide a high-level summary that explains the following:

- **Overview:**  
  - Describe the main purpose of the file.

- **Detailed Breakdown:**  
  - **Classes:** Explain the responsibilities of each class and their relationships.  
    *If a class is imported from a dependent file, include details from that file.*
  - **Functions/Methods:** Describe the intended behavior and purpose of each function or method.  
    *For any imported function or method, analyze its implementation in the dependent file to explain parameters, return types, exceptions, and overall functionality.*
  - **Variables/Constants:** Outline the significance of key variables and constants.
  - **Imports & External Dependencies:** Explain how the file interacts with other parts of the project through its imports. Include details from dependent files to clarify what external functionalities are being leveraged.
  - **Key Logic & Control Flow:** Summarize how the functions and control flows interact within the file.

---

### **2. Code Annotation**

Enhance the provided code by adding detailed documentation and inline comments:

#### **A. Docstrings**

- **Module-Level Docstring (Top of File):**  
  - Provide an overview of the file’s purpose.
  - Summarize key functionalities.
  - List any external dependencies and important module-level context.
  - **Include references to any imported components** (from dependent files) that play a key role in the file's logic.

- **Function, Method, and Class Docstrings:**  
  For each class, function, or method, include:
  - A clear description of its purpose.
  - Detailed parameter descriptions (name, type, expected values).
  - Return value details (type and meaning).
  - Any side effects.
  - Exceptions raised, if any.
  - **If the function, method, or class originates from a dependent file, include additional documentation drawn from that file’s content.**

#### **B. Inline Comments**

- Provide clear, line-by-line explanations for complex logic.
- Clarify important decisions, control flow, and algorithms.
- **For lines that invoke or rely on imported components,** add inline comments that summarize what those external functions or classes do, based on your analysis of the dependent files.
- Ensure that even without the actual code, an experienced Python developer could reconstruct the functionality solely from your comments.

---

## **Step 2: Leveraging Dependent File Content**

For each imported module, perform the following:

1. **Identify Imported Components:**  
   List and briefly describe each function, class, or variable imported from the dependent files.

2. **In-depth Analysis of Dependencies:**  
   For each component:
   - Provide details on its implementation as defined in its respective dependent file.
   - Explain its parameters, return types, and any exceptions it may raise.
   - Describe how it integrates with and supports the main file’s logic.

3. **Integrate Dependency Analysis:**  
   Where applicable, modify the annotations in the main file to reference these in-depth details. This helps create a self-explanatory guide that unifies the context from both the main file and its dependencies.

---

## **Step 3: Ensuring Clarity & Completeness**

The final annotated version of the main file should serve as a **self-explanatory guide**, enabling any experienced Python developer to:

1. **Understand** the file’s structure and its dependencies.
2. **Reconstruct** the logic using the docstrings and inline comments.
3. **Navigate** the module effectively with the provided context and the additional insights derived from the dependent files.

---

## **Execution Instructions**

1. **Receive the Python file and its dependent files (if applicable).**
2. **Analyze and annotate only the main file** as detailed in Steps 1 and 2 above, ensuring that you fully incorporate insights from the dependent files when annotating any imported components.
"""
