import os
import ast
import glob

def extract_info(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
            tree = ast.parse(content)
        except Exception:
            return []
    
    info = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            docstring = ast.get_docstring(node)
            info.append({
                'type': 'class',
                'name': node.name,
                'docstring': docstring,
                'methods': methods
            })
        elif isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            info.append({
                'type': 'function',
                'name': node.name,
                'docstring': docstring
            })
    return info

def generate_docs():
    src_dir = os.path.join("src", "local_rag")
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)) and d != "__pycache__"]
    
    index_content = "# Local RAG Documentation\n\n"
    
    for subdir in subdirs:
        subdir_path = os.path.join(src_dir, subdir)
        py_files = glob.glob(os.path.join(subdir_path, "*.py"))
        
        doc_filepath = os.path.join(docs_dir, f"{subdir}.md")
        index_content += f"- [{subdir}]({subdir}.md)\n"
        
        with open(doc_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Module: `{subdir}`\n\n")
            f.write(f"Context and technical details for the `{subdir}` module.\n\n")
            
            for py_file in py_files:
                basename = os.path.basename(py_file)
                if basename == "__init__.py":
                    continue
                f.write(f"## File: `{basename}`\n\n")
                
                info = extract_info(py_file)
                for item in info:
                    if item['type'] == 'class':
                        f.write(f"### Class: `{item['name']}`\n")
                        if item['docstring']:
                            f.write(f"> {item['docstring']}\n\n")
                        if item['methods']:
                            f.write("**Methods:**\n")
                            for method in item['methods']:
                                if not method.startswith("__") or method == "__init__":
                                    f.write(f"- `{method}`\n")
                        f.write("\n")
                    elif item['type'] == 'function':
                        if not item['name'].startswith("__"):
                            f.write(f"### Function: `{item['name']}`\n")
                            if item['docstring']:
                                f.write(f"> {item['docstring']}\n\n")
                            f.write("\n")

    with open(os.path.join(docs_dir, "index.md"), 'w', encoding='utf-8') as f:
        f.write(index_content)
        
if __name__ == "__main__":
    generate_docs()
    print("Documentation generated in docs/")
