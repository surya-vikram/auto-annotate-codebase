import sys
import os
import logging
from networkx.drawing.nx_pydot import read_dot
# from annotate import annotate_file

def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python main.py packages.dot [all|<package>] <logfilename>")
        sys.exit(1)

    dot_file = sys.argv[1]
    filter_arg = sys.argv[2]  
    logfilename = sys.argv[3]  

    if not logfilename.endswith(".log"):
        logfilename += ".log"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, logfilename)
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

    try:
        graph = read_dot(dot_file)

    except Exception as e:
        print(f"Error reading {dot_file}: {e}")
        sys.exit(1)

    node_children = {}
    for node in graph.nodes():
        node_children[node] = list(graph.neighbors(node))


    def dfs(node, annotated):

        annotated.add(node)
        children = node_children.get(node, []) 
        children.sort(key=lambda x: len(node_children.get(x, [])), reverse=True)

        for child in children:
            if child not in annotated:
                dfs(child, annotated)

        # annotate_file(node, node_children.get(node, []))

        logging.info("Node: %s, Children: %s", node, node_children.get(node, []))


    annotated = set()

    if filter_arg.lower() == "all":
        subgraph_nodes = list(node_children.keys())
    else:
        subgraph_nodes = [node for node in node_children if filter_arg in node]

    print(f'{len(subgraph_nodes)} Files')

    subgraph_nodes.sort(key=lambda n: len(node_children.get(n, [])), reverse=True) 
    for node in subgraph_nodes:
        if node not in annotated:
            dfs(node, annotated)

if __name__ == "__main__":
    if input('Confirm that you want to start annotation with "Y" :') == 'Y':
        main()
        print('Annotation complete.')
    else:
        print('Operation cancelled')
