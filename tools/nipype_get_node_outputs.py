import nipype


def getoutimglocs(exec_graph, node, output_name):
    hierarchyName = str(node)
    while type(node) == nipype.pipeline.engine.Workflow:
        node = node.get_node(output_name.split(".")[0])
        hierarchyName = hierarchyName + "." + node.name
        output_name = ".".join(output_name.split(".")[1:])
    # get all nodes from the workflow
    nodelist = list(exec_graph.nodes())
    # get all nodes hierarchical string
    indexOfMatches = []
    for i in range(len(nodelist)):
        hierarchy = nodelist[i]._hierarchy + "." + nodelist[i].name
        if hierarchy == hierarchyName:
            indexOfMatches.append(i)
    if len(indexOfMatches) == 0:
        print("getoutimg: Couldn't find node " + str(node))
        return
    else:
        imglocs = []
        for i in indexOfMatches:
            imglocs.append(nodelist[i].get_output(output_name))
        return imglocs


def get_node(exec_graph, hierarchy_node_name):
    output_node = None
    for node in exec_graph.nodes():
        if hierarchy_node_name == node._hierarchy + "." + node.name:
            output_node = node
            break
    return output_node


def print_available_nodes(exec_graph):
    for x in exec_graph.nodes():
        print(x)

def get_node_output(exec_graph, hierarchy_node_name,output_name):
    node = get_node(exec_graph,hierarchy_node_name)
    if node is not None:
        return node.get_output(output_name)
    else:
        print('Could not find node.')
        return None