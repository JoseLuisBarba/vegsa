import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from src.variables import N, coords, a, b

def generate_html_colors(n : int):
    def htmlcolor():
        import random
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        return color
    if n == 1:
        return htmlcolor()
    return [ htmlcolor() for i in range(0, n)]




def draw_graph(routeGraph: dict):
    G = nx.Graph()


    nodes = list()
    if len(routeGraph) > 1:
        html_colors = generate_html_colors(len(routeGraph))
    else:
        html_colors = generate_html_colors(2)

    for vehicle, route in routeGraph.items():
        vehicleColor = html_colors[vehicle]
        for travel in route:
            attributes = {'color': vehicleColor, 'window': f'{travel[0]}: \n [{a[travel[0]]},{b[travel[0]]}]'}
            nodes.append((travel[0], attributes))

    G.add_nodes_from(nodes)
    edges = list()

    for vehicle, route in routeGraph.items():
        vehicleColor = html_colors[vehicle]
        for i in range(0, len(route)):
            if i == len(route) - 1:
                break
            attributes = {'color': vehicleColor}
            edges.append((route[i][0], route[i+1][0]))
            
    G.add_edges_from(edges)



    #colors = [node[1]['color'] for node in G.nodes(data=True)] 
    #edge_colors = [edge[2] for edge in edges]
    colors = [node[1]['color'] for node in G.nodes(data=True)]
    window = {node[0]:node[1]['window'] for node in G.nodes(data=True)}
    ax = plt.figure(figsize=(10, 8)).gca()
    ax.set_axis_off()
    #print([edge[2] for edge in edges])
    options = {"node_size": 500, "node_color": colors, 'labels':window,} #"node_color": colors
    coords[:,0] = coords[:,0] -716
    coords[:,1] = coords[:,1] -9100
    positions: dict = { id: pos for id, pos in zip(N[:len(N)-1], coords[:len(coords) - 1])}
    
    nx.draw_networkx(G, positions, with_labels=True, **options)