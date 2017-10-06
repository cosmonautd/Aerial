import numpy
import gtde
import matplotlib
import graph_tool as graphtool
import graph_tool.draw as draw
import graph_tool.search as search

""" The graphmap module provides methods to generate 
    traversal difficulty graphs from traversal difficulty images.
    Tools for path computation must also be provided.
    Class RouteEstimator: defines a route estimator and its configuration.
    Method tdi2graph: builds a graph from a TDI source matrix
    Method route: returns the best route between two regions
    Method drawgraph: saves a graph as an image (optionally, draws paths)
"""

def coord2(position, columns):
    """ Converts two-dimensional indexes to one-dimension coordinate
    """
    return position[0]*columns + position[1]

def drawgraph(G, path=[], filename="tdg.png"):
    for i, v in enumerate(path):
        G.vp.vfcolor[v] = [0.640625, 0, 0, 0.9]
        if i < len(path) - 1:
            for e in v.out_edges():
                if e.target() == path[i+1]:
                    G.ep.ecolor[e] = [0.640625, 0, 0, 0.9]
                    G.ep.ewidth[e] = 16

    draw.graph_draw(G, pos=G.vp.pos2, output_size=(1200, 1200), vertex_fill_color=G.vp.vfcolor,\
                    edge_color=G.ep.ecolor, edge_pen_width=G.ep.ewidth, output=filename)

class Visitor(search.DijkstraVisitor):

    def __init__(self):
        pass

    def discover_vertex(self, u):
        pass

    def examine_edge(self, e):
        pass

    def edge_relaxed(self, e):
        pass

class RouteEstimator:

    def __init__(self):
        pass
    
    def tdm2graph(self, tdmatrix):

        G = graphtool.Graph(directed=False)

        G.vp.pos = G.new_vertex_property("vector<double>")
        G.vp.pos2 = G.new_vertex_property("vector<double>")
        G.vp.diff = G.new_vertex_property("double")
        G.ep.weight = G.new_edge_property("double")

        for i, row in enumerate(tdmatrix):
            for j, element in enumerate(row):
                v = G.add_vertex()
                G.vp.pos[v] = [i, j]
                G.vp.pos2[v] = [j, i]
                #G.vp.diff[v] = tdmatrix[i][j] if tdmatrix[i][j] < 48 else tdmatrix[i][j]**2
                G.vp.diff[v] = tdmatrix[i][j]**2

        for v in G.vertices():
            (i, j) = G.vp.pos[v][0], G.vp.pos[v][1]

            top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            if i-1 > -1:
                if G.edge(coord2(top, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(top, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(top, tdmatrix.shape[1]))])
            if i+1 < tdmatrix.shape[0]:
                if G.edge(coord2(bottom, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(bottom, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(bottom, tdmatrix.shape[1]))])
            if j-1 > -1:
                if G.edge(coord2(left, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(left, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(left, tdmatrix.shape[1]))])
            if j+1 < tdmatrix.shape[1]:
                if G.edge(coord2(right, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(right, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(right, tdmatrix.shape[1]))])
            
            topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
            if i-1 > -1 and j-1 > -1:
                if G.edge(coord2(topleft, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(topleft, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(topleft, tdmatrix.shape[1]))])
            if i-1 > -1 and j+1 < tdmatrix.shape[1]:
                if G.edge(coord2(topright, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(topright, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(topright, tdmatrix.shape[1]))])
            if i+1 < tdmatrix.shape[0] and j-1 > -1:
                if G.edge(coord2(bottomleft, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(bottomleft, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(bottomleft, tdmatrix.shape[1]))])
            if i+1 < tdmatrix.shape[0] and j+1 < tdmatrix.shape[1]:
                if G.edge(coord2(bottomright, tdmatrix.shape[1]), v) == None:
                    e = G.add_edge(v, G.vertex(coord2(bottomright, tdmatrix.shape[1])))
                    G.ep.weight[e] = (G.vp.diff[v] + G.vp.diff[G.vertex(coord2(bottomright, tdmatrix.shape[1]))])

        G.vp.vfcolor = G.new_vertex_property("vector<double>")
        G.ep.ecolor = G.new_edge_property("vector<double>")
        G.ep.ewidth = G.new_edge_property("int")
        for v in G.vertices():
            G.vp.vfcolor[v] = [0, 0.0, 0.0, 1.0]
        for e in G.edges():
            G.ep.ewidth[e] = numpy.sqrt(G.ep.weight[e]/16 + 1)
            G.ep.ecolor[e] = [0.179, 0.203, 0.210, 0.8]
        
        return G
    
    def route(self, G, source, target):

        dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())

        path = list()
        path.append(target)

        v = target
        while v != source:
            v = G.vertex(pred[v])
            path.append(v)
        
        return path[::-1]
