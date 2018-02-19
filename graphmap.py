""" The graphmap module provides methods to generate 
    traversability graphs from traversability matrices.
    Tools for path computation are also be provided.
    Class RouteEstimator: defines a route estimator and its configuration.
    Method tdi2graph: builds a graph from a traversability matrix
    Method route: returns the best route between two regions
    Method draw_graph: saves a graph as an image (optionally, draws paths)
"""

import cv2
import numpy
import matplotlib
import graph_tool as graphtool
import graph_tool.draw as draw
import graph_tool.search as search

def coord2(position, columns):
    """ Converts two-dimensional indexes to one-dimension coordinate
    """
    return position[0]*columns + position[1]

def get_keypoints(image, grid):
    """
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-image
    keypoints = detector.detect(reversemask)

    indexes = list()

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        size = int(keypoint.size)
        for i, (tlx, tly,sqsize) in enumerate(grid):
            if tlx <= x and x < tlx + sqsize:
                if tly <= y and y < tly + sqsize:
                    indexes.append(i)

    return indexes

def draw_graph(G, filename="traversability-graph.png", path=[]):

    G.vp.vfcolor = G.new_vertex_property("vector<double>")
    G.ep.ecolor = G.new_edge_property("vector<double>")
    G.ep.ewidth = G.new_edge_property("int")

    for v in G.vertices():
        diff = G.vp.traversability[v]
        G.vp.vfcolor[v] = [1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1.0]
    for e in G.edges():
        G.ep.ewidth[e] = 6
        G.ep.ecolor[e] = [0.179, 0.203, 0.210, 0.8]
    
    for i, v in enumerate(path):
        G.vp.vfcolor[v] = [0, 0.640625, 0, 0.9]
        if i < len(path) - 1:
            for e in v.out_edges():
                if e.target() == path[i+1]:
                    G.ep.ecolor[e] = [0, 0.640625, 0, 1]
                    G.ep.ewidth[e] = 10

    draw.graph_draw(G, pos=G.vp.pos, output_size=(1200, 1200), vertex_color=[0,0,0,1], vertex_fill_color=G.vp.vfcolor,\
                    edge_color=G.ep.ecolor, edge_pen_width=G.ep.ewidth, output=filename, edge_marker_size=4)

class Visitor(search.DijkstraVisitor):
        """
        """
        def __init__(self, target=None):
            if target is not None:
                self.target = target

        def finish_vertex(self, v):
            if v == self.target:
                raise graphtool.search.StopSearch()

class RouteEstimator:
    """
    """
    def __init__(self, c=0.7):
        self.c = c

    def tm2graph(self, tdmatrix):

        G = graphtool.Graph(directed=True)

        G.vp.pos = G.new_vertex_property("vector<double>")
        G.vp.traversability = G.new_vertex_property("double")
        G.vp.cut = G.new_vertex_property("bool")
        G.ep.weight = G.new_edge_property("double")

        for i, row in enumerate(tdmatrix):
            for j, element in enumerate(row):
                v = G.add_vertex()
                G.vp.pos[v] = [j, i]
                if tdmatrix[i][j] == 0:
                    G.vp.traversability[v] = float('inf')
                else:
                    G.vp.traversability[v] = (100*(1/tdmatrix[i][j]))**2
                if G.vp.traversability[v] > (100*(1/self.c))**2:
                    G.vp.cut[v] = True
                else:
                    G.vp.cut[v] = False

        edges = list()

        for v in [vv for vv in G.vertices() if not G.vp.cut[vv]]:

            (i, j) = G.vp.pos[v][1], G.vp.pos[v][0]

            top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            if i-1 > -1:
                u = G.vertex(coord2(top, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if i+1 < tdmatrix.shape[0]:
                u = G.vertex(coord2(bottom, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if j-1 > -1:
                u = G.vertex(coord2(left, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(right, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))

            topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
            if i-1 > -1 and j-1 > -1:
                u = G.vertex(coord2(topleft, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if i-1 > -1 and j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(topright, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if i+1 < tdmatrix.shape[0] and j-1 > -1:
                u = G.vertex(coord2(bottomleft, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))
            if i+1 < tdmatrix.shape[0] and j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(bottomright, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.traversability[v] + G.vp.traversability[u]))

        G.add_edge_list(edges, eprops=[G.ep.weight])

        del edges

        return G

    def map_from_source(self, G, source):
        source = G.vertex(source)
        dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())
        return dist, pred

    def route(self, G, source, target, dist=None, pred=None):

        source = G.vertex(source)
        target = G.vertex(target)

        if dist and pred:
            pass
        else:
            dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor(target))

        path = list()
        found = False

        v = target
        if G.vertex(pred[target]) and dist[target] != float('inf'):
            found = True
            path.append(target)
            while v != source:
                v = G.vertex(pred[v])
                path.append(v)

        if len(path) == 0:
            path.append(target)
            path.append(source)

        return path[::-1], found
