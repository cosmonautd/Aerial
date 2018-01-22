import cv2
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

def label2keypoints(image, grid):
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

def drawgraph(G, path=[], filename="tdg.png"):
    for i, v in enumerate(path):
        G.vp.vfcolor[v] = [0, 0.640625, 0, 0.9]
        if i < len(path) - 1:
            for e in v.out_edges():
                if e.target() == path[i+1]:
                    G.ep.ecolor[e] = [0, 0.640625, 0, 0.9]
                    G.ep.ewidth[e] = 6

    draw.graph_draw(G, pos=G.vp.pos2, output_size=(1200, 1200), vertex_fill_color=G.vp.vfcolor,\
                    edge_color=G.ep.ecolor, edge_pen_width=G.ep.ewidth, output=filename)

class Visitor(search.DijkstraVisitor):

    def __init__(self, target):
        self.target = target

    def finish_vertex(self, v):
        if v == self.target:
            raise graphtool.search.StopSearch()

class RouteEstimator:

    def __init__(self):
        pass

    def tdm2graph(self, tdmatrix, confidence=0.7):

        G = graphtool.Graph(directed=False)

        G.vp.pos = G.new_vertex_property("vector<double>")
        G.vp.pos2 = G.new_vertex_property("vector<double>")
        G.vp.diff = G.new_vertex_property("double")
        G.vp.cut = G.new_vertex_property("bool")
        G.ep.weight = G.new_edge_property("double")

        for i, row in enumerate(tdmatrix):
            for j, element in enumerate(row):
                v = G.add_vertex()
                G.vp.pos[v] = [i, j]
                G.vp.pos2[v] = [j, i]
                if tdmatrix[i][j] == 0:
                    G.vp.diff[v] = float('inf')
                else:
                    G.vp.diff[v] = (100*(1/tdmatrix[i][j]))**2
                if G.vp.diff[v] > (100*(1/confidence))**2:
                    G.vp.cut[v] = True
                else:
                    G.vp.cut[v] = False

        edges = list()

        for v in [vv for vv in G.vertices() if not G.vp.cut[vv]]:

            (i, j) = G.vp.pos[v][0], G.vp.pos[v][1]

            top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            if i-1 > -1:
                u = G.vertex(coord2(top, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if i+1 < tdmatrix.shape[0]:
                u = G.vertex(coord2(bottom, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if j-1 > -1:
                u = G.vertex(coord2(left, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(right, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))

            topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
            if i-1 > -1 and j-1 > -1:
                u = G.vertex(coord2(topleft, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if i-1 > -1 and j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(topright, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if i+1 < tdmatrix.shape[0] and j-1 > -1:
                u = G.vertex(coord2(bottomleft, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))
            if i+1 < tdmatrix.shape[0] and j+1 < tdmatrix.shape[1]:
                u = G.vertex(coord2(bottomright, tdmatrix.shape[1]))
                if not G.vp.cut[u]:
                    edges.append((v, u, G.vp.diff[v] + G.vp.diff[u]))

        G.add_edge_list(edges, eprops=[G.ep.weight])

        del edges

        G.vp.vfcolor = G.new_vertex_property("vector<double>")
        G.ep.ecolor = G.new_edge_property("vector<double>")
        G.ep.ewidth = G.new_edge_property("int")
        for v in G.vertices():
            diff = G.vp.diff[v]
            G.vp.vfcolor[v] = [1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1.0]
        for e in G.edges():
            G.ep.ewidth[e] = 2
            G.ep.ecolor[e] = [0.179, 0.203, 0.210, 0.8]

        return G

    def routeall(self, G, source):
        dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())
        return dist, pred

    def route(self, G, source, target, dist=None, pred=None):

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
