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
import networkx

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

# def draw_graph(G, filename="traversability-graph.png", path=[]):

#     G.vp.vfcolor = G.new_vertex_property("vector<double>")
#     G.ep.ecolor = G.new_edge_property("vector<double>")
#     G.ep.ewidth = G.new_edge_property("int")

#     for v in G.vertices():
#         diff = G.vp.traversability[v]
#         G.vp.vfcolor[v] = [1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1/(numpy.sqrt(diff)/100), 1.0]
#     for e in G.edges():
#         G.ep.ewidth[e] = 6
#         G.ep.ecolor[e] = [0.179, 0.203, 0.210, 0.8]
    
#     for i, v in enumerate(path):
#         G.vp.vfcolor[v] = [0, 0.640625, 0, 0.9]
#         if i < len(path) - 1:
#             for e in v.out_edges():
#                 if e.target() == path[i+1]:
#                     G.ep.ecolor[e] = [0, 0.640625, 0, 1]
#                     G.ep.ewidth[e] = 10

#     draw.graph_draw(G, pos=G.vp.pos, output_size=(1200, 1200), vertex_color=[0,0,0,1], vertex_fill_color=G.vp.vfcolor,\
#                     edge_color=G.ep.ecolor, edge_pen_width=G.ep.ewidth, output=filename, edge_marker_size=4)

class RouteEstimator:
    """
    """
    def __init__(self, c=0.7):
        self.c = c

    def tm2graph(self, tmatrix):

        G = networkx.DiGraph()

        for i, row in enumerate(tmatrix):
            for j, element in enumerate(row):
                index = coord2((i,j), tmatrix.shape[1])
                G.add_node( index,
                            pos=[j,i],
                            inv_traversability=float('inf') if tmatrix[i][j] == 0 else (100*(1/tmatrix[i][j]))**2,
                            cut=False
                )
                G.nodes[i]['cut'] = True if G.nodes[i]['inv_traversability'] < (100*(1/self.c))**2 else False

        edges = list()

        for v in [vv for vv in G.nodes() if not G.nodes[vv]['cut']]:

            (i, j) = G.nodes[v]['pos'][1], G.nodes[v]['pos'][0]

            top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            if i-1 > -1:
                u = coord2(top, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+1 < tmatrix.shape[0]:
                u = coord2(bottom, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if j-1 > -1:
                u = coord2(left, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if j+1 < tmatrix.shape[1]:
                u = coord2(right, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))

            topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
            if i-1 > -1 and j-1 > -1:
                u = coord2(topleft, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i-1 > -1 and j+1 < tmatrix.shape[1]:
                u = coord2(topright, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+1 < tmatrix.shape[0] and j-1 > -1:
                u = coord2(bottomleft, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+1 < tmatrix.shape[0] and j+1 < tmatrix.shape[1]:
                u = coord2(bottomright, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))

        G.add_edges_from(edges)

        del edges

        return G

    # def map_from_source(self, G, source):
    #     dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())
    #     return dist, pred

    def route(self, G, source, target):

        path = networkx.shortest_path(G, source, target, 'weight')

        if len(path) == 2:
            found = False
        else: found = True

        return path, found
