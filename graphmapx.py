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
import scipy.interpolate

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
        for i, (tlx, tly, sqsize) in enumerate(grid):
            if tlx <= x and x < tlx + sqsize:
                if tly <= y and y < tly + sqsize:
                    indexes.append(i)

    return indexes

def get_keypoints_overlap(image, grid, ov=3):
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
        for i, (tlx, tly, sqsize) in enumerate(grid):
            if tlx <= x and x < tlx + sqsize/ov:
                if tly <= y and y < tly + sqsize/ov:
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

# Ramer-Douglas-Peucker from https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line

def _vec2d_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _vec2d_sub(p1, p2):
    return (p1[0]-p2[0], p1[1]-p2[1])

def _vec2d_mult(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

def ramerdouglas(line, dist):
    """Does Ramer-Douglas-Peucker simplification of a curve with 'dist' threshold.

    'line' is a list-of-tuples, where each tuple is a 2D coordinate

    Usage is like so:

    >>> myline = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
    >>> simplified = ramerdouglas(myline, dist = 1.0)
    """

    if len(line) < 3:
        return line

    (begin, end) = (line[0], line[-1]) if line[0] != line[-1] else (line[0], line[-2])

    distSq = []
    for curr in line[1:-1]:
        tmp = (
            _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin),
            _vec2d_sub(curr, begin)) ** 2 / _vec2d_dist(begin, end))
        distSq.append(tmp)

    maxdist = max(distSq)
    if maxdist < dist ** 2:
        return [begin, end]

    pos = distSq.index(maxdist)
    return (ramerdouglas(line[:pos + 2], dist) + 
            ramerdouglas(line[pos + 1:], dist)[1:])

class RouteEstimator:
    """
    """
    def __init__(self, r, c, grid):
        self.r = r
        self.c = c
        self.grid = grid

    def tm2graph(self, tmatrix):

        G = networkx.DiGraph()

        for i, row in enumerate(tmatrix):
            for j, element in enumerate(row):
                index = coord2((i,j), tmatrix.shape[1])
                G.add_node( index,
                            pos=[j,i],
                            inv_traversability=float('inf') if tmatrix[i][j] == 0 else (100*(1/tmatrix[i][j]))**2,
                            cut=True if tmatrix[i][j] < self.c else False
                )

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
    
    def tm2graph_overlap(self, tmatrix):

        G = networkx.DiGraph()

        for i, row in enumerate(tmatrix):
            for j, element in enumerate(row):
                index = coord2((i,j), tmatrix.shape[1])
                G.add_node( index,
                            pos=[j,i],
                            inv_traversability=float('inf') if tmatrix[i][j] == 0 else (100*(1/tmatrix[i][j]))**2,
                            cut=True if tmatrix[i][j] < self.c else False
                )

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

            top2, bottom2, left2, right2 = (i-2, j), (i+2, j), (i, j-2), (i, j+2)
            if i-2 > -1:
                u = coord2(top2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+2 < tmatrix.shape[0]:
                u = coord2(bottom2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if j-2 > -1:
                u = coord2(left2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if j+2 < tmatrix.shape[1]:
                u = coord2(right2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            
            topleft2, topright2, bottomleft2, bottomright2 = (i-2, j-2), (i-2, j+2), (i+2, j-2), (i+2, j+2)
            if i-2 > -1 and j-2 > -1:
                u = coord2(topleft2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i-2 > -1 and j+2 < tmatrix.shape[1]:
                u = coord2(topright2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+2 < tmatrix.shape[0] and j-2 > -1:
                u = coord2(bottomleft2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))
            if i+2 < tmatrix.shape[0] and j+2 < tmatrix.shape[1]:
                u = coord2(bottomright2, tmatrix.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append((v, u, {'weight' : G.nodes[v]['inv_traversability'] + G.nodes[u]['inv_traversability']}))

        G.add_edges_from(edges)

        del edges

        return G

    # def map_from_source(self, G, source):
    #     dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())
    #     return dist, pred

    # def route(self, G, source, target):
    #     try:
    #         path = networkx.shortest_path(G, source, target, 'weight')
    #         found = True
    #     except networkx.exception.NetworkXNoPath:
    #         path = [source, target]
    #         found = False
    #     return path, found    

    def route(self, G, source, target, tmatrix):
        def dist(a, b):
            (y1, x1) = G.nodes[a]['pos'][1], G.nodes[a]['pos'][0]
            (y2, x2) = G.nodes[b]['pos'][1], G.nodes[b]['pos'][0]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        def dist2(a, b):
            (y1, x1) = G.nodes[a]['pos'][1], G.nodes[a]['pos'][0]
            (y2, x2) = G.nodes[b]['pos'][1], G.nodes[b]['pos'][0]
            xs, xe, ys, ye = 0, 0, 0, 0
            if x1 < x2: xs, xe = x1, x2
            else: xs, xe = x2, x1
            if y1 < y2: ys, ye = y1, y2
            else: ys, ye = y2, y1
            return numpy.mean(tmatrix[ys:ye, xs:xe])
        try:
            path = networkx.astar_path(G, source, target, dist, 'weight')
            centers = list()
            for k in path:
                tly, tlx, size = self.grid[k]
                centers.append((int(tlx+(size/2)), int(tly+(size/2))))
            # Ramer-Douglas-Peucker from https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line
            centers = ramerdouglas(centers, self.r*1.5)
            # Linear interpolation from https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            X = numpy.array(centers)
            alpha = numpy.linspace(0, 1, len(path))
            distance = numpy.cumsum(numpy.sqrt(numpy.sum(numpy.diff(X, axis=0)**2, axis=1)))
            distance = numpy.insert(distance, 0, 0)/distance[-1]
            interpolator =  scipy.interpolate.interp1d(distance, X, kind='slinear', axis=0)
            curve = interpolator(alpha)
            curve = numpy.round(curve).astype(int)
            # Spline smoothing from https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            X = numpy.array(curve)
            distance = numpy.cumsum(numpy.sqrt(numpy.sum(numpy.diff(X, axis=0)**2, axis=1)))
            distance = numpy.insert(distance, 0, 0)/distance[-1]
            splines = [scipy.interpolate.UnivariateSpline(distance, coords, k=2) for coords in X.T]
            points_fitted = numpy.vstack( spl(alpha) for spl in splines ).T
            points_fitted = numpy.round(points_fitted).astype(int)
            centers = points_fitted
            # TODO: FIX WORKAROUND FOR IMAGES 1000x1000
            #centers = numpy.clip(centers, 0, 999)
            # Returning pixel coordinates
            path = centers
            found = True
        except (networkx.exception.NetworkXNoPath, ValueError):
            path = [source, target]
            centers = list()
            for k in path:
                tly, tlx, size = self.grid[k]
                centers.append((int(tlx+(size/2)), int(tly+(size/2))))
            path = centers
            found = False
        return path, found
