import gtde
import matplotlib
import graph_tool as graphtool
import graph_tool.draw as draw
import graph_tool.search as search

def coord2(position, columns):
    """ Converts two-dimensional indexes to one-dimension coordinate
    """
    return position[0]*columns + position[1]

class VisitorExample(search.DijkstraVisitor):

    def __init__(self):
        pass

    def discover_vertex(self, u):
        pass

    def examine_edge(self, e):
        pass

    def edge_relaxed(self, e):
        pass

estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.superpixels)

frame = gtde.loadimage('img/aerial2.jpg')
framematrix = estimator.computematrix(frame)

G = graphtool.Graph(directed=False)
pos = G.new_vertex_property("vector<double>")
pos2 = G.new_vertex_property("vector<double>")
diff = G.new_vertex_property("double")
weight = G.new_edge_property("double")

for i, row in enumerate(framematrix):
    for j, element in enumerate(row):
        v = G.add_vertex()
        pos[v] = [i, j]
        pos2[v] = [j, i]
        diff[v] = framematrix[i][j]

for v in G.vertices():
    (i, j) = pos[v][0], pos[v][1]

    top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
    if i-1 > -1:
        if G.edge(coord2(top, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(top, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(top, framematrix.shape[1]))])
    if i+1 < framematrix.shape[0]:
        if G.edge(coord2(bottom, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(bottom, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(bottom, framematrix.shape[1]))])
    if j-1 > -1:
        if G.edge(coord2(left, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(left, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(left, framematrix.shape[1]))])
    if j+1 < framematrix.shape[1]:
        if G.edge(coord2(right, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(right, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(right, framematrix.shape[1]))])
    
    topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
    if i-1 > -1 and j-1 > -1:
        if G.edge(coord2(topleft, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(topleft, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(topleft, framematrix.shape[1]))])
    if i-1 > -1 and j+1 < framematrix.shape[1]:
        if G.edge(coord2(topright, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(topright, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(topright, framematrix.shape[1]))])
    if i+1 < framematrix.shape[0] and j-1 > -1:
        if G.edge(coord2(bottomleft, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(bottomleft, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(bottomleft, framematrix.shape[1]))])
    if i+1 < framematrix.shape[0] and j+1 < framematrix.shape[1]:
        if G.edge(coord2(bottomright, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(bottomright, framematrix.shape[1])))
            weight[e] = (diff[v] + diff[G.vertex(coord2(bottomright, framematrix.shape[1]))])

vfcolor = G.new_vertex_property("vector<double>")
ecolor = G.new_edge_property("vector<double>")
ewidth = G.new_edge_property("int")
for v in G.vertices():
    vfcolor[v] = [0, 0.0, 0.0, 1.0]
for e in G.edges():
    ewidth[e] = weight[e]/16 + 1
    ecolor[e] = [0.179, 0.203, 0.210, 0.8]

draw.graph_draw(G, pos=pos2, output_size=(1200, 1200), vertex_fill_color=vfcolor,\
                edge_color=ecolor, edge_pen_width=ewidth, output="tdg.png")

source = G.vertex(coord2((12, 1), framematrix.shape[1]))
target = G.vertex(coord2((4, 14), framematrix.shape[1]))

dist, pred = search.dijkstra_search(G, weight, source, VisitorExample())

v = target
vfcolor[v] = [0.640625, 0, 0, 0.9]
while v != source:
    p = G.vertex(pred[v])
    for e in v.out_edges():
        if e.target() == p:
            ecolor[e] = [0.640625, 0, 0, 0.9]
            ewidth[e] = 16
    v = p
    vfcolor[v] = [0.640625, 0, 0, 0.9]

draw.graph_draw(G, pos=pos2, output_size=(1200, 1200), vertex_fill_color=vfcolor,\
                edge_color=ecolor, edge_pen_width=ewidth, output="dijkstra.png")