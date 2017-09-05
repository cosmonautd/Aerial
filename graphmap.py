import gtde
import graph_tool as graphtool
import graph_tool.draw as draw

def coord2(position, columns):
    """ Converts two-dimensional indexes to one-dimension coordinate
    """
    return position[0]*columns + position[1]

estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.colorhistogram)

frame = gtde.loadimage('img/aerial2.jpg')
framematrix = estimator.computematrix(frame)
print(framematrix.shape)

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
            weight[e] = abs(diff[v] - diff[G.vertex(coord2(top, framematrix.shape[1]))])
    if i+1 < framematrix.shape[0]:
        if G.edge(coord2(bottom, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(bottom, framematrix.shape[1])))
            weight[e] = abs(diff[v] - diff[G.vertex(coord2(bottom, framematrix.shape[1]))])
    if j-1 > -1:
        if G.edge(coord2(left, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(left, framematrix.shape[1])))
            weight[e] = abs(diff[v] - diff[G.vertex(coord2(left, framematrix.shape[1]))])
    if j+1 < framematrix.shape[1]:
        if G.edge(coord2(right, framematrix.shape[1]), v) == None:
            e = G.add_edge(v, G.vertex(coord2(right, framematrix.shape[1])))
            weight[e] = abs(diff[v] - diff[G.vertex(coord2(right, framematrix.shape[1]))])

draw.graph_draw(G, pos=pos2, output_size=(1000, 1000), output="tdg.png",\
                edge_pen_width=weight)