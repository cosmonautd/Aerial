import trav
import graphmapx

def compute_traversability_matrix():
    """
    """
    image = trav.load_image('image/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    matrix = mapper.get_traversability_matrix(image)
    print(matrix)

def compute_traversability_image():
    """
    """
    image = trav.load_image('image/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    t_image = mapper.get_traversability_image(image)
    trav.show_image([t_image])

def compare_with_ground_truth():
    """
    """
    mapper = trav.TraversabilityEstimator(r=6)
    image = trav.load_image('image/aerial01.jpg')
    ground_truth = trav.load_image('ground-truth/aerial01.jpg')
    t_image = mapper.get_traversability_image(image)
    t_ground_truth = mapper.get_ground_truth(ground_truth)
    print("Correlation:", mapper.error(t_image, t_ground_truth, 'corr'))
    trav.show_image([t_image, t_ground_truth])

def compute_path_random_keypoints():
    """
    """
    import random

    r = 6
    c = 0.4
    f = trav.tf_grayhist

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r)

    image = trav.load_image('image/aerial01.jpg')
    t_matrix = tdigenerator.get_traversability_matrix(image)

    keypoints_image = trav.load_image('keypoints-positive/aerial01.jpg')
    grid = trav.grid_list(image, r)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(c=c)
    G = router.tm2graph(t_matrix)

    [source, target] = [v for v in random.sample(keypoints, 2)]

    path, found = router.route(G, source, target)
    # graphmap.draw_graph(G, 'output/path-graph.pdf', path) # only graph-tool

    path_indexes = [int(v) for v in path]
    path_image = trav.draw_path(image, path_indexes, grid, found=found)
    trav.show_image([path_image])

def compute_path_all_keypoints():
    """
    """
    import os
    import itertools

    r = 6
    c = 0.4
    f = trav.tf_grayhist
    image_path = 'aerial01.jpg'

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r)

    image = trav.load_image(os.path.join('image', image_path))
    t_matrix = tdigenerator.get_traversability_matrix(image)

    keypoints_image = trav.load_image(os.path.join('keypoints-positive', image_path))
    grid = trav.grid_list(image, r)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(c=c)
    G = router.tm2graph(t_matrix)

    output_path = 'output/paths-%s-%d-0%d-%s' % (f.__name__, r, 100*c, image_path.split('.')[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t)

        path_indexes = [int(v) for v in path]
        path_image = trav.draw_path(image, path_indexes, grid, found=found)
        trav.save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])

compute_path_all_keypoints()