import trav
import graphmapx

def compute_traversability_matrix():
    """
    """
    image = trav.load_image('dataset/images/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=10)
    matrix = mapper.get_traversability_matrix(image)
    print(matrix)

def compute_traversability_image():
    """
    """
    image = trav.load_image('dataset/images/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=10)
    t_image = mapper.get_traversability_image(image)
    trav.show_image([image, t_image])

def compare_with_ground_truth(image_path='aerial01.jpg', r=10):
    """
    """
    import cv2
    mapper = trav.TraversabilityEstimator(r=r)
    image = trav.load_image('dataset/images/'+image_path)
    h, w, d = image.shape
    ground_truth = trav.load_image('dataset/labels/'+image_path)
    t_image = mapper.get_traversability_matrix(image)
    t_ground_truth = cv2.resize(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY), (h//r, w//r))/255
    print("MSE: %.4f" % (((t_image - t_ground_truth)**2).mean(axis=None)))
    trav.show_image([image, t_ground_truth, t_image])

def compute_path_random_keypoints():
    """
    """
    import random
    import numpy
    import cv2

    r = 10
    c = 0.4
    f = trav.tf_grayhist

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r)

    image = trav.load_image('dataset/images/aerial01.jpg')
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image('dataset/labels/aerial01.jpg')

    keypoints_image = trav.load_image('dataset/keypoints-reachable/aerial01.jpg')
    grid = trav.grid_list(image, r)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    G = router.tm2graph(t_matrix)

    [source, target] = [v for v in random.sample(keypoints, 2)]

    path, found = router.route(G, source, target, t_matrix)
    # graphmap.draw_graph(G, 'output/path-graph.pdf', path) # only graph-tool

    score = trav.score(path, ground_truth, r)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (10, 40)
    fontScale              = 1
    lineType               = 2

    fontColor = (255,0,0) if score < 0.7 else (0,255,0)

    path_image = trav.draw_path(image, path, found=found, color=fontColor)
    cv2.putText(path_image, 'Score: %2.f' % (score),
                topLeftCornerOfText, font, fontScale, fontColor, lineType)

    trav.show_image([path_image])

def compute_path_all_keypoints(r=10, c=0.4, f=trav.tf_grayhist, image_path='aerial01.jpg'):
    """
    """
    import os
    import itertools
    import numpy
    import cv2

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r)

    image = trav.load_image(os.path.join('dataset/images', image_path))
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image(os.path.join('dataset/labels', image_path))

    keypoints_image = trav.load_image(os.path.join('dataset/keypoints-reachable', image_path))
    grid = trav.grid_list(image, r)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    G = router.tm2graph(t_matrix)

    output_path = 'output/paths-%s-%d-0%d-%s' % (f.__name__, r, 10*c, image_path.split('.')[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, t_matrix)

        score = trav.score(path, ground_truth, r)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        fontColor = (255,0,0) if score < 0.7 else (0,255,0)

        path_image = trav.draw_path(image, path, found=found, color=fontColor)
        cv2.putText(path_image, 'Score: %.2f' % (score),
                    topLeftCornerOfText, font, fontScale, fontColor, lineType)

        trav.save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])

def compute_path_random_keypoints_overlap():
    """
    """
    import random
    import cv2

    r = 10
    c = 0.4
    f = trav.tf_grayhist
    ov = 2

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r, overlap=True, ov=ov)

    image = trav.load_image('dataset/images/aerial01.jpg')
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image('dataset/labels/aerial01.jpg')

    keypoints_image = trav.load_image('dataset/keypoints-reachable/aerial01.jpg')
    grid = trav.grid_list_overlap(image, r, ov=ov)
    keypoints = graphmapx.get_keypoints_overlap(keypoints_image, grid, ov=ov)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    G = router.tm2graph_overlap(t_matrix)

    [source, target] = [v for v in random.sample(keypoints, 2)]

    path, found = router.route(G, source, target, t_matrix)
    # graphmap.draw_graph(G, 'output/path-graph.pdf', path) # only graph-tool

    score = trav.score(path, ground_truth, r)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (10, 40)
    fontScale              = 1
    lineType               = 2

    fontColor = (255,0,0) if score < 0.7 else (0,255,0)

    path_image = trav.draw_path(image, path, found=found, color=fontColor)
    cv2.putText(path_image, 'Score: %2.f' % (score),
                topLeftCornerOfText, font, fontScale, fontColor, lineType)

    trav.show_image([path_image])

def compute_path_all_keypoints_overlap(r=10, c=0.4, f=trav.tf_grayhist, image_path='aerial01.jpg'):
    """
    """
    import os
    import itertools
    import cv2

    ov = 2

    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r, overlap=True, ov=ov)

    image = trav.load_image(os.path.join('dataset/images', image_path))
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image(os.path.join('dataset/labels', image_path))

    keypoints_image = trav.load_image(os.path.join('dataset/keypoints-reachable', image_path))
    grid = trav.grid_list_overlap(image, r, ov=ov)
    keypoints = graphmapx.get_keypoints_overlap(keypoints_image, grid, ov=ov)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    G = router.tm2graph_overlap(t_matrix)

    output_path = 'output/paths-%s-%d-0%d-%s' % (f.__name__, r, 10*c, image_path.split('.')[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, t_matrix)

        score = trav.score(path, ground_truth, r)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        fontColor = (255,0,0) if score < 0.7 else (0,255,0)

        path_image = trav.draw_path(image, path, found=found, color=fontColor)
        cv2.putText(path_image, 'Score: %.2f' % (score),
                    topLeftCornerOfText, font, fontScale, fontColor, lineType)

        trav.save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])
