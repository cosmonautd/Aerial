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
    print("MSE: %.3f" % (((t_image - t_ground_truth)**2).mean(axis=None)))
    trav.show_image([image, t_ground_truth, t_image])

def compute_path_random_keypoints(image_path='dataset/images/aerial01.jpg', 
                                  label_path='dataset/labels/aerial01.jpg',
                                  keypoints_path='dataset/keypoints-reachable/aerial01.jpg',
                                  f=trav.tf_grayhist, r=10, c=0.4, overlap=0.5):
    """
    """
    import random
    import numpy
    import cv2

    use_overlap = (overlap is not None) and (overlap != 0)
    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r, use_overlap=use_overlap, overlap=overlap)

    image = trav.load_image(image_path)
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image(label_path) if label_path is not None else None

    keypoints_image = trav.load_image(keypoints_path)
    if use_overlap:
        grid = trav.grid_list_overlap(image, r, overlap=overlap)
        keypoints = graphmapx.get_keypoints_overlap(keypoints_image, grid, overlap)
    else:
        grid = trav.grid_list(image, r)
        keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    if use_overlap: G = router.tm2graph_overlap(t_matrix)
    else: G = router.tm2graph(t_matrix)

    [source, target] = [v for v in random.sample(keypoints, 2)]

    path, found = router.route(G, source, target, t_matrix)

    if ground_truth is not None:
        score = trav.score(path, ground_truth, r)
        score_str = 'Score: %.2f' % (score)
        fontColor = (255,0,0) if score < 0.7 else (0,255,0)
    else:
        score = None
        score_str = ''
        fontColor = (0,255,0)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (10, 40)
    fontScale              = 1
    lineType               = 2

    path_image = trav.draw_path(image, path, found=found, color=fontColor)

    if score is not None:
        cv2.putText(path_image, score_str, topLeftCornerOfText,
                    font, fontScale, fontColor, lineType)

    trav.show_image([path_image])

def compute_path_all_keypoints(image_path='dataset/images/aerial01.jpg', 
                               label_path='dataset/labels/aerial01.jpg',
                               keypoints_path='dataset/keypoints-reachable/aerial01.jpg',
                               f=trav.tf_grayhist, r=10, c=0.4, overlap=0.5):
    """
    """
    import os
    import itertools
    import numpy
    import cv2

    use_overlap = (overlap is not None) and (overlap != 0)
    tdigenerator = trav.TraversabilityEstimator(tf=f, r=r, use_overlap=use_overlap, overlap=overlap)

    image = trav.load_image(image_path)
    t_matrix = tdigenerator.get_traversability_matrix(image)
    ground_truth = trav.load_image(label_path) if label_path is not None else None

    keypoints_image = trav.load_image(keypoints_path)
    if use_overlap:
        grid = trav.grid_list_overlap(image, r, overlap=overlap)
        keypoints = graphmapx.get_keypoints_overlap(keypoints_image, grid, overlap)
    else:
        grid = trav.grid_list(image, r)
        keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
    if use_overlap: G = router.tm2graph_overlap(t_matrix)
    else: G = router.tm2graph(t_matrix)

    output_path = 'output/paths-%s-%s-r%d-c%02d' % (image_path.split('/')[-1].split('.')[0], f.__name__.replace('_', ''), r, 10*c)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, t_matrix)

        if ground_truth is not None:
            score = trav.score(path, ground_truth, r)
            score_str = 'Score: %.2f' % (score)
            fontColor = (255,0,0) if score < 0.7 else (0,255,0)
        else:
            score = None
            score_str = ''
            fontColor = (0,255,0)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        path_image = trav.draw_path(image, path, found=found, color=fontColor)

        if score is not None:
            cv2.putText(path_image, score_str, topLeftCornerOfText,
                        font, fontScale, fontColor, lineType)

        trav.save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])

def draw_traversability_graph():
    """
    """
    import graphmap
    r = 6
    c = 0.4
    image = trav.load_image('dataset/images/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=r)
    matrix = mapper.get_traversability_matrix(image)
    grid = trav.grid_list(image, r=r)
    router = graphmap.RouteEstimator(c=c)
    G = router.tm2graph(matrix)
    graphmap.draw_graph(G)

for i in range(8):
    i += 1
    compare_with_ground_truth(image_path='aerial%02d.jpg' % (i), r=8)
    compute_path_all_keypoints(image_path='dataset/images/aerial%02d.jpg' % (i),
                               label_path='dataset/labels/aerial%02d.jpg' % (i),
                               keypoints_path='dataset/keypoints-reachable/aerial%02d.jpg' % (i),
                               f=trav.tf_grayhist, r=8, c=0.4)

compute_path_all_keypoints(image_path='fieldtest/aerial-ufc.jpg',
                           label_path=None,
                           keypoints_path='fieldtest/keypoints-ufc.jpg',
                           f=trav.tf_grayhist, r=20, c=0.2)