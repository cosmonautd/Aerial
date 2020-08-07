import trav
import graphmapx

import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--use", type=int, default=8, required=False, help="Select how many images to use")
args = ap.parse_args()

def main_experiment_overlap():
    """
    """
    import os
    import time
    import json
    # import tqdm
    import numpy
    import itertools

    dataset_path = 'dataset/images/'
    ground_truth_path = 'dataset/labels/'
    positive_keypoints_path = 'dataset/keypoints-reachable/'
    negative_keypoints_path = 'dataset/keypoints-unreachable/'
    output_path = 'output_msc/'
    output_file = 'data-overlap-%d.json' % (args.use)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ['aerial%02d.jpg' % (i+1) for i in range(args.use)]
    f_set = [trav.tf_grayhist]
    r_set = [8]
    c_set = [0.3, 0.4]

    ov = 0.5

    dataset = list()
    for (_, _, filenames) in os.walk(ground_truth_path):
        dataset.extend(filenames)
        break

    selected = list(set(dataset).intersection(images)) if len(images) > 0 else dataset
    selected.sort()

    if os.path.exists(os.path.join(output_path, output_file)):
        with open(os.path.join(output_path, output_file)) as datafile:
            data = json.load(datafile)
    else:
        data = list()

    # for i in tqdm.trange(len(selected), desc="            Input image "):
    for i in range(len(selected)):

        image_path = selected[i]
        image = trav.load_image(os.path.join(dataset_path, image_path))
        ground_truth = trav.load_image(os.path.join(ground_truth_path, image_path))
        positive_keypoints = trav.load_image(os.path.join(positive_keypoints_path, image_path))
        negative_keypoints = trav.load_image(os.path.join(negative_keypoints_path, image_path))

        # for j in tqdm.trange(len(f_set), desc="Traversability function "):
        for j in range(len(f_set)):

            f = f_set[j]

            # for k in tqdm.trange(len(r_set), desc="            Region size "):
            for k in range(len(r_set)):

                r = r_set[k]

                mapper = trav.TraversabilityEstimator(tf=f, r=r, use_overlap=True, overlap=ov)

                start_matrix_time = time.time()
                t_matrix = mapper.get_traversability_matrix(image)
                matrix_time = time.time() - start_matrix_time

                gt_matrix = mapper.get_ground_truth(ground_truth, matrix=True)

                grid = trav.grid_list_overlap(image, r, overlap=ov)

                # for ii in tqdm.trange(len(c_set), desc="          Cut threshold "):
                for ii in range(len(c_set)):

                    c = c_set[ii]

                    print("Processing: %s, tf: %s, r=%d, c=%.1f" % (image_path, f.__name__, r, c))

                    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)

                    start_graph_time = time.time()
                    if f == trav.reference:
                        G = router.tm2graph_overlap(gt_matrix)
                    else:
                        G = router.tm2graph_overlap(t_matrix)
                    graph_time = time.time() - start_graph_time

                    keypoints = graphmapx.get_keypoints_overlap(positive_keypoints, grid, overlap=ov)
                    combinations = list(itertools.combinations(keypoints, 2))

                    # for counter in tqdm.trange(len(combinations), desc="         Positive paths "):
                    for counter in range(len(combinations)):

                        (s, t) = combinations[counter]

                        start_route_time = time.time()
                        path, found = router.route(G, s, t, t_matrix)
                        route_time = time.time() - start_route_time

                        score = trav.score(path, ground_truth, r)

                        results = dict()
                        results['image'] = image_path
                        results['traversability_function'] = f.__name__
                        results['region_size'] = r
                        results['cut_threshold'] = c
                        results['path_existence'] = True
                        results['matrix_build_time'] = matrix_time
                        results['graph_build_time'] = graph_time
                        results['path_build_time'] = route_time
                        results['path_found'] = found
                        results['path_score'] = score if found else 0.0
                        results['path_coordinates'] = [(int(p[0]), int(p[1])) for p in path]
                        results['path_length'] = len(path)

                        data.append(results)

                    keypoints = graphmapx.get_keypoints_overlap(negative_keypoints, grid, overlap=ov)
                    combinations = list(itertools.combinations(keypoints, 2))

                    # for counter in tqdm.trange(len(combinations), desc="         Negative paths "):
                    for counter in range(len(combinations)):

                        (s, t) = combinations[counter]

                        start_route_time = time.time()
                        path, found = router.route(G, s, t, t_matrix)
                        route_time = time.time() - start_route_time

                        results = dict()
                        results['image'] = image_path
                        results['traversability_function'] = f.__name__
                        results['region_size'] = r
                        results['cut_threshold'] = c
                        results['matrix_build_time'] = matrix_time
                        results['graph_build_time'] = graph_time
                        results['path_build_time'] = route_time
                        results['path_existence'] = False
                        results['path_found'] = found
                        results['path_score'] = 1.0 if not found else 0.0
                        results['path_coordinates'] = [(int(p[0]), int(p[1])) for p in path]
                        results['path_length'] = len(path)

                        data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def get_results(datapath, f=trav.tf_grayhist, r=8, c=0.4):
    """
    """
    import json
    import numpy

    with open(datapath) as datafile:
        data = json.load(datafile)
    
    print('r: %d, c: %0.1f' % (r, c))
    print("Total samples:", len(data))

    images = ['aerial%02d.jpg' % (i+1) for i in range(args.use)]

    matrix_time = list()
    graph_time = list()
    route_time = list()
    score = list()
    lengths = list()
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for sample in data:
        if sample['image'] in images:
            if sample['traversability_function'] == f.__name__:
                if sample['region_size'] == r:
                    if sample['cut_threshold'] == c:

                        matrix_time.append(sample['matrix_build_time'])
                        graph_time.append(sample['graph_build_time'])
                        route_time.append(sample['path_build_time'])

                        score.append(sample['path_score'])
                        if sample['path_existence'] and sample['path_found'] and sample['path_score'] > 0.7:
                            lengths.append(sample['path_length'])

                        if sample['path_existence'] and sample['path_found']:
                            true_positive += 1
                        elif not sample['path_existence'] and not sample['path_found']:
                            true_negative += 1
                        elif not sample['path_existence'] and sample['path_found']:
                            false_positive += 1
                        elif sample['path_existence'] and not sample['path_found']:
                            false_negative += 1
    
    matrix_time = numpy.array(matrix_time)
    graph_time = numpy.array(graph_time)
    route_time = numpy.array(route_time)

    print("Avg matrix build time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time), numpy.std(matrix_time)))
    print("Avg graph build time: %.3f (+/- %.3f)" % (numpy.mean(graph_time), numpy.std(graph_time)))
    print("Avg mapping time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time + graph_time), numpy.std(matrix_time + graph_time)))
    print("Avg route build time: %.3f (+/- %.3f)" % (numpy.mean(route_time), numpy.std(route_time)))
    print("Avg total time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time + graph_time + route_time), numpy.std(matrix_time + graph_time + route_time)))

    print("Avg path length: %d" % int(numpy.round(numpy.mean(lengths))))
    print("Avg path quality: %.3f (+/- %.3f)" % (numpy.mean(score), numpy.std(score)))

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)

    frdr = recall
    irdr = true_negative/(true_negative + false_positive)
    fdr  = accuracy

    print("Accuracy: %.3f" % accuracy)
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)

    print("Feasible route detection rate: %.3f" % frdr)
    print("Infeasible route detection rate: %.3f" % irdr)
    print("Feasiblity detection rate: %.3f" % fdr)

    print("Evaluated samples:", len(matrix_time))

if not os.path.exists('output_msc/data-overlap-%d.json' % (args.use)):
    main_experiment_overlap()

get_results('output_msc/data-overlap-%d.json' % (args.use), r=8, c=0.4)