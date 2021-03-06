import trav
import graphmapx

def traversability_matrix_histogram_plot():
    """
    """
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_style("darkgrid")
    matplotlib.rcParams.update({'font.size': 14})

    image = trav.load_image('dataset/images/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    matrix = mapper.get_traversability_matrix(image, normalize=True)

    fig, ax = plt.subplots(1,1)

    weights = numpy.ones_like(matrix.flatten())/float(len(matrix.flatten()))
    n, _, _ = ax.hist(matrix.flatten(), bins=numpy.arange(0, 1 + 0.1, 0.1), weights=weights, facecolor='green', alpha=0.5)

    ax.set_xticks(numpy.arange(0, 1.01, 0.1))
    ax.set_xlim([0, 1])
    ax.set_yticks(numpy.arange(0.05, max(n)+0.05, 0.05))
    ax.set_yticklabels(["%.0f%%" % (100*x) for x in numpy.arange(0.05, max(n)+0.05, 0.05)])
    ax.grid(True)
    fig.tight_layout()
    plt.show()

def traversability_image_correlation_plot():
    """
    """
    import os
    import numpy
    import matplotlib.pyplot as plt

    dt_path = 'dataset/images'
    gt_path = 'dataset/labels'
    output_path = 'output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    measure = 'corr'
    f_set = [trav.tf_grayhist, trav.tf_rgbhist, trav.tf_superpixels]
    r_set = [6, 12]

    dataset = list()
    for (_, _, filenames) in os.walk(gt_path):
        dataset.extend(filenames)
        break
    
    dataset.sort(key=str.lower)

    data = dict()
    data[measure] = dict()

    for image_path in dataset:

        image = trav.load_image(os.path.join(dt_path, image_path))
        ground_truth = trav.load_image(os.path.join(gt_path, image_path))

        for f in f_set:

            if not f.__name__ in data[measure]:
                data[measure][f.__name__] = dict()

            for r in r_set:

                print("Running %s %s %s %s" % (measure, image_path, f.__name__, str(r)))
                
                if not str(r) in data[measure][f.__name__]:
                    data[measure][f.__name__][str(r)] = list()

                mapper = trav.TraversabilityEstimator(tf=f, r=r)
                
                t_image = mapper.get_traversability_matrix(image)
                t_ground_truth = mapper.get_ground_truth(ground_truth, matrix=True)
                data[measure][f.__name__][str(r)].append(mapper.error(t_image, t_ground_truth, measure))
    
    plot_title = {  
        "corr" : "Pearson's correlation coefficient", 
        "jaccard" : "Generalized Jaccard similarity index", 
        "rmse" : "Root mean square error", 
        "nrmse" : "Normalized root mean square error", 
        "psnr" : "Peak signal to noise ratio", 
        "ssim" : "Structural similarity index",
        "nmae" : "Normalized mean absolute error"
    }
    
    ft_curve = {
        "tf_random" : "Random",
        "tf_grayhist" : "Gray Histogram",
        "tf_rgbhist" : "RGB Histogram",
        "tf_superpixels" : "Superpixels"
    }

    fig, (ax0) = plt.subplots(ncols=1)
    for f in f_set:
        x = numpy.array(r_set)
        y = numpy.array([numpy.mean(data[measure][f.__name__][str(element)]) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ft_curve[f.__name__])
    plt.title(plot_title[measure])
    #ax0.legend(loc='upper left')
    #ax0.set_xscale('log')
    ax0.set_xlabel("r")
    ax0.set_ylabel(plot_title[measure].split(" ")[-1].title())
    ax0.set_xticks(r_set)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in r_set])
    ax0.tick_params(axis='x', which='minor', bottom='off')
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "score-traversability_image-%s.png" % (measure)), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main_experiment():
    """
    """
    import os
    import time
    import json
    # import tqdm
    import numpy
    import itertools
    import scipy.interpolate

    dataset_path = 'dataset/images/'
    ground_truth_path = 'dataset/labels/'
    positive_keypoints_path = 'dataset/keypoints-reachable/'
    negative_keypoints_path = 'dataset/keypoints-unreachable/'
    output_path = 'output/'
    output_file = 'data.json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

                mapper = trav.TraversabilityEstimator(tf=f, r=r)

                start_matrix_time = time.time()
                t_matrix = mapper.get_traversability_matrix(image)
                matrix_time = time.time() - start_matrix_time

                gt_matrix = mapper.get_ground_truth(ground_truth, matrix=True)

                grid = trav.grid_list(image, r)

                # for ii in tqdm.trange(len(c_set), desc="          Cut threshold "):
                for ii in range(len(c_set)):

                    c = c_set[ii]

                    print("Processing: %s, tf: %s, r=%d, c=%.1f" % (image_path, f.__name__, r, c))

                    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)

                    start_graph_time = time.time()
                    if f == trav.reference:
                        G = router.tm2graph(gt_matrix)
                    else:
                        G = router.tm2graph(t_matrix)
                    graph_time = time.time() - start_graph_time

                    keypoints = graphmapx.get_keypoints(positive_keypoints, grid)
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

                        data.append(results)

                    keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
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

                        data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

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
    output_path = 'output/'
    output_file = 'data-overlap.json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

                        data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def main_experiment_multiscale():
    """
    """
    import os
    import time
    import json
    # import tqdm
    import numpy
    import itertools
    import scipy.interpolate

    dataset_path = 'dataset/images/'
    ground_truth_path = 'dataset/labels/'
    positive_keypoints_path = 'dataset/keypoints-reachable/'
    negative_keypoints_path = 'dataset/keypoints-unreachable/'
    output_path = 'output/'
    output_file = 'data-multiscale.json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

                mapper = trav.TraversabilityEstimator(tf=f, r=r)

                start_matrix_time = time.time()
                t_matrix = mapper.get_traversability_matrix_multiscale(image)
                matrix_time = time.time() - start_matrix_time

                gt_matrix = mapper.get_ground_truth(ground_truth, matrix=True)

                grid = trav.grid_list(image, r)

                # for ii in tqdm.trange(len(c_set), desc="          Cut threshold "):
                for ii in range(len(c_set)):

                    c = c_set[ii]

                    print("Processing: %s, tf: %s, r=%d, c=%.1f" % (image_path, f.__name__, r, c))

                    router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)

                    start_graph_time = time.time()
                    if f == trav.reference:
                        G = router.tm2graph(gt_matrix)
                    else:
                        G = router.tm2graph(t_matrix)
                    graph_time = time.time() - start_graph_time

                    keypoints = graphmapx.get_keypoints(positive_keypoints, grid)
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

                        data.append(results)

                    keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
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

                        data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def main_experiment_overlap_multiscale():
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
    output_path = 'output/'
    output_file = 'data-overlap-multiscale.json'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
                t_matrix = mapper.get_traversability_matrix_multiscale(image)
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

                        data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def heatmaps_plot(datapath):
    """
    """
    import os
    import json
    import numpy
    import pandas
    import seaborn
    import matplotlib.pyplot as plt

    output_path = 'output/'

    with open(datapath) as datafile:
        data = json.load(datafile)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Average path quality plot

    heatmatrix = -numpy.ones((len(r_set), len(c_set)))
    counter = numpy.ones((len(r_set), len(c_set)))

    for sample in data:
        if sample['image'] in images \
            and sample['traversability_function'] in [ft.__name__ for ft in f_set] \
            and sample['path_found'] == True:
            r = r_set.index(sample['region_size'])
            c = c_set.index(sample['cut_threshold'])
            if heatmatrix[r][c] == -1: heatmatrix[r][c] = 0
            heatmatrix[r][c] += sample['path_score']
            counter[r][c] += 1

    heatmatrix /= counter

    f1 = plt.figure()
    df1 = pandas.DataFrame(heatmatrix, index=r_set, columns=c_set)
    seaborn.heatmap(df1, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")
    plt.xlabel("c")
    plt.ylabel("r")
    f1.savefig(os.path.join(output_path, datapath[7:-5]+"-path_quality.pdf"), dpi=300, bbox_inches='tight')

    # Feasible path detection plot

    heatmatrix = numpy.zeros((len(r_set), len(c_set)))
    counter = numpy.ones((len(r_set), len(c_set)))

    for sample in data:
        if sample['image'] in images \
            and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
            r = r_set.index(sample['region_size'])
            c = c_set.index(sample['cut_threshold'])
            if sample['path_existence'] == True:
                counter[r][c] += 1
                if sample['path_found'] == True:
                    heatmatrix[r][c] += 1

    heatmatrix /= counter

    f2 = plt.figure()
    df2 = pandas.DataFrame(heatmatrix, index=r_set, columns=c_set)
    seaborn.heatmap(df2, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")
    plt.xlabel("c")
    plt.ylabel("r")
    f2.savefig(os.path.join(output_path, datapath[7:-5]+"-path_positives.pdf"), dpi=300, bbox_inches='tight')

    # Infeasible path detection plot

    heatmatrix = numpy.zeros((len(r_set), len(c_set)))
    counter = numpy.ones((len(r_set), len(c_set)))

    for sample in data:
        if sample['image'] in images \
            and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
            r = r_set.index(sample['region_size'])
            c = c_set.index(sample['cut_threshold'])
            if sample['path_existence'] == False:
                counter[r][c] += 1
                if sample['path_found'] == False:
                    heatmatrix[r][c] += 1

    heatmatrix /= counter

    f3 = plt.figure()
    df3 = pandas.DataFrame(heatmatrix, index=r_set, columns=c_set)
    seaborn.heatmap(df3, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")
    plt.xlabel("c")
    plt.ylabel("r")
    f3.savefig(os.path.join(output_path, datapath[7:-5]+"-path_negatives.pdf"), dpi=300, bbox_inches='tight')

    # Feasibility detection plot

    heatmatrix = numpy.zeros((len(r_set), len(c_set)))
    counter = numpy.ones((len(r_set), len(c_set)))

    for sample in data:
        if sample['image'] in images \
            and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
            r = r_set.index(sample['region_size'])
            c = c_set.index(sample['cut_threshold'])
            counter[r][c] += 1
            if sample['path_existence'] == sample['path_found']:
                heatmatrix[r][c] += 1

    heatmatrix /= counter

    f4 = plt.figure()
    df4 = pandas.DataFrame(heatmatrix, index=r_set, columns=c_set)
    seaborn.heatmap(df4, vmin=0.5, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")
    plt.xlabel("c")
    plt.ylabel("r")
    f4.savefig(os.path.join(output_path, datapath[7:-5]+"-path_feasibility.pdf"), dpi=300, bbox_inches='tight')

def execution_time_plot(datapath):
    """
    """
    import os
    import json
    import numpy
    import seaborn
    import matplotlib.pyplot as plt

    seaborn.set_style("darkgrid")

    output_path = 'output/'

    with open(datapath) as datafile:
        data = json.load(datafile)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    variables = ['matrix_build_time', 'graph_build_time', 'path_build_time']
    
    for var in variables + ['total_time']:

        info_time = dict()
        for c in c_set:
            info_time[str(c)] = dict()
            for r in r_set:
                info_time[str(c)][str(r)] = list()

        for sample in data:
            if sample['image'] in images \
                and sample['region_size'] in r_set and sample['cut_threshold'] in c_set \
                and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
                if var in variables:
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sample[var])
                elif var == 'total_time':
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sum([sample[v] for v in variables]))

        fig, (ax0) = plt.subplots(ncols=1)
        for c in c_set:
            x = numpy.array(r_set)
            y = numpy.array([numpy.mean(info_time[str(c)][str(element)]) for element in x])
            yerr_up = numpy.array([-y[i]+numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] >= y[i]]) for i, element in enumerate(x)])
            yerr_down = numpy.array([y[i]-numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] < y[i]]) for i, element in enumerate(x)])
            ax0.errorbar(x, y, capsize=0, fmt='--o', markevery=range(len(x)), label="c = "+str(c))

        ax0.legend(loc='upper right')
        ax0.set_xlabel("Region size")
        ax0.tick_params(axis='x', which='minor', bottom='off')
        ax0.set_xticks(r_set)
        ax0.set_xticklabels(["%dx%d" % (r,r) for r in r_set])
        ax0.set_ylabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, datapath[7:-5]+"-%s.pdf" % var), dpi=300, bbox_inches='tight')
        plt.close(fig)

def execution_time_plot_combined(datapath1, datapath2):
    """
    """
    import os
    import json
    import numpy
    import seaborn
    import matplotlib.pyplot as plt

    seaborn.set_style("darkgrid")

    output_path = 'output/'

    with open(datapath1) as datafile:
        data1 = json.load(datafile)
    
    with open(datapath2) as datafile:
        data2 = json.load(datafile)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    f_set = [trav.tf_grayhist]
    r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    c_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    variables = ['matrix_build_time', 'graph_build_time', 'path_build_time']
    
    for var in variables + ['total_time']:

        info_time = dict()
        for c in c_set:
            info_time[str(c)] = dict()
            for r in r_set:
                info_time[str(c)][str(r)] = list()

        for sample in data1:
            if sample['image'] in images \
                and sample['region_size'] in r_set and sample['cut_threshold'] in c_set \
                and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
                if var in variables:
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sample[var])
                elif var == 'total_time':
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sum([sample[v] for v in variables]))

        fig, (ax0) = plt.subplots(ncols=1)
        for c in c_set:
            x = numpy.array(r_set)
            y = numpy.array([numpy.mean(info_time[str(c)][str(element)]) for element in x])
            yerr_up = numpy.array([-y[i]+numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] >= y[i]]) for i, element in enumerate(x)])
            yerr_down = numpy.array([y[i]-numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] < y[i]]) for i, element in enumerate(x)])
            ax0.errorbar(x, y, capsize=0, fmt='--o', markevery=range(len(x)), label="c = "+str(c))
        

        info_time = dict()
        for c in c_set:
            info_time[str(c)] = dict()
            for r in r_set:
                info_time[str(c)][str(r)] = list()

        for sample in data2:
            if sample['image'] in images \
                and sample['region_size'] in r_set and sample['cut_threshold'] in c_set \
                and sample['traversability_function'] in [ft.__name__ for ft in f_set]:
                if var in variables:
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sample[var])
                elif var == 'total_time':
                    info_time[str(sample['cut_threshold'])][str(sample['region_size'])].append(sum([sample[v] for v in variables]))

        # fig, (ax0) = plt.subplots(ncols=1)
        for c in c_set:
            x = numpy.array(r_set)
            y = numpy.array([numpy.mean(info_time[str(c)][str(element)]) for element in x])
            yerr_up = numpy.array([-y[i]+numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] >= y[i]]) for i, element in enumerate(x)])
            yerr_down = numpy.array([y[i]-numpy.mean([info_time[str(c)][str(element)][j] for j in range(len(info_time[str(c)][str(element)])) if info_time[str(c)][str(element)][j] < y[i]]) for i, element in enumerate(x)])
            ax0.errorbar(x, y, capsize=0, fmt='--X', markevery=range(len(x)), label="c = "+str(c))
        

        ax0.legend(loc='upper right', title='     No overlap            Overlap       ', ncol=2)
        ax0.set_xlabel("Region size")
        ax0.tick_params(axis='x', which='minor', bottom='off')
        ax0.set_xticks(r_set)
        ax0.set_xticklabels(["%dx%d" % (r,r) for r in r_set])
        ax0.set_ylabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "%s.pdf" % var), dpi=300, bbox_inches='tight')
        plt.close(fig)

def average_time_for_param_combination(datapath, f=trav.tf_grayhist, r=10, c=0.4):
    """
    """
    import json
    import numpy

    with open(datapath) as datafile:
        data = json.load(datafile)
    
    print("Total samples:", len(data))

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]

    matrix_time = list()
    graph_time = list()
    route_time = list()

    for sample in data:
        if sample['image'] in images:
            if sample['traversability_function'] == f.__name__:
                if sample['region_size'] == r:
                    if sample['cut_threshold'] == c:
                        matrix_time.append(sample['matrix_build_time'])
                        graph_time.append(sample['graph_build_time'])
                        route_time.append(sample['path_build_time'])
    
    print("Matrix build time:", numpy.mean(matrix_time))
    print("Graph build time:", numpy.mean(graph_time))
    print("Route build time:", numpy.mean(route_time))

    print("Evaluated samples:", len(matrix_time))

main_experiment()
heatmaps_plot('output/data.json')
execution_time_plot('output/data.json')

main_experiment_overlap()
heatmaps_plot('output/data-overlap.json')
execution_time_plot('output/data-overlap.json')

execution_time_plot_combined('output/data.json', 'output/data-overlap.json')

average_time_for_param_combination('output/data-overlap.json', r=10, c=0.4)