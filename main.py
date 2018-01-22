""" Test file
"""
import os
import time
import random
import numpy
import progressbar
import itertools
import matplotlib.pyplot as pyplot
import gtde
import graphmap
import tqdm
import json

def one():
    """ Example 1: Computes a TDI and shows on screen
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=20)

    frame = gtde.loadimage('image/example.jpg')
    diffimage = estimator.computetdi(frame)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.saveimage('output/example.jpg', [frame, diffimage])

def two():
    """ Example 2: Computes a TDM and writes to stdout
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=12)

    frame = gtde.loadimage('image/example.jpg')
    diffmatrix = estimator.computematrix(frame)

    import matplotlib
    import matplotlib.mlab as mlab
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = pyplot.subplots(1,1)

    weights = numpy.ones_like(diffmatrix.flatten())/float(len(diffmatrix.flatten()))
    n, _, _ = ax.hist(diffmatrix.flatten(), bins=numpy.arange(0, 1 + 0.1, 0.1), weights=weights, facecolor='green', alpha=0.75)

    ax.set_xlabel("Atravessabilidade")
    ax.set_xticks(numpy.arange(0, 1.01, 0.1))
    ax.set_xlim([0, 1])
    ax.set_ylabel("Porcentagem")
    ax.set_yticks(numpy.arange(0, max(n)+0.1, 0.1))
    ax.set_yticklabels(["%.0f%%" % (100*x) for x in numpy.arange(0, max(n)+0.1, 0.1)])
    ax.grid(True)
    fig.tight_layout()
    pyplot.show()

def three():
    """ Example 3: Computes a TDI and compares to its ground truth
        Computes root mean squared error and saves image file on disk
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=6,
                    function=gtde.grayhistogram)

    frame = gtde.loadimage('image/aerial01.jpg')
    truth = gtde.loadimage('labels/aerial01.jpg')
    framediff = estimator.computetdi(frame)
    truthdiff = estimator.groundtruth(truth)
    print("Correlation:", estimator.error(framediff, truthdiff, 'corr'))
    gtde.saveimage('ground-truth.jpg', [truthdiff, framediff])

def four():
    """ Example 4: Computes TDIs for all files in datasetpath and saves to tdipath
    """
    tdipath = 'output'
    datasetpath = 'image'

    for g in [4]:

        outputpath = os.path.join(tdipath, 'R%03d' % g)

        if not os.path.exists(outputpath):
                os.makedirs(outputpath)

        with open(os.path.join(outputpath, 'time.log'), 'w') as timelog:

            estimator = gtde.GroundTraversalDifficultyEstimator( \
                            granularity=g,
                            function=gtde.grayhistogram)
            
            dataset = list()
            for (dirpath, dirnames, filenames) in os.walk(datasetpath):
                dataset.extend(filenames)
                break
            
            dataset.sort(key=str.lower)

            times = list()

            widgets = [progressbar.Percentage(), ' Progress',
                        progressbar.Bar(), ' ', progressbar.ETA()]

            bar = progressbar.ProgressBar(widgets=widgets, maxval=len(dataset))

            print("Generating TDI with %dx%d regions" % (g, g))
            bar.start()

            for i, imagename in enumerate(dataset):
                img = gtde.loadimage(os.path.join(datasetpath, imagename))
                start = time.time()
                tdi = estimator.computetdi(img)
                times.append(time.time() - start)
                gtde.save2image(os.path.join(outputpath, imagename), img, tdi)
                timelog.write("%s: %.3f s\n" % (imagename, times[-1]))
                timelog.flush()
                bar.update(i+1)
            
            bar.finish()
            
            timelog.write("Average: %.3f s" % (numpy.mean(times)))

def five():
    """ Example 5: Computes a route between two regions pictured in an input image
        Saves image to disk
    """
    tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    function=gtde.grayhistogram)

    image = gtde.loadimage('image/aerial01.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix)

    source = G.vertex(graphmap.coord2((12, 1), tdmatrix.shape[1]))
    target = G.vertex(graphmap.coord2((4, 14), tdmatrix.shape[1]))

    path, found = router.route(G, source, target)
    graphmap.drawgraph(G, path, 'output/tdg.png')

def six():
    """ Example 6: Computes one TDI for each defined function
        Saves a concatenation of input image and its TDIs
    """
    gray_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=8,
                    function=gtde.grayhistogram)
    
    rgb_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=8,
                    function=gtde.rgbhistogram)
    
    superpixels_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=8,
                    function=gtde.superpixels)

    frame = gtde.loadimage('image/aerial01.jpg')

    graydiffimage = gray_estimator.computetdi(frame, contrast=False)
    rgbdiffimage = rgb_estimator.computetdi(frame, contrast=False)
    superpixelsdiffimage = superpixels_estimator.computetdi(frame, contrast=False)

    gtde.saveimage('output/comparison.jpg', [frame, graydiffimage, rgbdiffimage, superpixelsdiffimage])

def seven():
    """ Example 7: Generate graphs for all similarity measures available in gtde
    """
    rootpath = 'output'
    labelpath = 'labels'
    datasetpath = 'image'

    measures = ['corr']
    functions = [gtde.randomftd, gtde.grayhistogram, gtde.rgbhistogram, gtde.superpixels]
    resolutions = [4, 6, 8, 10, 12, 14, 16]

    labeldataset = list()
    for (dirpath, dirnames, filenames) in os.walk(labelpath):
        labeldataset.extend(filenames)
        break
    
    labeldataset.sort(key=str.lower)

    data = dict()

    widgets = [progressbar.Percentage(), ' Progress',
               progressbar.Bar(), ' ', progressbar.ETA()]

    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(labeldataset)*len(measures)*len(functions)*len(resolutions))

    print("Running TDI performance tests")
    bar.start()

    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    with open(os.path.join(rootpath, 'tdi.log'), 'w') as tdilog:

        counter = 0
        bar.update(counter)

        for i, imagename in enumerate(labeldataset):

            lbl = gtde.loadimage(os.path.join(labelpath, imagename))
            img = gtde.loadimage(os.path.join(datasetpath, imagename))

            for measure in measures:

                if not measure in data:
                    data[measure] = dict()

                for ftd in functions:

                    if not ftd.__name__ in data[measure]:
                        data[measure][ftd.__name__] = dict()

                    for g in resolutions:
                        
                        if not str(g) in data[measure][ftd.__name__]:
                            data[measure][ftd.__name__][str(g)] = list()

                        estimator = gtde.GroundTraversalDifficultyEstimator( \
                                        granularity=g,
                                        function=ftd)
                        
                        gt = estimator.groundtruth(lbl, matrix=True)
                        start = time.time()
                        tdm = estimator.computematrix(img)
                        data[measure][ftd.__name__][str(g)].append(estimator.error(tdm, gt, measure))
                        tdilog.write("%s\n" % (imagename))
                        tdilog.write("    %s %s %3d %.3f\n" % (measure, ftd.__name__, g, data[measure][ftd.__name__][str(g)][-1]))
                        tdilog.flush()
                        counter += 1
                        bar.update(counter)
        
        bar.finish()
    
    plot_title = {  
        "corr" : "Pearson's correlation coefficient", 
        "jaccard" : "Generalized Jaccard similarity index", 
        "rmse" : "Root mean square error", 
        "nrmse" : "Normalized root mean square error", 
        "psnr" : "Peak signal to noise ratio", 
        "ssim" : "Structural similarity index",
        "nmae" : "Normalized mean absolute error"
    }
    
    ftd_curve = {   
        "randomftd" : "Random",
        "grayhistogram" : "Gray Histogram",
        "rgbhistogram" : "RGB Histogram",
        "superpixels" : "Superpixels"
    }
    
    for measure in measures:
        fig, (ax0) = pyplot.subplots(ncols=1)
        for ftd in functions:
            x = numpy.array(resolutions)
            y = numpy.array([numpy.mean(data[measure][ftd.__name__][str(element)]) for element in x])
            ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[ftd.__name__])
        pyplot.title(plot_title[measure])
        ax0.legend(loc='upper left')
        ax0.set_xlabel("Region size")
        # ax0.set_xscale('log')
        ax0.tick_params(axis='x', which='minor', bottom='off')
        ax0.set_xticks(resolutions)
        ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
        ax0.set_ylabel(plot_title[measure].split(" ")[-1].title())
        fig.tight_layout()
        fig.savefig(os.path.join(rootpath, "score-tdi-%s.png" % (measure)), dpi=300, bbox_inches='tight')
        pyplot.close(fig)

def eight():
    """ Example 8: Computes a route between two labeled keypoints
        Shows the route over image on screen
    """
    g = 12
    c = 0.5
    tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=g,
                    function=gtde.grayhistogram)

    image = gtde.loadimage('image/example.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    labelpoints = gtde.loadimage('keypoints/example.jpg')
    grid = gtde.gridlist(image, g)
    keypoints = graphmap.label2keypoints(labelpoints, grid)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix, confidence=c)

    [source, target] = [G.vertex(v) for v in random.sample(keypoints, 2)]

    path, found = router.route(G, source, target)
    graphmap.drawgraph(G, path, 'output/path-graph.pdf')

    ipath = [int(v) for v in path]
    pathtdi = gtde.imagepath(image, ipath, grid, found=found)
    gtde.saveimage('output/ti.png', [pathtdi])

def nine():
    """ Example 9: Computes a route between all labeled keypoints
        Shows the routes over image on screen
    """
<<<<<<< HEAD
    inputdata = 'aerial05.jpg'
    resolutions = [6]
    confidence = 0.3
=======
    inputdata = 'aerial01.jpg'
    resolutions = [10]
    confidence = 0.2
>>>>>>> ad22e04e70522f8c99f35324a8e8172c0c868ef9

    for g in resolutions:

        penalty = g*(0.2/6)

        if not os.path.exists(os.path.join('output', inputdata.split('.')[0], str(g))):
            os.makedirs(os.path.join('output', inputdata.split('.')[0], str(g)))

        tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                        granularity=g,
                        function=gtde.grayhistogram)

        image = gtde.loadimage(os.path.join('image', inputdata))
        tdmatrix = tdigenerator.computematrix(image)
        tdimage = tdigenerator.computetdi(image)

        if os.path.isfile(os.path.join('labels', inputdata)):
            gt = gtde.loadimage(os.path.join(os.path.join('labels', inputdata)))
            gtmatrix = tdigenerator.groundtruth(gt, matrix=True)
            gtimage = tdigenerator.groundtruth(gt)

        labelpoints = gtde.loadimage(os.path.join('keypoints', inputdata))
        grid = gtde.gridlist(image, g)
        keypoints = graphmap.label2keypoints(labelpoints, grid)

        router = graphmap.RouteEstimator()
        G = router.tdm2graph(tdmatrix, confidence)

        results = list()

        for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

            source = G.vertex(s)
            target = G.vertex(t)

            path, found = router.route(G, source, target)

            results.append(1.0)

            rpath = [gtde.coord(int(v), gtmatrix.shape[1]) for v in path]
            for row, column in rpath:
                if gtmatrix[row][column] < 0.20:
                    results[-1] = numpy.maximum(0, results[-1] - penalty)

            ipath = [int(v) for v in path]
            if results[-1] > 0.7:
                pathtdi = gtde.imagepath(tdimage.copy(), ipath, grid, found=found)
                pathlabel = gtde.imagepath(gtimage.copy(), ipath, grid, found=found)
                pathimage = gtde.imagepath(image.copy(), ipath, grid, found=found)
            else:
                pathtdi = gtde.imagepath(tdimage.copy(), ipath, grid, pathcolor=(255, 0, 0), found=found)
                pathlabel = gtde.imagepath(gtimage.copy(), ipath, grid, pathcolor=(255, 0, 0), found=found)
                pathimage = gtde.imagepath(image.copy(), ipath, grid, pathcolor=(255, 0, 0), found=found)
            
            gtde.saveimage(os.path.join("output", inputdata.split('.')[0], str(g), \
                            "%s-%03d-%03d.jpg" % (inputdata.split('.')[0], g, counter + 1)), [pathtdi, pathlabel, pathimage])


def ten(confidence=0.5):
    """ Example 10:
    """
    labelpath = 'labels/'
    datasetpath = 'image/'
    keypointspath = 'keypoints/'
    ikeypointspath = 'keypoints-impossible/'
    outputpath = 'output/'

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
    functions = [gtde.grayhistogram, gtde.rgbhistogram, gtde.superpixels]
    resolutions = [4, 6, 8, 10, 12, 14, 16, 18, 20]

    labeldataset = list()
    for (dirpath, dirnames, filenames) in os.walk(labelpath):
        labeldataset.extend(filenames)
        break

    labeldataset.sort()

    selected = list(set(labeldataset).intersection(images)) if len(images) > 0 else labeldataset

    data = dict()

    for i in tqdm.trange(len(selected), desc="            Input image "):

        inputdata = selected[i]
        image = gtde.loadimage(os.path.join(datasetpath, inputdata))
        gt = gtde.loadimage(os.path.join(labelpath, inputdata))
        labelpoints = gtde.loadimage(os.path.join(keypointspath, inputdata))
        ilabelpoints = gtde.loadimage(os.path.join(ikeypointspath, inputdata))

        for j in tqdm.trange(len(functions), desc="Traversability function "):

            ftd = functions[j]

            if not ftd.__name__ in data:
                data[ftd.__name__] = dict()

            for k in tqdm.trange(len(resolutions), desc="             Resolution "):

                g = resolutions[k]

                if not str(g) in data[ftd.__name__]:
                    data[ftd.__name__][str(g)] = dict()
                    data[ftd.__name__][str(g)]['score'] = list()
                    data[ftd.__name__][str(g)]['positive'] = list()
                    data[ftd.__name__][str(g)]['negative'] = list()

                penalty = g*(0.2/6)

                tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                                granularity=g,
                                function=ftd)

                tdmatrix = tdigenerator.computematrix(image)

                gtmatrix = tdigenerator.groundtruth(gt, matrix=True)

                grid = gtde.gridlist(image, g)

                keypoints = graphmap.label2keypoints(labelpoints, grid)

                router = graphmap.RouteEstimator()
                G = router.tdm2graph(tdmatrix, confidence)

                results = list()

                combinations = list(itertools.combinations(keypoints, 2))

                for counter in tqdm.trange(len(combinations), desc="         Positive paths "):

                    (s, t) = combinations[counter]

                    results.append(1.0)

                    source = G.vertex(s)
                    target = G.vertex(t)

                    path, found = router.route(G, source, target)

                    rpath = [gtde.coord(int(v), gtmatrix.shape[1]) for v in path]
                    for row, column in rpath:
                        if gtmatrix[row][column] < 0.20:
                            results[-1] = numpy.maximum(0, results[-1] - penalty)

                    data[ftd.__name__][str(g)]['score'].append(results[-1])
                    data[ftd.__name__][str(g)]['positive'].append(float(found))

                ikeypoints = graphmap.label2keypoints(ilabelpoints, grid)
                icombinations = list(itertools.combinations(ikeypoints, 2))

                for counter in tqdm.trange(len(icombinations), desc="         Negative paths "):

                    (s, t) = icombinations[counter]

                    source = G.vertex(s)
                    target = G.vertex(t)

                    path, found = router.route(G, source, target)

                    data[ftd.__name__][str(g)]['negative'].append(float(not found))

    ftd_curve = {
        "randomftd" : "Random",
        "grayhistogram" : "Gray Histogram",
        "rgbhistogram" : "RGB Histogram",
        "superpixels" : "Superpixels"
    }

    fig, (ax0) = pyplot.subplots(ncols=1)
    for ftd in functions:
        x = numpy.array(resolutions)
        y = numpy.array([numpy.mean(data[ftd.__name__][str(element)]['score']) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[ftd.__name__])
    pyplot.title("Path quality score")
    pyplot.grid()
    ax0.legend(loc='lower right')
    ax0.set_xlabel("Region size")
    ax0.tick_params(axis='x', which='minor', bottom='off')
    ax0.set_xticks(resolutions)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
    ax0.set_ylabel("Score")
    ax0.set_ylim([-0.02, 1.02])
    ax0.set_yticks(numpy.arange(0, 1.02, 0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(outputpath, "path_quality_%.1f.png" % confidence), dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    fig, (ax0) = pyplot.subplots(ncols=1)
    for ftd in functions:
        x = numpy.array(resolutions)
        y = numpy.array([numpy.mean(data[ftd.__name__][str(element)]['positive']) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[ftd.__name__])
    pyplot.title("Path finding score")
    pyplot.grid()
    ax0.legend(loc='lower right')
    ax0.set_xlabel("Region size")
    ax0.tick_params(axis='x', which='minor', bottom='off')
    ax0.set_xticks(resolutions)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
    ax0.set_ylabel("Score")
    ax0.set_ylim([-0.02, 1.02])
    ax0.set_yticks(numpy.arange(0, 1.02, 0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(outputpath, "path_positives_%.1f.png" % confidence), dpi=300, bbox_inches='tight')
    pyplot.close(fig)

    fig, (ax0) = pyplot.subplots(ncols=1)
    for ftd in functions:
        x = numpy.array(resolutions)
        y = numpy.array([numpy.mean(data[ftd.__name__][str(element)]['negative']) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[ftd.__name__])
    pyplot.title("Impossible path detection score")
    pyplot.grid()
    ax0.legend(loc='lower right')
    ax0.set_xlabel("Region size")
    ax0.tick_params(axis='x', which='minor', bottom='off')
    ax0.set_xticks(resolutions)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
    ax0.set_ylabel("Score")
    ax0.set_ylim([-0.02, 1.02])
    ax0.set_yticks(numpy.arange(0, 1.02, 0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(outputpath, "path_negatives_%.1f.png" % confidence), dpi=300, bbox_inches='tight')
    pyplot.close(fig)

def eleven():
    """ Example 11:
    """
    labelpath = 'labels/'
    datasetpath = 'image/'
    keypointspath = 'keypoints/'
    ikeypointspath = 'keypoints-impossible/'
    outputpath = 'output/'

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    images = ['aerial%02d.jpg' % i for i in [1, 2, 3, 4, 5, 6, 7, 8]]
    functions = [gtde.grayhistogram, gtde.rgbhistogram, gtde.superpixels]
    resolutions = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    labeldataset = list()
    for (dirpath, dirnames, filenames) in os.walk(labelpath):
        labeldataset.extend(filenames)
        break

    selected = list(set(labeldataset).intersection(images)) if len(images) > 0 else labeldataset
    selected.sort()

    data = list()

    for i in tqdm.trange(len(selected), desc="            Input image "):

        inputdata = selected[i]
        image = gtde.loadimage(os.path.join(datasetpath, inputdata))
        gt = gtde.loadimage(os.path.join(labelpath, inputdata))
        labelpoints = gtde.loadimage(os.path.join(keypointspath, inputdata))
        ilabelpoints = gtde.loadimage(os.path.join(ikeypointspath, inputdata))

        for j in tqdm.trange(len(functions), desc="Traversability function "):

            ftd = functions[j]

            for k in tqdm.trange(len(resolutions), desc="             Resolution "):

                g = resolutions[k]

                penalty = g*(0.2/6)

                tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                                granularity=g,
                                function=ftd)

                tdmatrix = tdigenerator.computematrix(image)

                gtmatrix = tdigenerator.groundtruth(gt, matrix=True)

                grid = gtde.gridlist(image, g)

                for c in tqdm.trange(len(confidences), desc="             Confidence "):

                    confidence = confidences[c]

                    keypoints = graphmap.label2keypoints(labelpoints, grid)

                    router = graphmap.RouteEstimator()
                    G = router.tdm2graph(tdmatrix, confidence)

                    combinations = list(itertools.combinations(keypoints, 2))

                    for counter in tqdm.trange(len(combinations), desc="         Positive paths "):

                        (s, t) = combinations[counter]

                        score = 1.0

                        source = G.vertex(s)
                        target = G.vertex(t)

                        path, found = router.route(G, source, target)

                        rpath = [gtde.coord(int(v), gtmatrix.shape[1]) for v in path]
                        for row, column in rpath:
                            if gtmatrix[row][column] < 0.20:
                                score = numpy.maximum(0, score - penalty)

                        results = dict()
                        results['image'] = inputdata
                        results['traversability_function'] = ftd.__name__
                        results['region_size'] = g
                        results['confidence_threshold'] = confidence
                        results['path_existence'] = True
                        results['path_found'] = found
                        if found:
                            results['path_score'] = score
                            results['path_regions'] = rpath

                        data.append(results)

                    ikeypoints = graphmap.label2keypoints(ilabelpoints, grid)
                    icombinations = list(itertools.combinations(ikeypoints, 2))

                    for counter in tqdm.trange(len(icombinations), desc="         Negative paths "):

                        (s, t) = icombinations[counter]

                        source = G.vertex(s)
                        target = G.vertex(t)

                        path, found = router.route(G, source, target)

                        results = dict()
                        results['image'] = inputdata
                        results['traversability_function'] = ftd.__name__
                        results['region_size'] = g
                        results['confidence_threshold'] = confidence
                        results['path_existence'] = False
                        results['path_found'] = found

                        data.append(results)

        with open(os.path.join(outputpath, 'data.json'), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def twelve():
    """ Example 12
    """
    import seaborn as sns
    import pandas as pd
    import numpy as np

    with open('output/data3.json') as datafile:
        data = json.load(datafile)
<<<<<<< HEAD
    
    images = ['aerial%02d.jpg' % i for i in [5]]
=======

    images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]
>>>>>>> ad22e04e70522f8c99f35324a8e8172c0c868ef9
    functions = [gtde.grayhistogram]
    resolutions = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    heatmatrix = numpy.zeros((len(resolutions), len(confidences)))
    counter = numpy.ones((len(resolutions), len(confidences)))

    for sample in data:
        if sample['image'] in images and sample['traversability_function'] in [ft.__name__ for ft in functions] \
            and sample['path_existence'] == True and sample['path_found'] == True:
            r = resolutions.index(sample['region_size'])
            c = confidences.index(sample['confidence_threshold'])
            heatmatrix[r][c] += sample['path_score']
            counter[r][c] += 1

    heatmatrix /= counter

    f1 = pyplot.figure(1)
    df1 = pd.DataFrame(heatmatrix, index=resolutions, columns=confidences)
    hm1 = sns.heatmap(df1, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")

    #########################################################################

    heatmatrix = numpy.zeros((len(resolutions), len(confidences)))
    counter = numpy.ones((len(resolutions), len(confidences)))

    for sample in data:
        if sample['image'] in images and sample['traversability_function'] in [ft.__name__ for ft in functions]:
            r = resolutions.index(sample['region_size'])
            c = confidences.index(sample['confidence_threshold'])
            if sample['path_existence'] == True:
                counter[r][c] += 1
                if sample['path_found'] == True:
                    heatmatrix[r][c] += 1

    heatmatrix /= counter

    f2 = pyplot.figure(2)
    df2 = pd.DataFrame(heatmatrix, index=resolutions, columns=confidences)
    hm2 = sns.heatmap(df2, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")

    #########################################################################

    heatmatrix = numpy.zeros((len(resolutions), len(confidences)))
    counter = numpy.ones((len(resolutions), len(confidences)))

    for sample in data:
        if sample['image'] in images and sample['traversability_function'] in [ft.__name__ for ft in functions]:
            r = resolutions.index(sample['region_size'])
            c = confidences.index(sample['confidence_threshold'])
            if sample['path_existence'] == False:
                counter[r][c] += 1
                if sample['path_found'] == False:
                    heatmatrix[r][c] += 1

    heatmatrix /= counter

    f3 = pyplot.figure(3)
    df3 = pd.DataFrame(heatmatrix, index=resolutions, columns=confidences)
    hm3 = sns.heatmap(df3, vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")

    #########################################################################

    heatmatrix = numpy.zeros((len(resolutions), len(confidences)))
    counter = numpy.ones((len(resolutions), len(confidences)))

    for sample in data:
        if sample['image'] in images and sample['traversability_function'] in [ft.__name__ for ft in functions]:
            r = resolutions.index(sample['region_size'])
            c = confidences.index(sample['confidence_threshold'])
            counter[r][c] += 1
            if sample['path_existence'] == sample['path_found']:
                heatmatrix[r][c] += 1

    heatmatrix /= counter

    f4 = pyplot.figure(4)
    df4 = pd.DataFrame(heatmatrix, index=resolutions, columns=confidences)
    hm4 = sns.heatmap(df4, vmin=0.5, vmax=1, cmap='RdYlGn', annot=True, fmt=".2f")

    pyplot.show()

# import cProfile
# cProfile.run("eight()", sort="cumulative")

eleven()
