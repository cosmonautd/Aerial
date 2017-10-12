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

def one():
    """ Example 1: Computes a binary TDI and shows on screen
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    binary=True,
                    threshold=30)

    frame = gtde.loadimage('img/aerial2.jpg')
    diffimage = estimator.computetdi(frame, contrast=False)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.show2image(gtde.drawgrid(frame, grid), diffimage)

def two():
    """ Example 2: Computes a TDM and writes to stdout
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('img/aerial2.jpg')
    diffmatrix = estimator.computematrix(frame)
    print(diffmatrix)

def three():
    """ Example 3: Computes a TDI and compares to its ground truth
        Computes root mean squared error and saves image file on disk
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.superpixels)

    frame = gtde.loadimage('img/aerial2.jpg')
    truth = gtde.loadimage('labels/aerial2.jpg')
    framediff = estimator.computetdi(frame)
    truthdiff = estimator.groundtruth(truth)
    print("Root Mean Squared Error:", estimator.error(framediff, truthdiff, 'rmse'))
    gtde.save2image('truth.png', truthdiff, framediff)

def four():
    """ Example 4: Computes TDIs for all files in datasetpath and saves to tdipath
    """
    tdipath = '/home/dave/Datasets/DroneMapper/DroneMapper_AdobeButtes_TDI/'
    datasetpath = '/home/dave/Datasets/DroneMapper/DroneMapper_AdobeButtes/'

    for g in [512, 256, 128, 64, 32]:

        outputpath = os.path.join(tdipath, 'R%03d' % g)

        if not os.path.exists(outputpath):
                os.makedirs(outputpath)

        with open(os.path.join(outputpath, 'time.log'), 'w') as timelog:

            estimator = gtde.GroundTraversalDifficultyEstimator( \
                            granularity=g,
                            function=gtde.superpixels)
            
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
                    granularity=128,
                    function=gtde.superpixels)

    image = gtde.loadimage('img/aerial2.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix)

    source = G.vertex(graphmap.coord2((12, 1), tdmatrix.shape[1]))
    target = G.vertex(graphmap.coord2((4, 14), tdmatrix.shape[1]))

    path = router.route(G, source, target)
    graphmap.drawgraph(G, path, 'tdg.png')

def six():
    """ Example 6: Computes one TDI for each defined function
        Shows on screen a concatenation of input image and its TDIs
    """
    gray_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.grayhistogram)
    
    rgb_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.rgbhistogram)
    
    edge_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.cannyedge)
    
    superpixels_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.superpixels)

    frame = gtde.loadimage('img/aerial2.jpg')

    graydiffimage = gray_estimator.computetdi(frame, contrast=True)
    rgbdiffimage = rgb_estimator.computetdi(frame, contrast=True)
    edgediffimage = edge_estimator.computetdi(frame, contrast=True)
    superpixelsdiffimage = superpixels_estimator.computetdi(frame, contrast=True)

    gtde.show5image(frame, graydiffimage, rgbdiffimage, edgediffimage, superpixelsdiffimage)

def seven():
    """ Example 7: Generate graphs for all similarity measures available in gtde
    """
    rootpath = '/home/dave/Datasets/DroneMapper/MeasuresEvaluation'
    labelpath = '/home/dave/Datasets/DroneMapper/DroneMapper_AdobeButtes_LABELS/'
    datasetpath = '/home/dave/Datasets/DroneMapper/DroneMapper_AdobeButtes/'

    # measures = ['corr', 'jaccard', 'rmse', 'nrmse', 'psnr', 'ssim']
    # functions = [gtde.randomftd, gtde.grayhistogram, gtde.rgbhistogram, gtde.cannyedge, gtde.superpixels]
    # resolutions = [512, 256, 128, 64, 32]

    measures = ['nmae', 'nrmse']
    functions = [gtde.randomftd, gtde.grayhistogram, gtde.rgbhistogram, gtde.cannyedge, gtde.superpixels]
    resolutions = [512, 256, 128, 64, 32]

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

        for i, imagename in enumerate(labeldataset):

            lbl = gtde.loadimage(os.path.join(labelpath, imagename))
            img = gtde.loadimage(os.path.join(datasetpath, imagename))

            for measure in measures:

                data[measure] = dict()

                for ftd in functions:

                    data[measure][ftd.__name__] = dict()

                    for g in resolutions:
                        
                        data[measure][ftd.__name__][str(g)] = list()

                        estimator = gtde.GroundTraversalDifficultyEstimator( \
                                        granularity=g,
                                        function=ftd)
                        
                        gt = estimator.groundtruth(lbl)
                        start = time.time()
                        tdi = estimator.computetdi(img)
                        data[measure][ftd.__name__][str(g)].append(estimator.error(tdi, gt, measure))
                        tdilog.write("%s\n" % (imagename))
                        tdilog.write("    %s %s %3d %.3f\n" % (measure, ftd.__name__, g, data[measure][ftd.__name__][str(g)][-1]))
                        tdilog.flush()
                        bar.update(i+1)
        
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
        "cannyedge" : "Edge Density",
        "superpixels" : "Superpixels"
    }
    
    for measure in measures:
        fig, (ax0) = pyplot.subplots(ncols=1)
        for ftd in functions:
            x = numpy.array(resolutions)
            y = numpy.array([numpy.mean(data[measure][ftd.__name__][str(element)]) for element in x])
            ax0.plot(x, y, '-o', markevery=range(5), label=ftd_curve[ftd.__name__])
        pyplot.title(plot_title[measure])
        ax0.legend(loc='upper left')
        ax0.set_xlabel("Region size")
        ax0.set_xscale('log')
        ax0.tick_params(axis='x', which='minor', bottom='off')
        ax0.set_xticks(resolutions)
        ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
        ax0.set_ylabel(plot_title[measure].split(" ")[-1].title())
        fig.tight_layout()
        pyplot.show(block=False)
    pyplot.show()

def eight():
    """ Example 8: Computes a route between two labeled keypoints
        Shows the route over image on screen
    """
    tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=64,
                    function=gtde.rgbhistogram)

    image = gtde.loadimage('img/aerial2.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    labelpoints = gtde.loadimage('points/aerial2.jpg')
    grid = gtde.gridlist(image, 64)
    keypoints = graphmap.label2keypoints(labelpoints, grid)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix)

    [source, target] = [G.vertex(v) for v in random.sample(keypoints, 2)]

    path = router.route(G, source, target)
    graphmap.drawgraph(G, path, 'tdg.png')

    ipath = [int(v) for v in path]
    pathtdi = gtde.imagepath(image, ipath, grid)
    gtde.showimage(pathtdi)

def nine():
    """
    """
    """ Example 9: Computes a route between all labeled keypoints
        Shows the routes over image on screen
    """
    tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=64,
                    function=gtde.superpixels)

    image = gtde.loadimage('img/aerial2.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    labelpoints = gtde.loadimage('points/aerial2.jpg')
    grid = gtde.gridlist(image, 64)
    keypoints = graphmap.label2keypoints(labelpoints, grid)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix)

    for s, t in itertools.combinations(keypoints, 2):

        source = G.vertex(s)
        target = G.vertex(t)

        path = router.route(G, source, target)

        ipath = [int(v) for v in path]
        pathtdi = gtde.imagepath(image.copy(), ipath, grid)
        gtde.showimage(pathtdi)
    
    pyplot.show()

nine()