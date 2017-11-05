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
    """ Example 1: Computes a TDI and shows on screen
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16)

    frame = gtde.loadimage('image/aerial01.jpg')
    diffimage = estimator.computetdi(frame)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.saveimage('aerial01.jpg', [frame, diffimage])

def two():
    """ Example 2: Computes a TDM and writes to stdout
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('image/aerial01.jpg')
    diffmatrix = estimator.computematrix(frame)
    print(diffmatrix)

def three():
    """ Example 3: Computes a TDI and compares to its ground truth
        Computes root mean squared error and saves image file on disk
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    function=gtde.superpixels)

    frame = gtde.loadimage('image/aerial01.jpg')
    truth = gtde.loadimage('labels/aerial01.jpg')
    framediff = estimator.computetdi(frame)
    truthdiff = estimator.groundtruth(truth)
    print("Correlation:", estimator.error(framediff, truthdiff, 'corr'))
    gtde.saveimage('ground-truth.jpg', [truthdiff, framediff])

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

    image = gtde.loadimage('image/aerial01.jpg')
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
                    granularity=16,
                    function=gtde.grayhistogram)
    
    rgb_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    function=gtde.rgbhistogram)
    
    edge_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    function=gtde.cannyedge)
    
    superpixels_estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    function=gtde.superpixels)

    frame = gtde.loadimage('image/aerial01.jpg')

    graydiffimage = gray_estimator.computetdi(frame, contrast=False)
    rgbdiffimage = rgb_estimator.computetdi(frame, contrast=False)
    edgediffimage = edge_estimator.computetdi(frame, contrast=False)
    superpixelsdiffimage = superpixels_estimator.computetdi(frame, contrast=False)

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

    image = gtde.loadimage('image/aerial01.jpg')
    tdmatrix = tdigenerator.computematrix(image)

    labelpoints = gtde.loadimage('keypoints/aerial01.jpg')
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
    g = 8
    tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=g,
                    function=gtde.superpixels)

    image = gtde.loadimage('image/aerial01.jpg')
    tdmatrix = tdigenerator.computematrix(image)
    tdimage = tdigenerator.computetdi(image)

    labelpoints = gtde.loadimage('keypoints/aerial01.jpg')
    grid = gtde.gridlist(image, g)
    keypoints = graphmap.label2keypoints(labelpoints, grid)

    router = graphmap.RouteEstimator()
    G = router.tdm2graph(tdmatrix)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        source = G.vertex(s)
        target = G.vertex(t)

        path = router.route(G, source, target)

        ipath = [int(v) for v in path]
        pathimg = gtde.imagepath(image.copy(), ipath, grid)
        pathtdi = gtde.imagepath(tdimage.copy(), ipath, grid)
        gtde.save2image('output/%03d.jpg' % (counter + 1), pathimg, pathtdi)


def ten():
    """ Example 10:
    """
    labelpath = 'labels/'
    datasetpath = 'image/'
    keypointspath = 'keypoints/'
    outputpath = 'output/'

    # functions = [gtde.randomftd, gtde.grayhistogram, gtde.rgbhistogram, gtde.cannyedge, gtde.superpixels]
    # resolutions = [6, 8, 10, 12, 14, 16]

    functions = [gtde.grayhistogram]
    resolutions = [6, 8, 10, 12, 14, 16]

    labeldataset = list()
    for (dirpath, dirnames, filenames) in os.walk(labelpath):
        labeldataset.extend(filenames)
        break
    
    data = dict()
    
    for i, imagename in enumerate(labeldataset):

        for ftd in functions:

            data[ftd.__name__] = dict()

            for g in resolutions:

                data[ftd.__name__][str(g)] = list()

                penalty = (g*0.4)/8

                inputdata = imagename

                finaloutputpath = os.path.join(outputpath, ftd.__name__, str(g))

                if not os.path.exists(finaloutputpath):
                    os.makedirs(finaloutputpath)

                tdigenerator = gtde.GroundTraversalDifficultyEstimator( \
                                granularity=g,
                                function=ftd)

                image = gtde.loadimage(os.path.join(datasetpath, inputdata))
                tdmatrix = tdigenerator.computematrix(image)
                tdimage = tdigenerator.computetdi(image)

                gt = gtde.loadimage(os.path.join(labelpath, inputdata))
                gtmatrix = tdigenerator.groundtruth(gt, matrix=True)
                gtimage = tdigenerator.groundtruth(gt)

                labelpoints = gtde.loadimage(os.path.join(keypointspath, inputdata))
                grid = gtde.gridlist(image, g)
                keypoints = graphmap.label2keypoints(labelpoints, grid)

                router = graphmap.RouteEstimator()
                G = router.tdm2graph(tdmatrix)

                results = list()

                for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

                    results.append(1.0)

                    source = G.vertex(s)
                    target = G.vertex(t)

                    path = router.route(G, source, target)

                    rpath = [gtde.coord(int(v), gtmatrix.shape[1]) for v in path]
                    for row, column in rpath:
                        if gtmatrix[row][column] > 220:
                            results[-1] = numpy.maximum(0, results[-1] - penalty)

                    ipath = [int(v) for v in path]

                    if results[-1] > 0.65:
                        pathtdi = gtde.imagepath(tdimage.copy(), ipath, grid)
                        pathlabel = gtde.imagepath(gtimage.copy(), ipath, grid)
                        pathimage = gtde.imagepath(image.copy(), ipath, grid)
                    else:
                        pathtdi = gtde.imagepath(tdimage.copy(), ipath, grid, pathcolor=(255, 0, 0))
                        pathlabel = gtde.imagepath(gtimage.copy(), ipath, grid, pathcolor=(255, 0, 0))
                        pathimage = gtde.imagepath(image.copy(), ipath, grid, pathcolor=(255, 0, 0))
                    
                    print("Path %03d computed: %.2f" % (counter+1, results[-1]))
                    data[ftd.__name__][str(g)].append(results[-1])

                    gtde.saveimage( os.path.join(finaloutputpath, "%s-%03d.jpg" % (inputdata.split('.')[0], counter + 1)), [pathtdi, pathlabel, pathimage])
                
                print("Success rate: %.2f" % (numpy.mean(results)))
    
    ftd_curve = {
        "randomftd" : "Random",
        "grayhistogram" : "Gray Histogram",
        "rgbhistogram" : "RGB Histogram",
        "cannyedge" : "Edge Density",
        "superpixels" : "Superpixels"
    }
    
    fig, (ax0) = pyplot.subplots(ncols=1)
    for ftd in functions:
        x = numpy.array(resolutions)
        y = numpy.array([numpy.mean(data[ftd.__name__][str(element)]) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[ftd.__name__])
    pyplot.title("Path planning performance")
    ax0.legend(loc='upper left')
    ax0.set_xlabel("Region size")
    # ax0.set_xscale('log')
    ax0.tick_params(axis='x', which='minor', bottom='off')
    ax0.set_xticks(resolutions)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in resolutions])
    ax0.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(os.path.join(outputpath, "score.png"), dpi=300, bbox_inches='tight')
    pyplot.close(fig)

ten()