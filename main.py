""" Test file
"""
import os
import time
import numpy
import gtde
import progressbar
import graphmap

def one():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    binary=True,
                    threshold=25)

    frame = gtde.loadimage('img/aerial2.jpg')
    diffimage = estimator.computetdi(frame)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.show2image(gtde.drawgrid(frame, grid), diffimage)

def two():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('img/aerial2.jpg')
    diffmatrix = estimator.computematrix(frame)
    print(diffmatrix)

def three():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128,
                    function=gtde.superpixels)

    frame = gtde.loadimage('img/aerial2.jpg')
    truth = gtde.loadimage('labels/aerial2.jpg')
    framediff = estimator.computetdi(frame)
    truthdiff = estimator.groundtruth(truth)
    print("Mean Squared Error:", estimator.error(framediff, truthdiff))
    gtde.save2image(framediff, truthdiff)

def four():
    """ Test
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
                gtde.save2image(os.path.join(outputpath, imagename),img, tdi)
                timelog.write("%s: %.3f s\n" % (imagename, times[-1]))
                timelog.flush()
                bar.update(i)
            
            bar.finish()
            
            timelog.write("Average: %.3f s" % (numpy.mean(times)))

def five():
    """ Test
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

five()
