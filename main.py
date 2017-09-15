""" Test file
"""
import os
import gtde
import numpy

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

    for g in [32, 64, 128, 256, 512]:

        estimator = gtde.GroundTraversalDifficultyEstimator( \
                        granularity=g,
                        function=gtde.superpixels)

        outputpath = os.path.join(tdipath, 'R%03d' % g)

        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        
        dataset = []
        for (dirpath, dirnames, filenames) in os.walk(datasetpath):
            dataset.extend(filenames)
            break
        
        dataset.sort(key=str.lower)

        for i, imagename in enumerate(dataset):
            img = gtde.loadimage(os.path.join(datasetpath, imagename))
            tdi = estimator.computetdi(img)
            gtde.save2image(os.path.join(outputpath, imagename),img, tdi)
            break

four()
