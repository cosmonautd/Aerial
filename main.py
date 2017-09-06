""" Test file
"""
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
    gtde.show2image(framediff, truthdiff)

three()
