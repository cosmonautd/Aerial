""" Test file
"""
import gtde

def one():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('img/aerial2.jpg')
    diffimage = estimator.computeimage(frame)
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
                    granularity=256)

    frame = gtde.loadimage('img/aerial4.jpg')
    truth = gtde.loadimage('labels/aerial4.jpg')
    framediff = estimator.computeimage(frame)
    truthdiff = estimator.groundtruth(truth)
    gtde.show2image(framediff, truthdiff)

three()
