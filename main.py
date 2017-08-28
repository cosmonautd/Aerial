""" Test file
"""
import gtde

def one():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=16,
                    binary=True,
                    threshold=25)

    frame = gtde.loadimage('img/aerial1.jpg')
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

one()
