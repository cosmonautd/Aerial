""" Test file
"""
import gtde

def one():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('aerial3.jpg')
    diffimage = estimator.computeimage(frame)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.show2image(gtde.drawgrid(frame, grid), diffimage)

def two():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=128)

    frame = gtde.loadimage('aerial1.jpg')
    diffmatrix = estimator.computematrix(frame)

two()
