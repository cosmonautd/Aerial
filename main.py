""" Test file
"""
import gtde

def one():
    """ Test
    """
    estimator = gtde.GroundTraversalDifficultyEstimator( \
                    granularity=400)

    frame = gtde.loadimage('aerial4.JPG')
    diffimage = estimator.computeimage(frame)
    grid = gtde.gridlist(frame, estimator.granularity)
    gtde.show2image(gtde.drawgrid(frame, grid), diffimage)

one()
