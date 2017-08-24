""" Ground traversal difficulty estimatation package
"""

import cv2
import numpy
from matplotlib import pyplot
from skimage.segmentation import slic
from skimage.util import img_as_float

def loadimage(path):
    """ Loads image from path
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def gridlist(image, sqsize):
    """ Returns a list of square coordinates representing a grid over image
        Every square has length and height equals to sqsize
    """
    height, width, _ = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert sqsize > 0, "Parameter sqsize must be larger than zero"
    assert (height/sqsize).is_integer(), "Image height not a multiple of sqsize"
    assert (width/sqsize).is_integer(), "Image width not a multiple of sqsize"
    glist = []
    for toplefty in range(0, height, sqsize):
        for topleftx in range(0, width, sqsize):
            glist.append((topleftx, toplefty, sqsize))
    return glist

def drawgrid(image, squaregrid, marks=None):
    """ Draws squaregrid over image, optionally marking some squares
    """
    _, width, _ = image.shape
    for i, square in enumerate(squaregrid):
        tlx, tly, sqsize = square[0], square[1], square[2]
        topleft = (tlx, tly)
        bottomright = (tlx+sqsize, tly+sqsize)
        cv2.rectangle(image, topleft, bottomright, (255, 0, 0), 1)
        if marks:
            if (i%(width/sqsize), numpy.floor(i/(width/sqsize))) in marks:
                cv2.rectangle(image, topleft, bottomright, (255, 0, 0), -1)
    return image

def regionlist(image, squaregrid):
    """ Returns a list of regions from image, according to squaregrid
    """
    k = 0
    height, width, _ = image.shape
    sqsize = squaregrid[k][2]
    rows = int(height/sqsize)
    cols = int(width/sqsize)
    rlist = [None]*rows
    for i in range(rows):
        rlist[i] = [None]*cols
        for j in range(cols):
            square = squaregrid[k]
            tlx, tly, sqsize = square[0], square[1], square[2]
            region = image[tly:tly+sqsize, tlx:tlx+sqsize]
            rlist[i][j] = region
            k += 1
    return rlist

def traversaldiff(regions, function, view=False):
    """ Assigns traversal difficulty estimates to every region
    """
    td_rows = len(regions)
    td_columns = len(regions[0])
    tdiff = numpy.ndarray((td_rows, td_columns), dtype=float)
    for i in range(td_rows):
        for j in range(td_columns):
            region = regions[i][j]
            difficulty = function(region, view=view)
            tdiff[i][j] = difficulty
    tdiff *= 255/tdiff.max()
    return tdiff

def coord(i, columns):
    """ Converts one-dimensional index to square matrix coordinates with order c
    """
    return (int(i/columns), int(i%columns))

def tdi(image, squaregrid, diffmatrix):
    """ Returns traversal difficulty image from image, squaregrid and difficulty matrix
    """
    diffimage = image.copy()
    for k, square in enumerate(squaregrid):
        tlx, tly, sqsize = square[0], square[1], square[2]
        for i in range(sqsize):
            for j in range(sqsize):
                row, column = coord(k, len(diffmatrix[0]))
                diffimage[tly+i][tlx+j] = diffmatrix[row][column]
    return diffimage

def random(region, view=False):
    """ Returns a random difficulty value
    """
    return numpy.random.randint(256)

def grayhistogram(region, view=False):
    """ Returns a difficulty value based on grayscale histogram dispersion
    """
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    diff = numpy.std(region.flatten())
    if view:
        fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(8, 4))
        ax0.imshow(region, cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.hist(region.flatten(), 32, [0, 256], facecolor='k', alpha=0.75, histtype='stepfilled')
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return diff

def colorhistogram(region, view=False):
    """ Returns a difficulty value based on RGB histogram dispersion
    """
    red, green, blue = cv2.split(region)
    diff = numpy.std(red)**2 + numpy.std(green)**2 + numpy.std(blue)**2
    if view:
        red, green, blue = red.flatten(), green.flatten(), blue.flatten()
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(region, interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.hist(red, 32, [0, 256], facecolor='r', alpha=0.75, histtype='stepfilled', label='R')
        ax1.hist(green, 32, [0, 256], facecolor='g', alpha=0.75, histtype='stepfilled', label='G')
        ax1.hist(blue, 32, [0, 256], facecolor='b', alpha=0.75, histtype='stepfilled', label='B')
        ax1.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='upper left')
        fig.tight_layout()
        pyplot.show()
    return diff

def cannyedge(region, view=False):
    """ Returns a difficulty value based on edge density
    """
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(region, 100, 200)
    diff = numpy.mean(edges.flatten())
    if view:
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(region, cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.imshow(edges, cmap='gray', interpolation='bicubic')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return  diff

def superpixels(region, view=False):
    """ Returns a difficulty value based on superpixels dispersion
    """
    segments = slic(img_as_float(region), n_segments=32, sigma=5)
    region_superpixels = region.copy()
    superpxs = []
    stats_r, stats_g, stats_b = [], [], []

    for i in numpy.unique(segments): superpxs.append([])
    for i in range(len(segments)):
        for j in range(len(segments[0])):
            superpxs[segments[i][j]].append(region[i][j])
    for pixels in superpxs:
        stats_r.append(numpy.array([pixel[0] for pixel in pixels]).mean())
        stats_g.append(numpy.array([pixel[1] for pixel in pixels]).mean())
        stats_b.append(numpy.array([pixel[2] for pixel in pixels]).mean())
    
    diff = numpy.std(numpy.array(stats_r))**2 \
            + numpy.std(numpy.array(stats_g))**2 \
            + numpy.std(numpy.array(stats_b))**2
    if view:
        for i in range(len(segments)):
            for j in range(len(segments[0])):
                region_superpixels[i][j] = numpy.array([stats_r[segments[i][j]], \
                                                        stats_g[segments[i][j]], \
                                                        stats_b[segments[i][j]]])
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(region, interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.imshow(region_superpixels, interpolation='bicubic')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return diff

def showimage(image):
    """ Displays an image on screen
    """
    fig, (ax0) = pyplot.subplots(ncols=1)
    ax0.imshow(image, interpolation='bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    pyplot.show()

def show2image(image1, image2):
    """ Displays two images on screen, side by side
    """
    _, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(8, 4))
    ax0.imshow(image1, interpolation='bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    ax1.imshow(image2, cmap='gray', interpolation='bicubic')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    pyplot.show()

def showsquaregrid(image, grid):
    """ Displays an image on screen with grid overlay
    """
    imagegrid = drawgrid(image, grid, [''' (i,i) for i in range(int(640/16)) '''])
    showimage(imagegrid)

class GroundTraversalDifficultyEstimator():
    """ Defines a ground traversal difficulty estimator
    """

    def __init__(self, function=superpixels, granularity=128, binary=False, threshold=127):
        """ Ground traversal difficulty estimator constructor
            Sets all initial estimator parameters
        """
        self.function = function
        self.granularity = granularity
        self.binary = binary
        self.threshold = threshold
    
    def computematrix(self, image):
        """ Returns a difficulty matrix for image based on estimator parameters
        """
        squaregrid = gridlist(image, self.granularity)
        regions = regionlist(image, squaregrid)
        diffmatrix = traversaldiff(regions, self.function)
        if self.binary:
            _, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        return diffmatrix

    def computeimage(self, image):
        """ Returns a traversal difficulty image based on estimator parameters
        """
        squaregrid = gridlist(image, self.granularity)
        regions = regionlist(image, squaregrid)
        diffmatrix = traversaldiff(regions, self.function)
        if self.binary:
            _, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        diffimage = tdi(image, squaregrid, diffmatrix)
        return diffimage
