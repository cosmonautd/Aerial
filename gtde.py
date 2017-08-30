""" Ground traversal difficulty estimatation package
"""

import cv2
import numpy
import multiprocessing
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
    if (height/sqsize).is_integer() and (width/sqsize).is_integer():
        glist = []
        for toplefty in range(0, height, sqsize):
            for topleftx in range(0, width, sqsize):
                glist.append((topleftx, toplefty, sqsize))
        return glist
    else:
        new_height = int(sqsize*numpy.floor(height/sqsize))
        new_width = int(sqsize*numpy.floor(width/sqsize))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height)/2)
            x_edge = int((width - new_width)/2)
            glist = []
            for toplefty in range(y_edge, y_edge+new_height, sqsize):
                for topleftx in range(x_edge, x_edge+new_width, sqsize):
                    glist.append((topleftx, toplefty, sqsize))
            return glist
        else:
            raise ValueError("Granularity probably larger than image dimensions")


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

def regionmatrix(image, squaregrid):
    """ Returns a list of regions from image, according to squaregrid
    """
    k = 0
    height, width, _ = image.shape
    sqsize = squaregrid[k][2]
    rows = int(height/sqsize)
    cols = int(width/sqsize)
    rmatrix = [None]*rows
    for i in range(rows):
        rmatrix[i] = [None]*cols
        for j in range(cols):
            square = squaregrid[k]
            tlx, tly, sqsize = square[0], square[1], square[2]
            region = image[tly:tly+sqsize, tlx:tlx+sqsize]
            rmatrix[i][j] = region
            k += 1
    return rmatrix

def traversaldiff(regions, function, parallel=True, view=False):
    """ Assigns traversal difficulty estimates to every region
    """
    td_rows = len(regions)
    td_columns = len(regions[0])
    tdiff = numpy.ndarray((td_rows, td_columns), dtype=float)

    if not parallel:
        for i in range(td_rows):
            for j in range(td_columns):
                tdiff[i][j] = function(regions[i][j], view=view)
    else:
        with multiprocessing.Pool() as p:
            for i in range(td_rows):
                tdiff[i] = p.map(function, regions[i])

    tdiff *= 255/tdiff.max()
    return tdiff

def coord(i, columns):
    """ Converts one-dimensional index to square matrix coordinates with order c
    """
    return (int(i/columns), int(i%columns))

def tdi(image, grid, diffmatrix):
    """ Returns traversal difficulty image from image, grid and difficulty matrix
    """
    height, width, _ = image.shape
    diffimage = 255*numpy.ones((height, width))
    for k, element in enumerate(grid):
        tlx, tly, size = element[0], element[1], element[2]
        row, column = coord(k, diffmatrix.shape[1])
        diff = diffmatrix[row][column]
        region = diff*numpy.ones((size, size))
        diffimage[tly:tly+region.shape[0], tlx:tlx+region.shape[1]] = region
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
    for i, row in enumerate(segments):
        for j, pixel in enumerate(row):
            superpxs[pixel].append(region[i][j])
    for pixels in superpxs:
        stats_r.append(numpy.array([pixel[0] for pixel in pixels]).mean())
        stats_g.append(numpy.array([pixel[1] for pixel in pixels]).mean())
        stats_b.append(numpy.array([pixel[2] for pixel in pixels]).mean())
    
    diff = numpy.std(numpy.array(stats_r))**2 \
            + numpy.std(numpy.array(stats_g))**2 \
            + numpy.std(numpy.array(stats_b))**2
    if view:
        for i, row in enumerate(segments):
            for j, pixel in enumerate(row):
                region_superpixels[i][j] = numpy.array([stats_r[pixel], \
                                                        stats_g[pixel], \
                                                        stats_b[pixel]])
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

def density(region, view=False):
    """ Returns a difficulty value based on white pixel density (for labels)
    """
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    diff = numpy.mean(region.flatten())
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
    ax0.imshow(image1, cmap='gray', interpolation='bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    ax1.imshow(image2, cmap='gray', interpolation='bicubic')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    pyplot.show()

def showgrid(image, grid):
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
        regions = regionmatrix(image, squaregrid)
        diffmatrix = traversaldiff(regions, self.function)
        if self.binary:
            _, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        return diffmatrix

    def computeimage(self, image):
        """ Returns a traversal difficulty image based on estimator parameters
        """
        squaregrid = gridlist(image, self.granularity)
        regions = regionmatrix(image, squaregrid)
        diffmatrix = traversaldiff(regions, self.function)
        if self.binary:
            _, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        diffimage = tdi(image, squaregrid, diffmatrix)
        return diffimage
    
    def groundtruth(self, label):
        """ Compares traversal difficulty image with a provided labeled ground truth
        """
        squaregrid = gridlist(label, self.granularity)
        regions = regionmatrix(label, squaregrid)
        diffmatrix = traversaldiff(regions, density)
        diffimage = tdi(label, squaregrid, diffmatrix)
        return diffimage
