import cv2
import numpy
from matplotlib import pyplot
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import io
import argparse

# load an image
def loadimage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# get square grid list
def sgl(image, sqsize):
    height, width, channels = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert sqsize > 0, "Parameter sqsize must be larger than zero"
    assert (height/sqsize).is_integer(), "Image height not a multiple of sqsize"
    assert (width/sqsize).is_integer(), "Image width not a multiple of sqsize"
    gridlist = []
    for toplefty in range(0, height, sqsize):
        for topleftx in range(0, width, sqsize):
            gridlist.append((topleftx, toplefty, sqsize))
    return gridlist

# draw square grid
def dsq(image, squaregrid, marks=None):
    height, width, channels = image.shape
    for i, square in enumerate(squaregrid):
        tlx, tly, sqsize = square[0], square[1], square[2]
        topleft = (tlx, tly)
        bottomright = (tlx+sqsize, tly+sqsize)
        cv2.rectangle(image, topleft, bottomright, (255,0,0), 1)
        if marks:
            if (i%(width/sqsize), numpy.floor(i/(width/sqsize))) in marks:
                cv2.rectangle(image, topleft, bottomright, (255,0,0), -1)
    return image

def get_regions(image, squaregrid):
    k = 0
    height, width, channels = image.shape
    sqsize = squaregrid[k][2]
    rows = int(height/sqsize)
    cols = int(width/sqsize)
    R = [None]*rows
    for i in range(rows):
        R[i] = [None]*cols
        for j in range(cols):
            square = squaregrid[k]
            tlx, tly, sqsize = square[0], square[1], square[2]
            region = image[tly:tly+sqsize, tlx:tlx+sqsize]
            R[i][j] = region
            k+=1
    return R

# assign traversal difficulty to every region
def gtd(regions, diff):
    td_rows = len(regions)
    td_columns = len(regions[0])
    td = numpy.ndarray((td_rows, td_columns), dtype=float)
    for i in range(td_rows):
        for j in range(td_columns):
            region = regions[i][j]
            difficulty = diff(region)
            td[i][j] = difficulty
    td *= 255/td.max()
    return td

def get_coord(i, c):
    return (int(i/c), int(i%c))

def ddi(image, squaregrid, td):
    diffimage = image.copy()
    for k, square in enumerate(squaregrid):
        tlx, tly, sqsize = square[0], square[1], square[2]
        for i in range(sqsize):
            for j in range(sqsize):
                r, c = get_coord(k, len(td[0]))
                diffimage[tly+i][tlx+j] = td[r][c]
    return diffimage

def random(region):
    return numpy.random.randint(256)

def grayhistogram(region):
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # hist, bins = numpy.histogram(region.flatten(), 32, [0,256])
    # fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(8, 4))
    # ax0.imshow(region, cmap = 'gray', interpolation = 'bicubic')
    # ax0.axes.get_xaxis().set_visible(False)
    # ax0.axes.get_yaxis().set_visible(False)
    # ax1.hist(region.flatten(), 32, [0,256], facecolor='black', alpha=0.75, histtype='stepfilled')
    # ax1.axes.get_yaxis().set_visible(False)
    # fig.tight_layout()
    # pyplot.show()
    return numpy.std(region.flatten())

def colorhistogram(region):
    r, g, b = cv2.split(region)
    # hist_r, bins = numpy.histogram(r.flatten(), 32, [0,256])
    # hist_g, bins = numpy.histogram(g.flatten(), 32, [0,256])
    # hist_b, bins = numpy.histogram(b.flatten(), 32, [0,256])
    # fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # ax0.imshow(region, interpolation = 'bicubic')
    # ax0.axes.get_xaxis().set_visible(False)
    # ax0.axes.get_yaxis().set_visible(False)
    # ax1.hist(r.flatten(), 32, [0,256], facecolor='r', alpha=0.75, histtype='stepfilled', label='R')
    # ax1.hist(g.flatten(), 32, [0,256], facecolor='g', alpha=0.75, histtype='stepfilled', label='G')
    # ax1.hist(b.flatten(), 32, [0,256], facecolor='b', alpha=0.75, histtype='stepfilled', label='B')
    # ax1.axes.get_yaxis().set_visible(False)
    # ax1.legend(loc = 'upper left')
    # fig.tight_layout()
    # pyplot.show()
    return numpy.std(r)**2 + numpy.std(g)**2 + numpy.std(b)**2

def cannyedge(region):
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(region, 100, 200)
    # fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # ax0.imshow(region, cmap='gray', interpolation = 'bicubic')
    # ax0.axes.get_xaxis().set_visible(False)
    # ax0.axes.get_yaxis().set_visible(False)
    # ax1.imshow(edges, cmap='gray', interpolation = 'bicubic')
    # ax1.axes.get_xaxis().set_visible(False)
    # ax1.axes.get_yaxis().set_visible(False)
    # fig.tight_layout()
    # pyplot.show()
    return numpy.mean(edges.flatten())

def superpixels(region):
    segments = slic(img_as_float(region), n_segments = 32, sigma = 5)
    superpxs = []
    stats = []
    region_superpixels = region.copy()
    stats_r = []
    stats_g = []
    stats_b = []
    for i in numpy.unique(segments):
        superpxs.append([])
    for i in range(len(segments)):
        for j in range(len(segments[0])):
            superpxs[segments[i][j]].append(region[i][j])
    for pixels in superpxs:
        stats.append(numpy.array(pixels).mean())
        stats_r.append(numpy.array([pixel[0] for pixel in pixels]).mean())
        stats_g.append(numpy.array([pixel[1] for pixel in pixels]).mean())
        stats_b.append(numpy.array([pixel[2] for pixel in pixels]).mean())
    # for i in range(len(segments)):
    #     for j in range(len(segments[0])):
    #         region_superpixels[i][j] = numpy.array([stats_r[segments[i][j]], stats_g[segments[i][j]], stats_b[segments[i][j]]])
    # fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # ax0.imshow(region, interpolation = 'bicubic')
    # ax0.axes.get_xaxis().set_visible(False)
    # ax0.axes.get_yaxis().set_visible(False)
    # '''ax1.imshow(mark_boundaries(img_as_float(region), segments, color=(0,0,1)), interpolation = 'bicubic')'''
    # ax1.imshow(region_superpixels, interpolation = 'bicubic')
    # ax1.axes.get_xaxis().set_visible(False)
    # ax1.axes.get_yaxis().set_visible(False)
    # fig.tight_layout()
    # pyplot.show()
    #return numpy.std(numpy.array(stats))
    #return numpy.std(numpy.array(stats_r)) * numpy.std(numpy.array(stats_g)) * numpy.std(numpy.array(stats_b))
    return numpy.std(numpy.array(stats_r))**2 + numpy.std(numpy.array(stats_g))**2 + numpy.std(numpy.array(stats_b))**2

# example methods
def showsquaregrid():
    image = loadimage('aerial1.jpg')
    squaregrid = sgl(image, 16)
    imagegrid = dsq(image, squaregrid, [(i,i) for i in range(int(640/16))])

    fig, (ax0) = pyplot.subplots(ncols=1)
    ax0.imshow(imagegrid, interpolation = 'bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    pyplot.show()


def showimage(image):
    fig, (ax0) = pyplot.subplots(ncols=1)
    ax0.imshow(image, interpolation = 'bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    pyplot.show()

def show2image(image1, image2):
    fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(8, 4))
    ax0.imshow(image1, interpolation = 'bicubic')
    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)
    ax1.imshow(image2, cmap = 'gray', interpolation = 'bicubic')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    pyplot.show()

class GroundTraversalDifficultyEstimator():

    def __init__(self, function=superpixels, binary=False, threshold = 127):
        self.function = function
        self.binary = binary
        self.threshold = threshold
    
    def computematrix(self, image):
        squaregrid = sgl(image, 256)
        regions = get_regions(image, squaregrid)
        diffmatrix = gtd(regions, self.function)
        if self.binary:
            ret, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        return diffmatrix
    
    def computeimage(self, image):
        squaregrid = sgl(image, 256)
        regions = get_regions(image, squaregrid)
        diffmatrix = gtd(regions, self.function)
        if self.binary:
            ret, diffmatrix = cv2.threshold(diffmatrix, self.threshold, 255, cv2.THRESH_BINARY)
        diffimage = ddi(image, squaregrid, diffmatrix)
        return diffimage


# main program
image = loadimage('aerial2.jpg')
estimator = GroundTraversalDifficultyEstimator(binary=True, threshold=80)
diffimage = estimator.computeimage(image)
show2image(image, diffimage)
