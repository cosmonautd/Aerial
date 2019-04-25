""" Traversability matrix computation package
"""

import os
import random
import multiprocessing

import cv2
import numpy
import scipy
# import keras
# import mahotas
import matplotlib

from matplotlib import pyplot
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import exposure
from skimage.morphology import dilation, square
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim

def load_image(path):
    """ Loads image from path, converts to RGB
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def grid_list(image, r):
    """ Returns a list of square coordinates representing a grid over image
        Every square has length and height equals to r
    """
    height, width, _ = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert r > 0, "Parameter r must be larger than zero"
    if (height/r).is_integer() and (width/r).is_integer():
        glist = []
        for toplefty in range(0, height, r):
            for topleftx in range(0, width, r):
                glist.append((topleftx, toplefty, r))
        return glist
    else:
        new_height = int(r*numpy.floor(height/r))
        new_width = int(r*numpy.floor(width/r))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height)/2)
            x_edge = int((width - new_width)/2)
            glist = []
            for toplefty in range(y_edge, y_edge+new_height, r):
                for topleftx in range(x_edge, x_edge+new_width, r):
                    glist.append((topleftx, toplefty, r))
            return glist
        else:
            raise ValueError("r probably larger than image dimensions")

def grid_list_overlap(image, r, ov=3):
    """ Returns a list of square coordinates representing a grid over image with overlapping
        Every square has length and height equals to r
    """
    height, width, _ = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert r > 0, "Parameter r must be larger than zero"
    if (height/r).is_integer() and (width/r).is_integer():
        glist = []
        for toplefty in range(0, ov*height, r):
            for topleftx in range(0, ov*width, r):
                glist.append((int(topleftx/ov), int(toplefty/ov), r))
        return glist
    else:
        new_height = int(r*numpy.floor(height/r))
        new_width = int(r*numpy.floor(width/r))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height)/2)
            x_edge = int((width - new_width)/2)
            glist = []
            for toplefty in range(y_edge, (y_edge+ov*new_height), r):
                for topleftx in range(x_edge, (x_edge+ov*new_width), r):
                    glist.append((int(topleftx/ov), int(toplefty/ov), r))
            return glist
        else:
            raise ValueError("r probably larger than image dimensions")

def draw_grid(image, grid, marks=None):
    """ Draws grid over image, optionally marking some squares
    """
    _, width, _ = image.shape
    for i, square in enumerate(grid):
        tlx, tly, r = square[0], square[1], square[2]
        topleft = (tlx, tly)
        bottomright = (tlx+r, tly+r)
        if marks:
            if (i%(width/r), int(i/(width/r))) in marks:
                cv2.rectangle(image, topleft, bottomright, (255, 0, 0), -1)
        cv2.rectangle(image, topleft, bottomright, (0, 0, 0), 1)
        if i == 1:
            cv2.rectangle(image, topleft, bottomright, (0, 0, 0), 2)
    return image

def R_matrix(image, grid):
    """ Returns a matrix of regions from image, according to grid
    """
    k = 0
    height, width, _ = image.shape
    r = grid[k][2]
    rows = int(height/r)
    cols = int(width/r)
    rmatrix = [None]*rows
    for i in range(rows):
        rmatrix[i] = [None]*cols
        for j in range(cols):
            square = grid[k]
            tlx, tly, r = square[0], square[1], square[2]
            R = image[tly:tly+r, tlx:tlx+r]
            rmatrix[i][j] = R
            k += 1
    return rmatrix

def R_matrix_overlap(image, grid, ov=3):
    """ Returns a matrix of regions from image, according to grid
    """
    k = 0
    height, width, _ = image.shape
    r = grid[k][2]
    new_height = int(r*numpy.floor(height/r))
    new_width = int(r*numpy.floor(width/r))
    if new_height > 0 and new_width > 0:
        y_edge = int((height - new_height)/2)
        x_edge = int((width - new_width)/2)
        rows = int(len(list(range(y_edge, (y_edge+ov*new_height), r))))
        cols = int(len(list(range(x_edge, (x_edge+ov*new_width), r))))
        rmatrix = [None]*rows
        for i in range(rows):
            rmatrix[i] = [None]*cols
            for j in range(cols):
                square = grid[k]
                tlx, tly, r = square[0], square[1], square[2]
                R = image[tly:tly+r, tlx:tlx+r]
                rmatrix[i][j] = R
                k += 1
        return rmatrix

def traversability(regions, tf, parallel=True, view=False):
    """ Assigns traversal difficulty estimates to every region
    """
    td_rows = len(regions)
    td_columns = len(regions[0])
    trav = numpy.ndarray((td_rows, td_columns), dtype=float)
    if not parallel:
        for i in range(td_rows):
            for j in range(td_columns):
                trav[i][j] = tf(regions[i][j], view=view)
    else:
        array = [regions[i][j] for i in range(td_rows) for j in range(td_columns)]
        p = multiprocessing.Pool()
        tdarray = p.map(tf, array)
        p.close()
        trav = numpy.array(tdarray, dtype=float).reshape((td_rows, td_columns))

    trav /= trav.max()

    return trav

def coord(i, columns):
    """ Converts one-dimensional index to square matrix coordinates with order c
    """
    return (int(i/columns), int(i%columns))

def traversability_image(image, grid, traversability_matrix):
    """ Returns traversability matrix in image format 
        based on grid and traversability matrix
    """
    height, width, _ = image.shape
    t_image = numpy.ones((height, width), dtype=numpy.uint8)
    for k, element in enumerate(grid):
        tlx, tly, size = element[0], element[1], element[2]
        row, column = coord(k, traversability_matrix.shape[1])
        diff = traversability_matrix[row][column]
        R = (255*diff)*numpy.ones((size, size))
        t_image[tly:tly+R.shape[0], tlx:tlx+R.shape[1]] = R
    return t_image

def draw_path(image, path_indexes, grid, color=(0,255,0), found=False):
    image_copy = image.copy()
    if len(image_copy.shape) < 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
    centers = []
    for k in path_indexes:
        tly, tlx, size = grid[k]
        centers.append((int(tlx+(size/2)), int(tly+(size/2))))
    cv2.circle(image_copy, centers[0][::-1], 6, color, -1)
    cv2.circle(image_copy, centers[-1][::-1], 6, color, -1)
    if found:
        for k in range(len(centers)-1):
            r0, c0 = centers[k]
            r1, c1 = centers[k+1]
            cv2.line(image_copy, (c0, r0), (c1, r1), color, 5)
        r0, c0 = int(numpy.mean([center[0] for center in centers[-5:]])), int(numpy.mean([center[1] for center in centers[-5:]]))
        r1, c1 = centers[-1]
        cv2.arrowedLine(image_copy, (c0, r0), (c1, r1), color, 5, 2, 0, 1)
    return image_copy

def tf_random(R, view=False):
    """ Returns a random traversability value
    """
    #return numpy.random.randint(256)
    return - random.uniform(-1, 0)

def tf_grayhist(R, view=False):
    """ Returns a traversability value based on grayscale histogram dispersion
    """
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    std = numpy.std(R.flatten())
    if std == 0: diff = 1
    else: diff = numpy.minimum(1, 1/std)
    if view:
        fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(8, 4))
        ax0.imshow(R, cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.hist(R.flatten(), 32, [0, 256], facecolor='k', alpha=0.75, histtype='stepfilled')
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return diff

def haralick(image):
    h = mahotas.features.haralick(image)
    h_mean = h.mean(axis=0)
    return h_mean.reshape((1, h_mean.shape[0]))

# with open('model.json', 'r') as json_file:
#     model_json = json_file.read()
#     model = keras.models.model_from_json(model_json)
# model.load_weights("model.h5")

def tf_nn(R, view=False):
    """ Returns a traversability value based on a neural network
    """
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    x = haralick(R)
    y = model.predict(x)
    t = numpy.argmax(y)
    return t

def tf_rgbhist(R, view=False):
    """ Returns a traversability value based on RGB histogram dispersion
    """
    red, green, blue = cv2.split(R)
    std = numpy.std(red)**2 + numpy.std(green)**2 + numpy.std(blue)**2
    if std == 0: diff = 1
    else: diff = numpy.minimum(1, 3/std)
    if view:
        red, green, blue = red.flatten(), green.flatten(), blue.flatten()
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(R, interpolation='bicubic')
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

def tf_superpixels(R, view=False):
    """ Returns a traversability value based on tf_superpixels dispersion
    """
    segments = slic(img_as_float(R), n_segments=32, sigma=5)
    region_superpixels = R.copy()
    superpxs = []
    stats_r, stats_g, stats_b = [], [], []

    for i in numpy.unique(segments): superpxs.append([])
    for i, row in enumerate(segments):
        for j, pixel in enumerate(row):
            superpxs[pixel].append(R[i][j])
    for pixels in superpxs:
        stats_r.append(numpy.array([pixel[0] for pixel in pixels]).mean())
        stats_g.append(numpy.array([pixel[1] for pixel in pixels]).mean())
        stats_b.append(numpy.array([pixel[2] for pixel in pixels]).mean())

    std = numpy.std(numpy.array(stats_r))**2 + numpy.std(numpy.array(stats_g))**2 + numpy.std(numpy.array(stats_b))**2

    if std == 0: diff = 1
    else: diff = numpy.minimum(1, 3/std)

    if view:
        for i, row in enumerate(segments):
            for j, pixel in enumerate(row):
                region_superpixels[i][j] = numpy.array([stats_r[pixel], \
                                                        stats_g[pixel], \
                                                        stats_b[pixel]])
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(R, interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.imshow(region_superpixels, interpolation='bicubic')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return diff

def reference(R, view=False):
    """ Returns a difficulty value based on white pixel density (for labels)
    """
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    diff = numpy.mean(R.flatten())
    if view:
        fig, (ax0, ax1) = pyplot.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax0.imshow(R, cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax1.imshow(edges, cmap='gray', interpolation='bicubic')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        pyplot.show()
    return diff

def show_image(images):
    """ Displays images on screen
    """
    n = len(images)
    if n == 1:
        fig, (ax0) = pyplot.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = pyplot.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    pyplot.show()

def save_image(path, images):
    n = len(images)
    if n == 1:
        fig, (ax0) = pyplot.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = pyplot.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    pyplot.close(fig)

def show_grid(image, grid):
    """ Displays an image on screen with grid overlay
    """
    imagegrid = draw_grid(image, grid, [''' (i,i) for i in range(int(640/16)) '''])
    show_image(imagegrid)

class TraversabilityEstimator():
    """
    """
    def __init__(self, tf=tf_grayhist, r=6, binary=False, threshold=127, overlap=False, ov=3):
        """ Traversability estimator constructor
            Sets all initial estimator parameters
        """
        self.tf = tf
        self.r = r
        self.binary = binary
        self.threshold = threshold
        self.overlap = overlap
        self.ov = ov

    def get_traversability_matrix(self, image, normalize=False):
        """ Returns a difficulty matrix for image based on estimator parameters
        """
        image = cv2.bilateralFilter(image, 15, 75, 75)
        if not self.overlap:
            grid = grid_list(image, self.r)
            regions = R_matrix(image, grid)
        else:
            grid = grid_list_overlap(image, self.r, ov=self.ov)
            regions = R_matrix_overlap(image, grid, ov=self.ov)
        traversability_matrix = traversability(regions, self.tf, parallel=True)
        if self.binary:
            _, traversability_matrix = cv2.threshold(traversability_matrix, self.threshold, 255, cv2.THRESH_BINARY)
        if normalize:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            traversability_matrix = numpy.array(clahe.apply((255*traversability_matrix).astype(numpy.uint8)), dtype=float)/255
            # traversability_matrix = scipy.ndimage.filters.convolve(traversability_matrix, numpy.full((3, 3), 1.0/9))
        return traversability_matrix
    
    def get_traversability_matrix_multiscale(self, image, normalize=True):
        """ Returns a difficulty matrix for image based on estimator parameters
        """
        image = cv2.bilateralFilter(image, 15, 75, 75)
        r_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        traversability_matrix = None

        for r in r_set:
            if r >= self.r:
                if not self.overlap:
                    grid = grid_list(image, r)
                    regions = R_matrix(image, grid)
                else:
                    grid = grid_list_overlap(image, r, ov=self.ov)
                    regions = R_matrix_overlap(image, grid, ov=self.ov)
                if traversability_matrix is None:
                    traversability_matrix = traversability(regions, self.tf, parallel=True)
                else:
                    multiscale = traversability(regions, self.tf, parallel=True)
                    multiscale = cv2.resize(multiscale, traversability_matrix.shape)
                    traversability_matrix += (1/r)*multiscale
        traversability_matrix = traversability_matrix/numpy.amax(traversability_matrix)

        if self.binary:
            _, traversability_matrix = cv2.threshold(traversability_matrix, self.threshold, 255, cv2.THRESH_BINARY)
        if normalize:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            traversability_matrix = numpy.array(clahe.apply((255*traversability_matrix).astype(numpy.uint8)), dtype=float)/255
            # traversability_matrix = scipy.ndimage.filters.convolve(traversability_matrix, numpy.full((3, 3), 1.0/9))
        return traversability_matrix

    def get_traversability_image(self, image, normalize=True):
        """ Returns a traversal difficulty image based on estimator parameters
        """
        if not self.overlap: grid = grid_list(image, self.r)
        else: grid = grid_list_overlap(image, self.r)
        traversability_matrix = self.get_traversability_matrix(image, normalize=normalize)
        return traversability_image(image, grid, traversability_matrix)

    def get_ground_truth(self, imagelabel, matrix=False):
        """ Returns the ground truth based on a labeled binary image
        """
        if not self.overlap:
            grid = grid_list(imagelabel, self.r)
            regions = R_matrix(imagelabel, grid)
        else:
            grid = grid_list_overlap(imagelabel, self.r, ov=self.ov)
            regions = R_matrix_overlap(imagelabel, grid, ov=self.ov)
        traversability_matrix = traversability(regions, reference)
        if matrix:
            return traversability_matrix
        else:
            return traversability_image(imagelabel, grid, traversability_matrix)

    def error(self, traversability_image, ground_truth, function='corr'):
        """ Returns an similarity or error measurement between a traversal 
            difficulty image and a provided ground truth
        """
        if function == 'corr':
            """ Pearson's correlation coefficient
            """
            return numpy.corrcoef(traversability_image.flatten(), ground_truth.flatten())[0][1]
        elif function == 'jaccard':
            """ Generalized Jaccard similarity index
            """
            return numpy.sum(numpy.minimum(traversability_image.flatten(), ground_truth.flatten())) \
                    / numpy.sum(numpy.maximum(traversability_image.flatten(), ground_truth.flatten()))
        elif function == 'rmse':
            """ Root mean square error
            """
            return numpy.sqrt(compare_mse(ground_truth, traversability_image))
        elif function == 'nrmse':
            """ Normalized root mean square error
            """
            return compare_nrmse(ground_truth, traversability_image, norm_type='mean')
        elif function == 'psnr':
            """ Peak signal to noise ratio
            """
            return compare_psnr(ground_truth, traversability_image)
        elif function == 'ssim':
            """ Structural similarity index
            """
            return compare_ssim(ground_truth, traversability_image)
        elif function == 'nmae':
            """ Normalized mean absolute error
            """
            return numpy.mean(numpy.abs(traversability_image.flatten() - ground_truth.flatten()))/255
