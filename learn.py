import os
import cv2
import numpy
import skimage.feature
import trav

def lbph(image, radius, points):
    lbp = skimage.feature.local_binary_pattern(image, points, radius, method='uniform')
    (hist, _) = numpy.histogram(lbp.ravel(), 
                                bins=numpy.arange(0, points + 3),
                                range=(0, points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

r = 10
lbp_radius = int(numpy.log2(r))
lbp_points = 8*lbp_radius

images = ['aerial%02d.jpg' % i for i in [1,2,3,4,5,6,7,8]]

X = list()
Y = list()

for id_ in images:

    image_path = os.path.join('image', id_)
    ground_truth_path = os.path.join('ground-truth', id_)

    image = trav.load_image(image_path)
    ground_truth = trav.load_image(ground_truth_path)
    grid = trav.grid_list(image, r)

    im_regions = trav.R_matrix(image, grid)
    gt_regions = trav.R_matrix(ground_truth, grid)

    td_rows = len(im_regions)
    td_columns = len(im_regions[0])

    for i in range(td_rows):
        for j in range(td_columns):
            R = im_regions[i][j]
            R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
            R_lbph = lbph(R, lbp_radius, lbp_points)
            G = gt_regions[i][j]
            G = cv2.cvtColor(G, cv2.COLOR_BGR2GRAY)
            t = numpy.mean(G)/255
            if t < 0.4: c = 0
            elif 0.4 < t and t < 0.7: c = 1
            else: c = 2
            X.append(R_lbph)
            Y.append(c)

X = numpy.array(X)
Y = numpy.array(Y)