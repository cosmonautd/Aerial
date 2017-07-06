import cv2
import numpy

# load an image
def loadimage(path):
    return cv2.imread(path)

# get square grid list
def sgl(image, sqsize=32):
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

# main program
image = loadimage('aerial.jpg')
imagegrid = dsq(image, sgl(image, 8), [(i,i) for i in range(50)])
cv2.imshow('Image Grid', imagegrid)
cv2.waitKey(0)
cv2.destroyAllWindows()
