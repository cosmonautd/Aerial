import cv2
import numpy

# load an image
def loadimage(path):
    return cv2.imread(path)

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
    N = int(numpy.sqrt(len(squaregrid)))
    R = [None]*N
    for i in range(N):
        R[i] = [None]*N
        for j in range(N):
            square = squaregrid[k]
            tlx, tly, sqsize = square[0], square[1], square[2]
            region = image[tly:tly+sqsize, tlx:tlx+sqsize]
            R[i][j] = region
            k+=1
    return R

# assign traversal difficulty to every region
def gtd(regions):
    td_order = len(regions)
    td = numpy.ndarray((td_order,td_order), dtype=int)
    for i in range(td_order):
        for j in range(td_order):
            region = regions[i][j]
            difficulty = numpy.random.randint(256)
            td[i][j] = difficulty
    return td

def get_coord(i, c):
    return (int(i/c), int(i%c))

def ddi(image, squaregrid, td):
    diffimage = image.copy()
    for k, square in enumerate(squaregrid):
        tlx, tly, sqsize = square[0], square[1], square[2]
        for i in range(sqsize):
            for j in range(sqsize):
                r, c = get_coord(k, len(td))
                diffimage[tly+i][tlx+j] = td[r][c]
    return diffimage

# example methods
def showsquaregrid():
    image = loadimage('aerial.jpg')
    squaregrid = sgl(image, 16)
    imagegrid = dsq(image, squaregrid, [(i,i) for i in range(int(640/16))])
    cv2.imshow('Image Grid', imagegrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showdiffmatrix():
    image = loadimage('aerial.jpg')
    squaregrid = sgl(image, 16)
    regions = get_regions(image, squaregrid)
    diffmatrix = gtd(regions)
    print(diffmatrix.shape)
    print(diffmatrix)

def showdiffimage():
    image = loadimage('aerial.jpg')
    squaregrid = sgl(image, 16)
    regions = get_regions(image, squaregrid)
    diffmatrix = gtd(regions)
    diffimage = ddi(image, squaregrid, diffmatrix)
    cv2.imshow('Difficulty Image', diffimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# main program
showdiffimage()