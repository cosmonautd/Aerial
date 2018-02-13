import trav
import graphmap

def traversability_matrix_histogram_plot():
    """
    """
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_style("darkgrid")
    matplotlib.rcParams.update({'font.size': 14})

    image = trav.load_image('image/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    matrix = mapper.get_traversability_matrix(image, normalize=True)

    fig, ax = plt.subplots(1,1)

    weights = numpy.ones_like(matrix.flatten())/float(len(matrix.flatten()))
    n, _, _ = ax.hist(matrix.flatten(), bins=numpy.arange(0, 1 + 0.1, 0.1), weights=weights, facecolor='green', alpha=0.5)

    ax.set_xticks(numpy.arange(0, 1.01, 0.1))
    ax.set_xlim([0, 1])
    ax.set_yticks(numpy.arange(0.05, max(n)+0.05, 0.05))
    ax.set_yticklabels(["%.0f%%" % (100*x) for x in numpy.arange(0.05, max(n)+0.05, 0.05)])
    ax.grid(True)
    fig.tight_layout()
    plt.show()

traversability_matrix_histogram_plot()