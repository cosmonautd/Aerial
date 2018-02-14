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

def traversability_image_correlation_plot():
    """
    """
    import os
    import numpy
    import matplotlib.pyplot as plt

    dt_path = 'image'
    gt_path = 'ground-truth'
    output_path = 'output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    measure = 'corr'
    f_set = [trav.tf_grayhist, trav.tf_rgbhist, trav.tf_superpixels]
    r_set = [6, 12]

    dataset = list()
    for (_, _, filenames) in os.walk(gt_path):
        dataset.extend(filenames)
        break
    
    dataset.sort(key=str.lower)

    data = dict()
    data[measure] = dict()

    for image_path in dataset:

        image = trav.load_image(os.path.join(dt_path, image_path))
        ground_truth = trav.load_image(os.path.join(gt_path, image_path))

        for f in f_set:

            if not f.__name__ in data[measure]:
                data[measure][f.__name__] = dict()

            for r in r_set:

                print("Running %s %s %s %s" % (measure, image_path, f.__name__, str(r)))
                
                if not str(r) in data[measure][f.__name__]:
                    data[measure][f.__name__][str(r)] = list()

                mapper = trav.TraversabilityEstimator(tf=f, r=r)
                
                t_image = mapper.get_traversability_matrix(image)
                t_ground_truth = mapper.get_ground_truth(ground_truth, matrix=True)
                data[measure][f.__name__][str(r)].append(mapper.error(t_image, t_ground_truth, measure))
    
    plot_title = {  
        "corr" : "Pearson's correlation coefficient", 
        "jaccard" : "Generalized Jaccard similarity index", 
        "rmse" : "Root mean square error", 
        "nrmse" : "Normalized root mean square error", 
        "psnr" : "Peak signal to noise ratio", 
        "ssim" : "Structural similarity index",
        "nmae" : "Normalized mean absolute error"
    }
    
    ftd_curve = {
        "tf_random" : "Random",
        "tf_grayhist" : "Gray Histogram",
        "tf_rgbhist" : "RGB Histogram",
        "tf_superpixels" : "Superpixels"
    }

    fig, (ax0) = plt.subplots(ncols=1)
    for f in f_set:
        x = numpy.array(r_set)
        y = numpy.array([numpy.mean(data[measure][f.__name__][str(element)]) for element in x])
        ax0.plot(x, y, '-o', markevery=range(len(x)), label=ftd_curve[f.__name__])
    plt.title(plot_title[measure])
    #ax0.legend(loc='upper left')
    #ax0.set_xscale('log')
    ax0.set_xlabel("r")
    ax0.set_ylabel(plot_title[measure].split(" ")[-1].title())
    ax0.set_xticks(r_set)
    ax0.set_xticklabels(["%dx%d" % (r, r) for r in r_set])
    ax0.tick_params(axis='x', which='minor', bottom='off')
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "score-traversability_image-%s.png" % (measure)), dpi=300, bbox_inches='tight')
    plt.close(fig)

traversability_image_correlation_plot()