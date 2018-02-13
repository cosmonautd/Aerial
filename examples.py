import trav
import graphmap

def compute_traversability_matrix():
    """
    """
    image = trav.load_image('image/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    matrix = mapper.get_traversability_matrix(image)
    print(matrix)

def compute_traversability_image():
    """
    """
    image = trav.load_image('image/aerial01.jpg')
    mapper = trav.TraversabilityEstimator(r=6)
    t_image = mapper.get_traversability_image(image)
    trav.show_image([t_image])

def compare_with_ground_truth():
    """
    """
    mapper = trav.TraversabilityEstimator(r=6)
    image = trav.load_image('image/aerial01.jpg')
    ground_truth = trav.load_image('ground-truth/aerial01.jpg')
    t_image = mapper.get_traversability_image(image)
    t_ground_truth = mapper.get_ground_truth(ground_truth)
    print("Correlation:", mapper.error(t_image, t_ground_truth, 'corr'))
    trav.show_image([t_image, t_ground_truth])

compare_with_ground_truth()