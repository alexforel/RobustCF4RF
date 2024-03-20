# Import packages
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import copy


def dutta_counterfactual(x_cf, x_stable, rfClassifier, cfSimulator, tau):
    """
    Generate a robust counterfactual explanation using the method of
    Dutta et al that measures a local stability of the prediction
    function and moves naive explanations toward training
    examples with high stability.
    """
    NB_NEIGHBORS = 10  # <- no value provided in Dutta et al.

    # Get set of stable points with same target class
    x_stable_target = find_stable_set_with_target_class(
        x_cf, x_stable, rfClassifier)

    # If the stable set is empty, take the training points
    # with largest stability
    # <- no procedure given in Dutta et al. if the stable set
    #    is empty or not large enough
    if len(x_stable_target) < NB_NEIGHBORS:
        x_stable_target = get_most_stable_set_with_target_class(
            x_cf, cfSimulator.x_train, rfClassifier, NB_NEIGHBORS)

    # Find the nearest neighbor of x_cf
    idxs = nearest_neighbors(x_cf.reshape(1, -1),
                             x_stable_target, NB_NEIGHBORS)
    neighbors = x_stable[idxs[0], :]
    # Iteratively move from init explanation to neighbor
    neighbors_stability = []
    neighbor_explanations = []
    for neighbor in neighbors:
        # Main iteration loop
        x_temp = move_explanation_toward_neighbor_if_unstable(
            copy.copy(x_cf), neighbor, rfClassifier, tau)
        # Store stability of temporary explanation
        stability = stability_test(x_temp, rfClassifier)
        neighbors_stability.append(stability)
        # Store explanation for neighbor
        neighbor_explanations.append(x_temp)
    # Measure distance over neighbor explanations
    explanation_distance = [np.linalg.norm(expl, ord=1)
                            for expl in neighbor_explanations]
    # Return neighbor explanation with minimum distance
    closest = neighbor_explanations[np.argmin(explanation_distance)]
    return closest


def move_explanation_toward_neighbor_if_unstable(x_cf_init, neighbor,
                                                 rfClassifier, tau):
    """
    Performs a alpha-step towards a given neighbor and check
    if the obtained explanation is stable.
    """
    ALPHA = 0.05  # step size
    MAX_STEPS = int(1/ALPHA)  # number of steps
    stability = stability_test(x_cf_init, rfClassifier)
    if stability >= tau:
        return x_cf_init
    for i in range(0, MAX_STEPS):
        if stability >= tau:
            break
        else:
            # Step size constant over iterations rather
            # then relative to distance, and as such decreasing over
            # iterations
            factor = i*ALPHA
            x_cf = (1-factor) * x_cf_init + factor * neighbor
            # For relative step as in Dutta et al, use:
            # x_cf = x_cf + ALPHA * neighbor
            stability = stability_test(x_cf, rfClassifier)
    return x_cf


def nearest_neighbors(values, all_values, nbr_neighbors):
    nn = NearestNeighbors(n_neighbors=nbr_neighbors,
                          metric='l1',
                          algorithm='brute').fit(all_values)
    _, idxs = nn.kneighbors(values)
    return idxs


def stability_test(x, rfClassifier, K=1000, sigma=0.1):
    """
    Measure whether the input example x is locally stable,
    i.e. it is in a region where the prediction of the model
    has low variability.
    """
    nb_features = len(x)
    init_class = rfClassifier.predict([x])[0]

    # Draw K samples from Normal(0, sigma^2)
    perturbations = np.random.normal(0, sigma, (K, nb_features))
    # Add the pertubations to the input example x
    pertSamples = perturbations + x

    # Evaluate the model on all the perturbed samples:
    #   Note here that we measure the stability of the predicted
    #   class rather than the stability of the score. The latter
    #   shows worse results in our experiments.
    #   Replace predict by predict_proba to try other approach.
    if init_class == 1:
        pertPredictions = rfClassifier.predict(pertSamples)
    else:
        pertPredictions = 1-rfClassifier.predict(pertSamples)
    # Average the predictions
    avgPertPred = np.mean(pertPredictions)

    # Measure the stability and return it
    diff = pertPredictions - avgPertPred
    sqrtOfSquaredDiff = np.sqrt(np.mean(diff ** 2))
    stability = avgPertPred - sqrtOfSquaredDiff

    return stability


def measure_stability_training_set(x_train, rfClassifier):
    print('- Measure the stability of training points.')
    stability_train = []
    for x in tqdm(x_train):
        stability = stability_test(x, rfClassifier)
        stability_train.append(stability)
    return stability_train


def find_stable_set_with_target_class(x_cf, x_stable, rfClassifier):
    """
    Find elements of x_stable with same predicted class as x_cf.
    """
    targetClass = rfClassifier.predict(x_cf.reshape(1, -1))
    stableClasses = rfClassifier.predict(x_stable)
    # Compare and filter
    mask = (stableClasses == targetClass)
    return x_stable[mask, :]


def get_most_stable_set_with_target_class(x_cf, x_train, rfClassifier,
                                          nbNeighbors):
    targetClass = rfClassifier.predict(x_cf.reshape(1, -1))
    trainClasses = rfClassifier.predict(x_train)
    # Compare and filter
    mask = (trainClasses == targetClass)
    x_target = x_train[mask, :]

    # Measure stability over training examples with same class
    stability_train = []
    for x in x_target:
        stability = stability_test(x, rfClassifier)
        stability_train.append(stability)

    # Find most stable elements
    idxs = np.argpartition(stability_train, -nbNeighbors)[-nbNeighbors:]
    return x_target[idxs, :]
