import numpy as np
from sklearn.neighbors import NearestNeighbors
from ocean.CounterFactualParameters import FeatureType


def restricted_training_set_with_target_class(x_targetClass, x_init):
    SUBSET_SIZE = np.minimum(101, len(x_targetClass))
    nbrs = NearestNeighbors(n_neighbors=SUBSET_SIZE).fit(x_targetClass)
    distances, indices = nbrs.kneighbors(x_init.reshape(1, -1))
    restrictedTargetClassSet = [x_targetClass[i]
                                for i in range(len(x_targetClass)) if i in indices]
    return restrictedTargetClassSet


def find_1_NN_with_l1_norm_and_scaling_factor(x_targetClass, nbSamples, featuresType):
    """Find nearest neighbours with l1-norm and a scale factor for discrete features"""
    # Calculate 1NN on this subset with our custom weighted l1 norm
    nbFeatures = len(featuresType)
    assert len(x_targetClass[0]) == nbFeatures
    distances = nbFeatures * np.ones((nbSamples, nbSamples))
    for i in range(nbSamples):
        for j in range(nbSamples):
            if not i == j:
                distance = l1_norm_with_scaling_factor(
                    x_targetClass[i], x_targetClass[j], featuresType, nbFeatures)
                if distance == 0:
                    print("Warning: distance is 0")
                    distance = nbFeatures
                distances[i, j] = distance
    indexOf1NN = [np.argmin(distances[i, :]) for i in range(nbSamples)]
    distTo1NN = [distances[i, indexOf1NN[i]] for i in range(nbSamples)]
    return distTo1NN, indexOf1NN


def l1_norm_with_scaling_factor(x1, x2, featuresType, nbFeatures):
    distance = 0.0
    assert len(x1) == nbFeatures
    assert len(x2) == nbFeatures
    for f in range(nbFeatures):
        if featuresType[f] == FeatureType.Numeric:
            scalingFactor = 1
        else:
            scalingFactor = 1/4
        distance += scalingFactor * np.abs(x1[f] - x2[f])
    return distance


def local_reachability_density(x_targetClass, distTo1NN, indexOf1NN):
    """ Calculate lrd_1(x) for all samples of a training set. """
    localReachDens = np.zeros(len(x_targetClass))
    for i in range(len(x_targetClass)):
        rd_1 = reachability_distance(distTo1NN[i], distTo1NN[indexOf1NN[i]])
        localReachDens[i] = 1 / rd_1
    return localReachDens


def reachability_distance(delta_X1X2, d_x2):
    """
    Calculate rd_1(x1, x2) = max(Δ(x1,x2), d_1(x2)) where
    x2 is the 1-NN of x1.
    """
    return np.maximum(delta_X1X2, d_x2)
