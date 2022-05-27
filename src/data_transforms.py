import numpy as np
from bisect import bisect_left
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
# Load from ocean
from ocean.CounterFactualParameters import *


def sample_neighbours(x, reader, nbNeighbours=5, hypersphereRadius=0.1):
    """ Sample neighbours of input sample while respecting feature types."""
    nbFeatures = len(x)
    # Generate uniform points on the unit-hyperspehere
    gaussianSamples = np.random.normal(size=(nbNeighbours, nbFeatures))
    samplesNorm = np.linalg.norm(gaussianSamples, ord=2, axis=1)
    hypersphereSamples = np.array([gaussianSamples[i, :] / samplesNorm[i]
                                   for i in range(nbNeighbours)]) * hypersphereRadius
    # Create neighbours on the hypersphere centered on x
    neighbours = x + hypersphereSamples
    # Correct the values to closest acceptable values:
    # only relevant for binary and categorical/discrete features
    if reader:
        neighbours = correct_data_to_feasible_values(neighbours,
                                                     reader.featuresType,
                                                     reader.featuresPossibleValues)
    # Round and remove duplicates
    neighbours = np.unique(np.around(neighbours, decimals=10), axis=0)
    return neighbours


def augment_data(x_train, augmentedDataSize, reader=False, sample_weight=None):
    """ Return an augmented data using Kernel Density Estimation and sampling."""
    new_data = generate_data_from_kde(x_train, augmentedDataSize,
                                      reader=False, sample_weight=None)
    return np.concatenate((x_train, new_data), axis=0)


def generate_data_from_kde(x, augmentedDataSize,
                           useGridSearchCV=False, reader=False, sample_weight=None):
    """ KDE and sample data """
    if useGridSearchCV:
        # Use grid search cross-validation to optimize the bandwidth
        params = {"bandwidth": np.logspace(-2, 0, 10)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(x, sample_weight=sample_weight)
        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)

    # Sample additional datapoints
    new_data = kde.sample(augmentedDataSize)
    # Correct continuous data to feasible values
    new_data[new_data <= 0] = 0.0
    new_data[new_data >= 1] = 1.0
    # Correct to discrete features to feasible values
    if reader:
        new_data = correct_data_to_feasible_values(
            new_data, reader.featuresType, reader.featuresPossibleValues)
    return new_data


def correct_data_to_feasible_values(data, featuresType, featuresPossibleValues):
    """ If there are discrete or binary features: correct to closest possible value. """
    if not featuresType == False:
        for f in range(len(featuresType)):
            if featuresType[f] == FeatureType.Binary:
                valList = [0, 1]
            else:
                valList = featuresPossibleValues[f]
            if len(valList) > 0:
                for i in range(len(data[:, f])):
                    closestValue = takeClosest(valList, data[i, f])
                    data[i, f] = closestValue
    return data


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    If number is outside of min or max return False
    # From https://stackoverflow.com/a/52051984
    """
    if myNumber > myList[-1]:
        return myList[-1]
    if myNumber < myList[0]:
        return myList[0]
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after  # , pos
    else:
        return before  # , pos-1
