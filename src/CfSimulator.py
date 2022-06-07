# Import packages
import numpy as np
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
# Lod custom functions
from src.local_outlier_factor import find_1_NN_with_l1_norm_and_scaling_factor
from src.local_outlier_factor import local_reachability_density
from src.local_outlier_factor import l1_norm_with_scaling_factor
from src.local_outlier_factor import restricted_training_set_with_target_class
# Load ocean functions
from ocean.RandomForestCounterFactual import RandomForestCounterFactualMilp
from ocean.dataProcessing import DatasetReader


class CfSimulator():
    """
    Create a Simulator class to store training/test data and implement methods to
        - sample data points,
        - find robust counterfactuals, and
        - verify counterfactual validity.
    """

    def __init__(self, datasetName, nbCounterfactuals,
                 dataSeed, testSplitFactor=0.5, verbose=True):
        self.nbCounterfactuals = nbCounterfactuals
        self.verbose = verbose
        self.dataSeed = dataSeed
        self.objectiveNorm = 1
        self.testSplitFactor = testSplitFactor
        self.__read_data_from_file_and_divide_in_train_init_test(
            datasetName)
        self.nbFeatures = len(self.reader.featuresType)

    # - Methods for initialisation -
    def __read_data_from_file_and_divide_in_train_init_test(self, datasetName):
        """ Read data from file and split in train/test/init. """
        dirname = os.path.dirname(os.path.dirname(__file__))
        # Load data from file into dataframe
        filename = dirname+'/datasets/'+datasetName+'.csv'
        self.reader = DatasetReader(filename)
        # First, sample a fixed number of counterfactuals
        tempX, self.x_init, tempY, _ = train_test_split(
            self.reader.X.values, self.reader.y.values,
            test_size=self.nbCounterfactuals, random_state=self.dataSeed)
        # Separate remaining data into train and test sets
        if self.testSplitFactor > 0.0:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                tempX, tempY, test_size=self.testSplitFactor, random_state=self.dataSeed)
        else:
            self.x_train = tempX
            self.y_train = tempY

    def __predict_class_of_init_sample(self, rfClassifier):
        """ Predict label of init samples using input rfClassifier. """
        self.y_init = rfClassifier.predict(self.x_init)
        self.y_target = 1-self.y_init  # target class = flip initial class

    def __print_iter(self, i):
        """ Simple print feedback for each counterfactual in loop. """
        if self.verbose:
            print(" - Generating counterfactual ", i+1,
                  " out of ", self.nbCounterfactuals, ". - ")
        else:
            if i == (self.nbCounterfactuals-1):
                print(".")
            else:
                print(".", end='')
                sys.stdout.flush()

    def __use_isolation_forest(self, useIsolationForest, targetClass, contamination):
        if useIsolationForest:
            # Select all training samples with same class as target class targetClass
            xSamplesWithTargetClass = self.x_train[self.y_train == targetClass]
            # Define and train random forest
            ilf = IsolationForest(n_estimators=50, contamination=contamination)
            ilf.fit(xSamplesWithTargetClass)
        else:
            ilf = None
        return ilf

    def __use_local_outlier_factor(self, useLocalOutlierFactor, i, targetClass, rfClassifier):
        # Pre-compute the 1-LOF of the data set
        if useLocalOutlierFactor:
            hasTargetClass = rfClassifier.predict(self.x_train) == targetClass
            samplesWithTargetClass = self.x_train[hasTargetClass]
            samplesWithTargetClass = np.unique(samplesWithTargetClass, axis=0)
            x_targetClass = restricted_training_set_with_target_class(
                samplesWithTargetClass, self.x_init[i])
            nbSamples = len(x_targetClass)
            featuresType = self.reader.featuresType
            # Find nearest neighbours with l1-norm and a scale factor for discrete features
            distTo1NN, indexOf1NN = find_1_NN_with_l1_norm_and_scaling_factor(
                x_targetClass, nbSamples, featuresType)
            # Calcualte the 1-LOF of the selected training samples
            localReachDens = local_reachability_density(
                x_targetClass, distTo1NN, indexOf1NN)
            # - Subset selection -
            # Pick a subset of the target class samples
            SUBSET_SIZE = 10
            distToInitialObs = [l1_norm_with_scaling_factor(
                self.x_init[i], x_targetClass[k], featuresType, len(featuresType))
                            for k in range(nbSamples)]
            indexOfRankedDistToInitObs = np.argsort(distToInitialObs)
            isCloseNeighbour = indexOfRankedDistToInitObs <= SUBSET_SIZE
            x_targetClass_out = [x_targetClass[i]
                                 for i in range(nbSamples) if isCloseNeighbour[i]]
            localReachDens_out = [localReachDens[i]
                                  for i in range(nbSamples) if isCloseNeighbour[i]]
            distTo1NN_out = [distTo1NN[i]
                             for i in range(nbSamples) if isCloseNeighbour[i]]
        else:
            x_targetClass_out = False
            localReachDens_out = False
            distTo1NN_out = False
        return x_targetClass_out, localReachDens_out, distTo1NN_out

    # - Public methods -

    def generate_robust_counterfactuals(self, rfClassifier,
                                        gurobiEnv, useIsolationForest=False,
                                        contamination=0.1,
                                        useCalibratedAlpha=False,
                                        calibratedAlpha=False,
                                        useLocalOutlierFactor=False,
                                        lofWeight=0.0):
        """ Generate robust counterfactuals for all samples x_init."""
        # Predict label of init samples
        self.__predict_class_of_init_sample(rfClassifier)
        # Find a counterfactual example for each point in x_init
        self.x_cf = np.zeros((self.nbCounterfactuals, self.nbFeatures))
        self.robConstrRelax = np.zeros(self.nbCounterfactuals)
        self.runTime = np.zeros(self.nbCounterfactuals)
        for i in range(self.nbCounterfactuals):
            self.__print_iter(i)
            # Get counterfactual x0 from list x_init
            x0 = self.x_init[i]
            targetClass = int(self.y_target[i])
            # ------------ Configure arguments ------------
            # Train isolation forest if needed
            ilf = self.__use_isolation_forest(
                useIsolationForest, targetClass, contamination)
            # Get Local Outlier Factor parameters
            x_targetClass, localReachDens, distTo1NN = self.__use_local_outlier_factor(
                useLocalOutlierFactor, i, targetClass, rfClassifier)
            # Read feature types and actionability if real data set
            reader = self.reader
            featuresActionnability = reader.featuresActionnability
            featuresType = reader.featuresType
            featuresPossibleValues = reader.featuresPossibleValues
            oneHotEncoding = reader.oneHotEncoding
            # ------------ Solve MILP ------------
            # Instantiate MILP model, build and solve
            oceanRfMilp = RandomForestCounterFactualMilp(
                rfClassifier, x0.reshape(1, -1), targetClass, gurobiEnv,
                verbose=self.verbose, objectiveNorm=self.objectiveNorm,
                useCalibratedAlpha=useCalibratedAlpha,
                calibratedAlpha=calibratedAlpha,
                isolationForest=ilf,
                featuresActionnability=featuresActionnability,
                featuresType=featuresType,
                featuresPossibleValues=featuresPossibleValues,
                oneHotEncoding=oneHotEncoding,
                useLocalOutlierFactor=useLocalOutlierFactor,
                distanceTo1neighbors=distTo1NN,
                localReachDens=localReachDens,
                x_targetClass=x_targetClass,
                lofWeight=lofWeight)
            oceanRfMilp.buildModel()
            oceanRfMilp.solveModel()
            # Store counterfactuals
            self.x_cf[i, :] = oceanRfMilp.x_sol[0]
            self.robConstrRelax[i] = oceanRfMilp.targetRelaxOutput
            self.runTime[i] = oceanRfMilp.runTime
        # Predict class of generated counterfactuals
        y_cf = rfClassifier.predict(self.x_cf)
        if not(np.array_equal(y_cf, self.y_target)):
            print("Warning: could not reach target class for all CF")

    def verify_counterfactual_accuracy(self, testClassifier):
        self.y_true_cf_label = testClassifier.predict(self.x_cf)
        self.cfValidity = 100 * sum(self.y_target == self.y_true_cf_label) / \
            self.nbCounterfactuals
