import gurobipy as gp
from gurobipy import GRB
from ocean.CounterFactualParameters import BinaryDecisionVariables
from ocean.CounterFactualParameters import FeatureType
from ocean.CounterFactualParameters import FeatureActionnability
from ocean.CounterFactualParameters import eps


class ClassifierCounterFactualMilp:
    def __init__(self, classifier, x0, targetClass, constraintsType, gurobiEnv,
                 objectiveNorm=2,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y
                 ):
        self.verbose = verbose
        self.clf = classifier
        self.x0 = x0
        self.targetClass = targetClass
        self.objectiveNorm = objectiveNorm
        if self.clf.predict(self.x0)[0] == targetClass:
            print("Warning, predicted class of input sample is already", targetClass)
            print("-> It does not make sense to seek a counterfactual")
        self.constraintsType = constraintsType
        self.nFeatures = self.clf.n_features_in_
        assert len(self.x0[0]) == self.nFeatures

        if featuresPossibleValues:
            self.featuresPossibleValues = featuresPossibleValues
        else:
            self.featuresPossibleValues = [None for i in range(self.nFeatures)]
        if not featuresType:
            self.featuresType = [
                FeatureType.Numeric for f in range(self.nFeatures)]
        else:
            self.featuresType = featuresType
        assert self.nFeatures == len(self.featuresType)
        self.binaryFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Binary]
        for f in self.binaryFeatures:
            self.featuresPossibleValues[f] = [0, 1]
        self.continuousFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Numeric]
        self.discreteFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] == FeatureType.Discrete]
        self.categoricalNonOneHotFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] == FeatureType.CategoricalNonOneHot]
        self.discreteAndCategoricalNonOneHotFeatures = [i for i in range(
            self.nFeatures) if self.featuresType[i] in [FeatureType.CategoricalNonOneHot, FeatureType.Discrete]]
        for f in self.discreteAndCategoricalNonOneHotFeatures:
            assert len(self.featuresPossibleValues[f]) > 1
            self.featuresPossibleValues[f] = sorted(
                self.featuresPossibleValues[f])
            for v in range(len(self.featuresPossibleValues[f]) - 1):
                assert self.featuresPossibleValues[f][v
                                                      + 1] >= self.featuresPossibleValues[f][v] + eps

        if featuresActionnability:
            self.featuresActionnability = featuresActionnability
        else:
            self.featuresActionnability = [
                FeatureActionnability.Free for f in range(self.nFeatures)]

        if oneHotEncoding:
            self.oneHotEncoding = oneHotEncoding
            for oneHotEncodedClass in oneHotEncoding:
                assert len(oneHotEncoding[oneHotEncodedClass]) > 0
        else:
            self.oneHotEncoding = dict()

        self.binaryDecisionVariables = binaryDecisionVariables

        self.model = gp.Model("ClassifierCounterFactualMilp", env=gurobiEnv)

    def initSolution(self):
        """
        Intialize decision variables for each feature of the search space.
        Specify the type and domain of each decision variable according to the
        type of feature.
        """
        self.x_var_sol = dict()
        self.discreteFeaturesLevel_var = dict()
        self.discreteFeaturesLevelLinearOrderConstr = dict()
        self.discreteFeaturesLevelLinearCombinationConstr = dict()
        for f in range(self.nFeatures):
            if self.featuresType[f] == FeatureType.Binary:
                self.x_var_sol[f] = self.model.addVar(
                    vtype=GRB.BINARY, name="x_f"+str(f))
            elif self.featuresType[f] == FeatureType.Numeric:
                self.x_var_sol[f] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x_f"+str(f))
            elif self.featuresType[f] in [FeatureType.Discrete, FeatureType.CategoricalNonOneHot]:
                self.x_var_sol[f] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x_f"+str(f))
                # Discrete and categorialNonHot variables
                possibleValues = self.featuresPossibleValues[f]
                self.discreteFeaturesLevel_var[f] = dict()  # this is \nu_i^j
                self.discreteFeaturesLevelLinearOrderConstr[f] = dict()
                linearCombination = gp.LinExpr(possibleValues[0])
                for l in range(1, len(possibleValues)):
                    self.discreteFeaturesLevel_var[f][l] = self.model.addVar(
                        vtype=GRB.BINARY, name="level_f"+str(f)+"_l"+str(l))
                    if l > 1:
                        self.discreteFeaturesLevelLinearOrderConstr[f][l] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][l] <= self.discreteFeaturesLevel_var[f][l-1], "discreteFeaturesLevelLinearOrderConstr_f" + str(f) + "_l" + str(l))
                    linearCombination += self.discreteFeaturesLevel_var[f][l] * (
                        possibleValues[l] - possibleValues[l-1])
                self.discreteFeaturesLevelLinearCombinationConstr[f] = self.model.addConstr(
                    self.x_var_sol[f] == linearCombination, name="x_f"+str(f) + "_discreteLinearCombination")

    def addActionnabilityConstraints(self):
        self.actionnabilityConstraints = dict()
        for f in range(self.nFeatures):
            if self.featuresActionnability[f] == FeatureActionnability.Increasing:
                if self.featuresType[f] not in [FeatureType.Numeric, FeatureType.Discrete]:
                    print(
                        "Increasing actionnability is available only for numeric and discrete features")
                else:
                    self.actionnabilityConstraints[f] = self.model.addConstr(
                        self.x_var_sol[f] >= self.x0[0][f],
                        "ActionnabilityIncreasing_f" + str(f))
            elif self.featuresActionnability[f] == FeatureActionnability.Fixed:
                self.actionnabilityConstraints[f] = self.model.addConstr(
                    self.x_var_sol[f] == self.x0[0][f],
                    "ActionnabilityFixed_f" + str(f))

    def addOneHotEncodingConstraints(self):
        self.oneHotEncodingConstraints = dict()
        for featureName in self.oneHotEncoding:
            expr = gp.LinExpr(0.0)
            for f in self.oneHotEncoding[featureName]:
                expr += self.x_var_sol[f]
            self.model.addConstr(expr == 1, "oneHotEncodingOf_" + featureName)
