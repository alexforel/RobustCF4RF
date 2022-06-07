# Import packages
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.ensemble._iforest import _average_path_length
# Import local functions
from src.local_outlier_factor import l1_norm_with_scaling_factor
# Import OCEAN functions
from ocean.TreeMilpManager import TreeInMilpManager
from ocean.ClassifierCounterFactual import ClassifierCounterFactualMilp
from ocean.RandomAndIsolationForest import RandomAndIsolationForest
from ocean.CounterFactualParameters import TreeConstraintsType, eps
from ocean.CounterFactualParameters import BinaryDecisionVariables, FeatureType


class RandomForestCounterFactualMilp(ClassifierCounterFactualMilp):
    def __init__(self, classifier, x0, targetClass, gurobiEnv,
                 isolationForest=None, objectiveNorm=2, verbose=False,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda,
                 useCalibratedAlpha=False,
                 calibratedAlpha=False,
                 useLocalOutlierFactor=False,
                 distanceTo1neighbors=False,
                 localReachDens=False,
                 x_targetClass=False,
                 lofWeight=0.0):
        ClassifierCounterFactualMilp.__init__(
            self, classifier, x0, targetClass, constraintsType, gurobiEnv,
            objectiveNorm=objectiveNorm,
            verbose=verbose,
            binaryDecisionVariables=binaryDecisionVariables,
            featuresType=featuresType,
            featuresPossibleValues=featuresPossibleValues,
            featuresActionnability=featuresActionnability,
            oneHotEncoding=oneHotEncoding)
        self.model.modelName = "RandomForestCounterFactualMilp"
        # Calibrated alpha
        self.useCalibratedAlpha = useCalibratedAlpha
        self.calibratedAlpha = calibratedAlpha
        # Combine elevant forests into completeForest
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(
            self.clf, isolationForest=self.isolationForest)
        # Avoid numerical errors
        self.epsSplit = eps
        # Scale cost between numerical and discrete features
        self.discreteObjScale = 1/4
        # Local outlier factor
        self.useLocalOutlierFactor = useLocalOutlierFactor
        if useLocalOutlierFactor:
            self.distanceTo1neighbors = distanceTo1neighbors
            self.localReachDens = localReachDens
            self.x_targetClass = x_targetClass
            self.lofWeight = lofWeight

    def buildModel(self):
        self.initSolution()  # inherited from ClassifierCounterFactualMilp
        self.buildTrees()
        self.addInterTreeConstraints()
        if self.isolationForest:
            self.addIsolationForestPlausibilityConstraint()
        self.addActionnabilityConstraints()  # inherited function CCFMP
        self.addOneHotEncodingConstraints()  # inherited function CCFMP
        self.initObjective()

    def buildTrees(self):
        self.treeManagers = dict()
        nbTreesToBuild = self.completeForest.n_estimators
        if self.verbose:
            print("Building model for complete forest of {} estimators.".format(
                nbTreesToBuild))
        for t in range(nbTreesToBuild):
            self.treeManagers[t] = TreeInMilpManager(
                self.completeForest.estimators_[t].tree_,
                self.model, self.x_var_sol,
                self.featuresType, constraintsType=self.constraintsType,
                binaryDecisionVariables=self.binaryDecisionVariables)
            self.treeManagers[t].addTreeVariablesAndConstraintsToMilp()

    def addTreePredictionConstraints(self, treeIndices, forestType='Classifier'):
        """
        Determine the prediction score for the target class for each tree:
        soft voting mechanism for each tree.
        """
        for tree in treeIndices:
            treeMng = self.treeManagers[tree]
            treePredictLinExpr = self.treePredictionLinExprFromTreeMng(
                treeMng, forestType)
            # Define decision variable and fix its value to linear expression
            self.treePrediction[tree] = self.model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name="treePred_"+str(tree))
            self.model.addConstr(
                self.treePrediction[tree] == treePredictLinExpr, "treePredCstr_"+str(tree))

    def treePredictionLinExprFromTreeMng(self, treeMng, forestType):
        """ Get tree prediction as a linear expression. """
        USE_TREE_HARD_VOTING = False
        treePredictLinExpr = 0.0
        for node in range(treeMng.n_nodes):
            if treeMng.is_leaves[node]:  # Leaf node
                # Identify the target class proportion
                if forestType == 'Classifier':
                    if USE_TREE_HARD_VOTING:
                        # Read the majority class in the leaf node
                        majorityClass = np.argmax(treeMng.tree.value[node][0])
                        if majorityClass == self.targetClass:
                            treePredictLinExpr += treeMng.y_var[node]
                    else:
                        # Soft-voting: tree prediction is the average prediction in leaf
                        # Read number of samples of each class in the leaf node
                        leafSamples = treeMng.tree.value[node][0]
                        totSamples = sum(leafSamples)
                        # Identify the target class proportion
                        leafTargetClassProportion = leafSamples[self.targetClass]/totSamples
                        treePredictLinExpr += treeMng.y_var[node] * \
                            leafTargetClassProportion
                elif forestType == 'Regressor':
                    leafMeanValue = treeMng.tree.value[node][0][0]
                    treePredictLinExpr += treeMng.y_var[node] * leafMeanValue
        return treePredictLinExpr

    def forestTargetClassScore(self, rfTreesIndices):
        """ Average of tree predictions in a forest. """
        return sum(self.treePrediction[tree] for tree in rfTreesIndices) / len(rfTreesIndices)

    def addTargetClassConstraint(self):
        """
        Determine the RF score of the target class and add a constraint
        to ensure robust class prediction.
        """
        # Define target score as average of tree predictions
        rfTreesIndices = self.completeForest.randomForestEstimatorsIndices
        self.targetClassScore = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
        self.model.addConstr(self.targetClassScore
                             == self.forestTargetClassScore(rfTreesIndices))
        # Robust constraint on target score
        self.targetRelax = self.model.addVar(
            lb=0.0, vtype=GRB.CONTINUOUS, name="targetRelax")
        if self.useCalibratedAlpha:
            self.model.addConstr(
                    self.targetClassScore
                    >= self.calibratedAlpha[self.targetClass] - self.targetRelax,
                    "robustTargetScoreConstraint")
        else:
            self.model.addConstr(self.targetClassScore
                                 >= 1/2 + self.epsSplit - self.targetRelax, "targetClassConstraint")

    def addIsolationForestPlausibilityConstraint(self):
        """
        Add a plausibility constraint based on isolation forests.
        The OCEAN implementation is adapted to use varying contamination
        parameter. The contamination and anomaly score are implemented
        as in scikit-learn.
        """
        ilfTreeIndices = self.completeForest.isolationForestEstimatorsIndices
        self.ilfTreeDepth = dict()
        for t in ilfTreeIndices:
            self.ilfTreeDepth[t] = gp.LinExpr(0.0)
            tm = self.treeManagers[t]
            tree = self.completeForest.estimators_[t]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    leafDepth = tm.node_depth[v]
                    self.ilfTreeDepth[t] += (leafDepth + _average_path_length(
                        [tree.tree_.n_node_samples[v]])[0]) * tm.y_var[v]
        # Get the ILF anomaly score as average of all trees anomaly score
        self.ilfAnomaly = self.model.addVar(
            lb=0.0, vtype=GRB.CONTINUOUS, name="ilfAnomalyScore")
        meanDepth = sum(self.ilfTreeDepth[t]
                        for t in ilfTreeIndices) / len(ilfTreeIndices)
        # ... and normalize by the average depth of a Binary tree search
        avgPathLengthBST = _average_path_length(
            [self.isolationForest.max_samples_])[0]
        # Add inlier constraint
        self.model.addConstr(self.ilfAnomaly == meanDepth / avgPathLengthBST)
        minScore = -np.log(-self.isolationForest.offset_)/np.log(2)
        self.model.addConstr(self.ilfAnomaly >= minScore - 100*self.targetRelax,
                             "isolationForestInlierConstraint")

    def addInterTreeConstraints(self):
        """Add constraints to express x_cf as a linear combination of planes"""
        treeIndices = range(self.completeForest.n_estimators)
        # Consistency of continuous features and discrete/categorial features
        if self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            self.addPlaneConsistencyConstraints(treeIndices)
            self.addValidInequalitiesLambda(treeIndices)
        self.addDiscreteVariablesConsistencyConstraints(treeIndices)
        # Find the prediction score
        self.treePrediction = dict()
        self.addTreePredictionConstraints(
            self.completeForest.randomForestEstimatorsIndices)
        # Add constraint on target class
        self.addTargetClassConstraint()

    def addPlaneConsistencyConstraints(self, treeIndices, neighbour=None):
        self.planes = dict()
        for f in self.continuousFeatures:
            self.planes[f] = dict()
            # Add the initial value as a plane
            self.planes[f][self.x0[0][f]] = []

        # Go over all relevant trees, for each node read the feature on which it splits
        # Get the split level (=thres) and add it to the list of split levels self.planes[f]
        for t in treeIndices:
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Numeric:
                        thres = tm.tree.threshold[v]
                        newPlane = True
                        if self.planes[f]:
                            nearestThres = min(
                                self.planes[f].keys(), key=lambda k: abs(k-thres))
                            if abs(thres - nearestThres) < 0.8*self.epsSplit:
                                # Do not create a plane, but instead add the (t,v) pair
                                # to its nearest threshold
                                newPlane = False
                                self.planes[f][nearestThres].append((t, v))
                        # Add plane
                        if newPlane:
                            self.planes[f][thres] = [(t, v)]

        self.rightPlanesVar = dict()  # this is \mu_i^j
        self.rightMutuallyExclusivePlanesConstr = dict()
        self.rightPlanesDominateRightFlowConstr = dict()
        self.rightPlanesOrderConstr = dict()
        self.linearCombinationOfPlanesConstr = dict()
        for f in self.continuousFeatures:
            self.rightPlanesVar[f] = dict()
            self.rightMutuallyExclusivePlanesConstr[f] = dict()
            self.rightPlanesDominateRightFlowConstr[f] = dict()
            self.rightPlanesOrderConstr[f] = dict()
            previousThres = 0  # this is the split level x_i^{j+1}
            linearCombination = gp.LinExpr(0.0)
            self.rightPlanesVar[f][previousThres] = self.model.addVar(
                lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                name="rightPlanesVar_f" + str(f) + "_th" + str(previousThres))
            for thres in sorted(self.planes[f]):
                self.rightPlanesVar[f][thres] = self.model.addVar(
                    lb=0.0, ub=1, vtype=GRB.CONTINUOUS,
                    name="rightPlanesVar_f" + str(f) + "_th" + str(thres))
                self.rightMutuallyExclusivePlanesConstr[f][thres] = []
                self.rightPlanesDominateRightFlowConstr[f][thres] = []

                # Add Constraints (16) to (18)
                for t, v in self.planes[f][thres]:
                    tm = self.treeManagers[t]
                    # Constraint (16)
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(self.model.addConstr(
                        self.rightPlanesVar[f][thres] <= 1
                        - tm.y_var[tm.tree.children_left[v]],
                        "rightPlanesVar_f"+str(f)+"_t"+str(t)+"_v"+str(v)))
                    # Constraint (17)
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(self.model.addConstr(
                        self.rightPlanesVar[f][previousThres] >= tm.y_var[tm.tree.children_right[v]],
                        "rightPlanesDominatesLeftFlowConstr_t"+str(t)+"_v"+str(v)))
                    # Avoid numerical precision errors
                    self.rightMutuallyExclusivePlanesConstr[f][thres].append(self.model.addConstr(
                        (thres - previousThres) * self.rightPlanesVar[f][previousThres] <= (
                            thres - previousThres) - min(thres - previousThres, self.epsSplit) * tm.y_var[tm.tree.children_left[v]],
                        "rightPlanesVar_eps_f"+str(f)+"_t"+str(t)+"_v"+str(v)))
                    self.rightPlanesDominateRightFlowConstr[f][thres].append(self.model.addConstr(
                        self.epsSplit * tm.y_var[tm.tree.children_right[v]]
                        <= self.rightPlanesVar[f][thres] * max(self.epsSplit, (thres - previousThres)),
                        "rightPlanesDominatesLeftFlowConstr_eps_t"+str(t)+"_v"+str(v)))

                # Constraint (15): the splits are in decreasing order
                self.rightPlanesOrderConstr[f][thres] = self.model.addConstr(
                    self.rightPlanesVar[f][previousThres] >= self.rightPlanesVar[f][thres],
                    "rightPlanesOrderConstr_f"+str(f)+"_th"+str(thres))

                # Partial expression of Equation (20)
                linearCombination += self.rightPlanesVar[f][previousThres] * (
                    thres - previousThres)
                # Update the feature splitting thresholds to go to next iteration
                previousThres = thres
            # Last split
            linearCombination += self.rightPlanesVar[f][previousThres] * (
                1.0 - previousThres)
            # Constraint (20)
            self.linearCombinationOfPlanesConstr[f] = self.model.addConstr(
                    self.x_var_sol[f] == linearCombination, "x_as_linear_combination_of_planes_f")

    def addValidInequalitiesLambda(self, treeIndices):
        """
        Add constraints to link the binary variables across different trees.
        If two trees branch on the same feature at the same depth,
        then we can link their branching variable.
        """
        # -- Link branching binary variables at root: depth=0 --
        for f in range(len(self.featuresType)):
            if self.featuresType[f] == FeatureType.Numeric:
                # Get the list of trees branching on f at root node
                # and the value of the split level
                treesBranchingOnF = []
                tresholdList = []
                for t in treeIndices:
                    tm = self.treeManagers[t]
                    if tm.tree.feature[0] == f:
                        treesBranchingOnF.append(t)
                        tresholdList.append(tm.tree.threshold[0])
                # Add constraints that all lambda are ordered
                self.addOrderingConstraintsOnBranchingVar(0, tresholdList,
                                                          treesBranchingOnF)
        # -- Link branching binary variables at root: depth=d --
        # Collect the trees that branch at each possible depth
        # and collect the list of nodes branching at this depth for each tree
        treeBranchingAtDepth = dict()
        nodesInTreesBranchingAtDepth = dict()
        for t in treeIndices:
            tm = self.treeManagers[t]
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    depth = tm.node_depth[v]
                    # Add the depth to the treeBranchingAtDepth list
                    if depth in treeBranchingAtDepth:
                        if t not in treeBranchingAtDepth[depth]:
                            treeBranchingAtDepth[depth].append(t)
                    else:
                        treeBranchingAtDepth[depth] = [t]
                    # Add the node to the nodesInTreesBranchingAtDepth list
                    if (depth, t) in nodesInTreesBranchingAtDepth:
                        nodesInTreesBranchingAtDepth[(depth, t)].append(v)
                    else:
                        nodesInTreesBranchingAtDepth[(depth, t)] = [v]

        # Add the ordering constraints at each depth
        # ordering constraint already implemented at depth=0
        del treeBranchingAtDepth[0]
        for depth in treeBranchingAtDepth:
            for f in range(len(self.featuresType)):
                if self.featuresType[f] == FeatureType.Numeric:
                    treesBranchingOnF = []
                    minTresholdList = dict()
                    maxThresholdList = dict()
                    # Get the list of trees branching on f at root node
                    # and for each tree - the min threshold over the nodes
                    #                   - the max threshold over the nodes
                    for t in treeBranchingAtDepth[depth]:
                        tm = self.treeManagers[t]
                        featureList = [tm.tree.feature[v]
                                       for v in nodesInTreesBranchingAtDepth[(depth, t)]]
                        allNodesAreBranchingOnF = featureList.count(
                            f) == len(featureList)
                        if allNodesAreBranchingOnF:
                            treesBranchingOnF.append(t)
                            thresholdList = [tm.tree.threshold[v]
                                             for v in nodesInTreesBranchingAtDepth[(depth, t)]]
                            minTresholdList[t] = min(thresholdList)
                            maxThresholdList[t] = max(thresholdList)
                    # Add ordering constraints on all pairs of trees that satisfy:
                    # max(\tau in tree i) < min(\tau in tree j)
                    for t1 in treesBranchingOnF:
                        for t2 in treesBranchingOnF:
                            if maxThresholdList[t1] < minTresholdList[t2]:
                                tm1 = self.treeManagers[t1]
                                tm2 = self.treeManagers[t2]
                                # Add constraint: lambda_nextTree >= lambda_tree
                                self.model.addConstr(tm1.tree_branching_vars[depth]
                                                     <= tm2.tree_branching_vars[depth])

    def addOrderingConstraintsOnBranchingVar(self, depth,
                                             tresholdList, treesBranchingOnF):
        """ Add constraints on branching variables ordering. """
        thresholdOrderedArgs = np.argsort(tresholdList)
        for t in range(len(treesBranchingOnF)-1):
            treeIndex = treesBranchingOnF[thresholdOrderedArgs[t]]
            nextTreeIndex = treesBranchingOnF[thresholdOrderedArgs[t+1]]
            treeMng = self.treeManagers[treeIndex]
            nextTreeMng = self.treeManagers[nextTreeIndex]
            # Add constraint: lambda_nextTree >= lambda_tree
            self.model.addConstr(nextTreeMng.tree_branching_vars[0]
                                 >= treeMng.tree_branching_vars[0])

    def addDiscreteVariablesConsistencyConstraints(self, treeIndices):
        # Discrete variables consistency constraints
        self.leftDiscreteVariablesConsistencyConstraints = dict()
        self.rightDiscreteVariablesConsistencyConstraints = dict()
        for t in treeIndices:
            tm = self.treeManagers[t]
            self.leftDiscreteVariablesConsistencyConstraints[t] = dict()
            self.rightDiscreteVariablesConsistencyConstraints[t] = dict()
            for v in range(tm.n_nodes):
                if not tm.is_leaves[v]:
                    f = tm.tree.feature[v]
                    if self.featuresType[f] == FeatureType.Discrete or self.featuresType[f] == FeatureType.CategoricalNonOneHot:
                        thres = tm.tree.threshold[v]
                        levels = list(self.featuresPossibleValues[f])
                        levels.append(1.0)
                        v_level = -1
                        for l in range(len(levels)):
                            if levels[l] > thres:
                                v_level = l
                                break
                        # Constraint (24)
                        self.leftDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level]
                            + tm.y_var[tm.tree.children_left[v]] <= 1,
                            "leftDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v))
                        # Constraint (25)
                        self.rightDiscreteVariablesConsistencyConstraints[t][v] = self.model.addConstr(
                            self.discreteFeaturesLevel_var[f][v_level] >= tm.y_var[tm.tree.children_right[v]],
                            "rightDiscreteVariablesConsistencyConstraints_t"
                            + str(t) + "_v" + str(v))

    def initObjective(self):
        assert self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes
        self.initLinearCombinationOfPlanesObj()

    def get_l1_norm_to_training_samples(self, absValUpperBound, nbTraining):
        """
        Get the l1-norm between the counterfactual explanation
        and all samples in the (restricted) training set using a scaling factor
        for discrete features.
        """
        self.trainAbsDiffs = dict()
        self.trainDiffs = dict()
        for i in range(nbTraining):
            self.trainAbsDist[i] = self.model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name="trainAbsDist_"+str(i))
            # Calculate the absolute distance to all training samples
            self.trainAbsDiffs[i] = dict()
            self.trainDiffs[i] = dict()
            for f in range(self.nFeatures):
                if self.featuresType[f] == FeatureType.Numeric:
                    scaleFactor = 1
                elif self.featuresType[f] in [FeatureType.Binary, FeatureType.Discrete, FeatureType.CategoricalNonOneHot]:
                    # Scaling factor for discrete/categorical/binary features
                    scaleFactor = self.discreteObjScale
                else:
                    print("Warning: wrong feature type in LOF implementation.")
                self.trainAbsDiffs[i][f] = self.model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name="trainAbsDiff_"+str(i)+"_f"+str(f))
                self.trainDiffs[i][f] = self.model.addVar(
                    lb=-1.0, vtype=GRB.CONTINUOUS,
                    name="trainAbsDiff_"+str(i)+"_f"+str(f))
                # Linearization of the absolute difference distance
                self.model.addConstr(
                    self.trainDiffs[i][f] == scaleFactor * (self.x_var_sol[f] - self.x_targetClass[i][f]))
                self.model.addConstr(
                    self.trainAbsDiffs[i][f] == gp.abs_(self.trainDiffs[i][f]))
            # Sum absolute differences over features
            self.model.addConstr(self.trainAbsDist[i] == sum(
                self.trainAbsDiffs[i][f] for f in range(self.nFeatures)),
                "trainAbsConstr_"+str(i))

    def addLOFPenaltyCost(self):
        """
        Implement the 1-local outlier factor as a penalty cost similar to:
        Kanamori, K., Takagi, T., Kobayashi, K., & Arimura, H. (2020, July).
        DACE: Distribution-Aware Counterfactual Explanation by Mixed-Integer
        Linear Optimization. In IJCAI (pp. 2855-2862).
        """
        absValUpperBound = self.nFeatures+1
        nbTraining = len(self.x_targetClass)
        # Calcualte l1 distance between x_cf and each training sample
        self.trainAbsDist = dict()
        self.get_l1_norm_to_training_samples(absValUpperBound, nbTraining)
        # - Find 1-NN -
        # Define binary indicator variables
        self.is1NN = dict()
        for i in range(nbTraining):
            self.is1NN[i] = self.model.addVar(lb=0.0, vtype=GRB.BINARY,
                                              name="is1NN"+str(i))
        # Implement constraints to find nearest neighbour
        for i in range(nbTraining):
            for j in range(nbTraining):
                # Add constraints to find the 1-nearest neighbor
                self.model.addConstr(
                    self.trainAbsDist[i] - self.trainAbsDist[j] <= absValUpperBound * (1-self.is1NN[i]))
            self.model.addConstr(
                sum(self.is1NN[i] for i in range(nbTraining)) == 1)
        # - Local outlier factor -
        self.reachabilityDist = dict()
        for i in range(nbTraining):
            self.reachabilityDist[i] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                         name="reachabilityDist_"+str(i))
            # Add constraint to find the 1-LOF as max of distanceTo1neighbors and trainAbsDist
            self.model.addConstr(
                self.reachabilityDist[i] >= self.distanceTo1neighbors[i] * self.is1NN[i])
            self.model.addConstr(
                self.reachabilityDist[i] >= self.trainAbsDist[i] - absValUpperBound * (1-self.is1NN[i]))
            # Add the 1-LOF cost to objective
            self.obj += self.lofWeight * \
                self.localReachDens[i] * self.reachabilityDist[i]

    def lossFunctionValue(self, f, xf):
        if self.featuresType[f] == FeatureType.CategoricalNonOneHot:
            if abs(xf - self.x0[0][f]) > self.epsSplit:
                return 1.0
            else:
                return 0.0
        elif self.objectiveNorm == 0:
            if abs(xf - self.x0[0][f]) > self.epsSplit:
                return 1.0
            else:
                return 0.0
        elif self.objectiveNorm == 1:
            return abs(xf - self.x0[0][f])
        elif self.objectiveNorm == 2:
            return (xf - self.x0[0][f]) ** 2
        else:
            print("unsupported norm")
            return float("inf")

    def initLinearCombinationOfPlaneObjOfFeature(self, f):
        thresholds = list(self.rightPlanesVar[f].keys())
        assert 0.0 in thresholds
        if 1.0 not in thresholds:
            thresholds.append(1.0)
        else:
            thresholds.append(1.0 + self.epsSplit)
        self.obj += self.lossFunctionValue(f, 0.0)
        for t in range(len(self.rightPlanesVar[f])):
            thres = thresholds[t]
            cellVar = self.rightPlanesVar[f][thres]
            cellLb = thres
            cellUb = thresholds[t+1]
            self.obj += cellVar * \
                (self.lossFunctionValue(f, cellUb)
                 - self.lossFunctionValue(f, cellLb))

    def initDiscreteFeatureObj(self, f):
        self.obj += self.lossFunctionValue(
            f, self.featuresPossibleValues[f][0])
        for lIndex in range(1, len(self.featuresPossibleValues[f])):
            cellLb = self.featuresPossibleValues[f][lIndex-1]
            cellUb = self.featuresPossibleValues[f][lIndex]
            cellVar = self.discreteFeaturesLevel_var[f][lIndex]
            self.obj += cellVar * \
                (self.lossFunctionValue(f, cellUb)
                 - self.lossFunctionValue(f, cellLb)) * self.discreteObjScale

    def initBinaryFeatureObj(self, f):
        self.obj += (self.lossFunctionValue(f, 0) + self.x_var_sol[f] * (
            self.lossFunctionValue(f, 1) - self.lossFunctionValue(f, 0))) * self.discreteObjScale

    def initObjectiveStructures(self):
        assert self.objectiveNorm in [0, 1, 2]
        self.obj = gp.LinExpr(0.0)

    def initLinearCombinationOfPlanesObj(self):
        self.initObjectiveStructures()
        for f in self.continuousFeatures:
            self.initLinearCombinationOfPlaneObjOfFeature(f)
        for f in self.discreteFeatures:
            self.initDiscreteFeatureObj(f)
        for f in self.categoricalNonOneHotFeatures:
            self.initDiscreteFeatureObj(f)
        for f in self.binaryFeatures:
            self.initBinaryFeatureObj(f)
        # Add 1-LOF cost factor if method is used
        if self.useLocalOutlierFactor:
            self.addLOFPenaltyCost()
        # The target constraint is relaxed with high penalty cost
        # to always allow a feasible solution.
        # Infeasibility may happen for instance:
        # - with non-actionable features,
        # - very low error tolerance.
        PENALTY_FACTOR = 5000 * len(self.featuresType)
        self.obj += PENALTY_FACTOR * self.targetRelax
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def checkPredictionResult(self):
        # Read results of optimisation model
        x_sol = np.array(self.x_sol, dtype=np.float32)
        classifierScore = self.clf.predict_proba(x_sol)

        # Check if x_sol reaches the target class
        badPrediction = (
            self.targetClass not in np.argwhere(max(classifierScore)))
        if badPrediction:
            print("Error, the desired class is not the predicted one.")
        else:
            if self.verbose:
                print("The target class", self.targetClass,
                      "is one of the argmax of the prediction scores.")

                # Determine score of target class in MILP representation of the input classifier
                rfClassScores = [0 for i in self.clf.classes_]
                for t in range(self.clf.n_estimators):
                    tm = self.treeManagers[t]
                    for v in range(tm.n_nodes):
                        if tm.is_leaves[v]:
                            leaf_val = tm.tree.value[v][0]
                            tot = sum(leaf_val)
                            for output in range(len(leaf_val)):
                                rfClassScores[output] += tm.y_var[v].getAttr(
                                    GRB.Attr.X) * (leaf_val[output])/tot
                for p in range(len(rfClassScores)):
                    rfClassScores[p] /= self.clf.n_estimators

                if self.verbose:
                    print("Class scores given by sklearn: ", classifierScore)
                    print("Class scores given by RF MILP: ", rfClassScores)
                    print("Initial sample: ", [self.x0[0][i] for i in range(
                        len(self.x0[0]))], " with predicted class: ", self.clf.predict(self.x0))

                self.maxSkLearnError = 0.0
                self.maxMyMilpError = 0.0
                # Check decision path
                myMilpErrors = False
                skLearnErrors = False
                for t in range(self.clf.n_estimators):
                    estimator = self.clf.estimators_[t]
                    predictionPath = estimator.decision_path(self.x_sol)
                    predictionPathList = list(
                        [tuple(row) for row in np.transpose(predictionPath.nonzero())])
                    verticesInPath = [v for d, v in predictionPathList]
                    tm = self.treeManagers[t]
                    solutionPathList = [u for u in range(
                        tm.n_nodes) if tm.y_var[u].getAttr(GRB.attr.X) >= 0.1]
                    if verticesInPath != solutionPathList:
                        lastCommonVertex = max(
                            set(verticesInPath).intersection(set(solutionPathList)))
                        f = tm.tree.feature[lastCommonVertex]
                        if self.verbose:
                            print("Sklearn decision path ", verticesInPath,
                                  " and my MILP decision path ", solutionPathList)
                        if self.verbose:
                            print("Wrong decision vertex", lastCommonVertex, "Feature", f, " of type ", self.featuresType[f], " with threshold",
                                  tm.tree.threshold[lastCommonVertex], "solution feature value x_sol[f]=", self.x_sol[0][f])
                            nextVertex = -1
                            if (self.x_sol[0][f] <= tm.tree.threshold[lastCommonVertex]):
                                if self.verbose:
                                    print("x_sol[f] <= threshold, next vertex in decision path should be:",
                                          tm.tree.children_left[lastCommonVertex])
                                nextVertex = tm.tree.children_left[lastCommonVertex]
                            else:
                                if self.verbose:
                                    print("x_sol[f] > threshold,next vertex in decision path should be:",
                                          tm.tree.children_right[lastCommonVertex])
                                nextVertex = tm.tree.children_right[lastCommonVertex]
                            if nextVertex not in verticesInPath:
                                skLearnErrors = True
                                self.maxSkLearnError = max(self.maxSkLearnError, abs(
                                    self.x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
                                if self.verbose:
                                    print("sklearn is wrong")
                            if nextVertex not in solutionPathList:
                                print("MY MILP IS WRONG")
                                myMilpErrors = True
                                self.maxMyMilpError = max(self.maxMyMilpError, abs(
                                    self.x_sol[0][f]-tm.tree.threshold[lastCommonVertex]))
                            if skLearnErrors and not myMilpErrors:
                                print("Only sklearn numerical precision errors")

    def checkIsolationForestResults(self):
        x_sol = np.array(self.x_sol, dtype=np.float32)
        if self.isolationForest.predict(x_sol)[0] == 1:
            if self.verbose:
                print("Isolation Forest: Result is an inlier")
        else:
            assert self.isolationForest.predict(x_sol)[0] == -1
            print("Isolation Forest: --!--Result is an outlier--!--")
        if self.verbose:
            anomalyScore = 2 ** (- self.ilfAnomaly.getAttr(GRB.Attr.X))
            print("         My MILP: Conformity score = {}".format(
                - (anomalyScore + self.isolationForest.offset_)))
            print("         Sklearn: Conformity score = {}".format(
                    self.isolationForest.decision_function(x_sol)[0]))

    def checkLOFResults(self):
        for i in range(len(self.x_targetClass)):
            is1NN = self.is1NN[i].getAttr(GRB.Attr.X)
            if is1NN == 1:
                # Check that the right 1-NN has been found
                absDistances = [l1_norm_with_scaling_factor(
                    self.x_sol[0], self.x_targetClass[k], self.featuresType, len(self.featuresType))
                                for k in range(len(self.x_targetClass))]
                indexOfTrue1NN = np.argmin(absDistances)
                try:
                    assert (absDistances[indexOfTrue1NN]
                            - absDistances[i]) < 1e-9
                except AssertionError:
                    print("1-NN should have index {} with l1 distance {}".format(
                        indexOfTrue1NN, absDistances[indexOfTrue1NN]))
                    print("Instead, the identified sample has index {} with l1 distance {}".format(
                        i, absDistances[i]))
                    raise Exception("Error in identifying 1-NN.")
                # Print results
                if self.verbose:
                    lof = self.localReachDens[i] * \
                        self.reachabilityDist[i].getAttr(GRB.Attr.X)
                    print("LOF: 1-NN index is {} with 1-LOF = {}".format(i, lof))

    def solveModel(self):
        self.model.optimize()

        self.runTime = self.model.Runtime
        if self.model.status != GRB.OPTIMAL:
            self.objValue = "inf"
            self.maxSkLearnError = "inf"
            self.maxMyMilpError = "inf"
            self.x_sol = self.x0
            return False

        # Read solution
        self.objValue = self.model.ObjVal
        self.x_sol = [[]]
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X))

        # Check results
        self.checkPredictionResult()
        if self.isolationForest:
            self.checkIsolationForestResults()
        if self.useLocalOutlierFactor:
            self.checkLOFResults()
        self.targetRelaxOutput = self.targetRelax.getAttr(GRB.Attr.X)
        if self.verbose:
            print("Counterfactual: ", self.x_sol,
                  " with predicted class: ", self.clf.predict(self.x_sol))
            print("Target constraint relaxed by: {}".format(
                      self.targetRelaxOutput))
        return True
