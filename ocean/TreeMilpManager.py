from gurobipy import GRB
from ocean.CounterFactualParameters import *


class TreeInMilpManager:
    treeCount = 0

    def __init__(self, tree, model, x_var_sol,
                 featuresType,
                 constraintsType=False,
                 binaryDecisionVariables=False):
        self.id = TreeInMilpManager.treeCount
        TreeInMilpManager.treeCount += 1
        self.model = model
        self.tree = tree
        self.x_var_sol = x_var_sol
        self.nFeatures = len(self.x_var_sol)
        self.constraintsType = constraintsType
        self.binaryDecisionVariables = binaryDecisionVariables
        assert featuresType
        self.featuresType = featuresType
        self.initTreeInfo()

    def initTreeInfo(self):
        self.n_nodes = self.tree.node_count
        self.is_leaves = dict()
        self.node_depth = dict()
        self.continuousFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Numeric]
        self.binaryFeatures = [f for f in range(
            self.nFeatures) if self.featuresType[f] == FeatureType.Binary]
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            self.node_depth[node_id] = depth
            is_split_node = self.tree.children_left[node_id] != self.tree.children_right[node_id]
            if is_split_node:
                stack.append((self.tree.children_left[node_id], depth + 1))
                stack.append((self.tree.children_right[node_id], depth + 1))
                self.is_leaves[node_id] = False
            else:
                self.is_leaves[node_id] = True

    def addTreeVariablesAndConstraintsToMilp(self):
        self.addBranchingAndDecisionPathVariablesAndConstraints()
        self.addContinuousVariablesConsistencyConstraints()
        self.addBinaryVariablesConsistencyConstraints()

    def addBranchingAndDecisionPathVariablesAndConstraints(self):
        self.y_var = dict()
        if self.binaryDecisionVariables == BinaryDecisionVariables.LeftRight_lambda:
            for v in range(self.n_nodes):
                self.y_var[v] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                                  name="y"+str(v)+"_t"+str(self.id))
            # Path and branching constraints
            self.root_constr_y = self.model.addConstr(
                self.y_var[0] == 1, "root_constr_y"+"_t"+str(self.id))
            self.flow_constr = dict()
            for v in range(self.n_nodes):
                if not self.is_leaves[v]:
                    # Constraint (9)
                    self.flow_constr[v] = self.model.addConstr(
                                self.y_var[v] == self.y_var[self.tree.children_left[v]] + self.y_var[self.tree.children_right[v]], "flow_" + str(v)+"_t"+str(self.id))

            # Initialise tree branching integer variables
            self.tree_branching_vars = dict()
            for depth in range(self.tree.max_depth):
                self.tree_branching_vars[depth] = self.model.addVar(
                    vtype=GRB.BINARY, name="lambda"+str(depth)+"_t"+str(self.id))

            ## ---- Constraint (10) - Implemented as in paper ----
            # # First, get the list of nodes at each depth
            # listOfNodesAtDepth = dict()
            # for depth in range(self.tree.max_depth):
            #     listOfNodesAtDepth[depth] = []
            #     for v in range(self.n_nodes):
            #         if not self.is_leaves[v]:
            #             if self.node_depth[v] == depth:
            #                 listOfNodesAtDepth[depth].append(v)
            # # Then, add constraint at each depth level
            # self.branch_constr = dict()
            # for depth in range(self.tree.max_depth - 1):  # No constraint at maxdepth
            #     self.branch_constr[depth] = self.model.addConstr(
            #             sum(self.y_var[self.tree.children_left[v]]
            #                 for v in listOfNodesAtDepth[depth])
            #             <= self.tree_branching_vars[depth],
            #             "branching_t"+str(self.id)+"_d"+str(depth))

            ## ----- Constraints from github code -----
            self.branch_constr_left = dict()
            self.branch_constr_right = dict()
            for v in range(self.n_nodes):
                if not self.is_leaves[v]:
                    self.branch_constr_left[v] = self.model.addConstr(
                        self.y_var[self.tree.children_left[v]] <= self.tree_branching_vars[self.node_depth[v]], "branch_left_v" + str(v)+"_t"+str(self.id))
                    self.branch_constr_right[v] = self.model.addConstr(
                        self.y_var[self.tree.children_right[v]] <= 1 - self.tree_branching_vars[self.node_depth[v]], "branch_right_v" + str(v)+"_t"+str(self.id))
        else:
            print("Error, unknown binary decision variables")

    def addContinuousVariablesConsistencyConstraints(self):
        if self.constraintsType == TreeConstraintsType.LinearCombinationOfPlanes:
            pass
        else:
            print("unknown constraints type")

    def addBinaryVariablesConsistencyConstraints(self):
        self.leftBinaryVariablesConsistencyConstraints = dict()
        self.rightBinaryVariablesConsistencyConstraints = dict()
        for v in range(self.n_nodes):
            if not self.is_leaves[v]:
                f = self.tree.feature[v]
                if self.featuresType[f] == FeatureType.Binary:
                    assert self.tree.threshold[v] > 0
                    assert self.tree.threshold[v] < 1
                    # Constraint (21)
                    self.leftBinaryVariablesConsistencyConstraints[v] = self.model.addConstr(
                        self.x_var_sol[f]
                        + self.y_var[self.tree.children_left[v]] <= 1,
                        "leftBinaryVariablesConsistencyConstraints_t"+str(self.id)+"_v" + str(v))
                    # Constraint (22)
                    self.rightBinaryVariablesConsistencyConstraints[v] = self.model.addConstr(
                        self.x_var_sol[f] >= self.y_var[self.tree.children_right[v]],
                        "rightBinaryVariablesConsistencyConstraints_t"+str(self.id)+"_v"+str(v))
