class RandomAndIsolationAndVarianceForest:
    """
    Combines all forests and trees into a single compleForest object.
    The complete forest may include:
        - the random forest classifier,
        - an isolation forest to improve plausibility,
        - a variance estimation forest,
        - a standard deviation regressor (tree or forest),
        - a regressor to predict the expected forest score (tree or forest),
        - neighbour forests, which are copies of the initial random forest
          classifier to track the score of neighbour of x_cf.
    """

    def __init__(self, randomForest, isolationForest=None):
        # Read and store inputs
        self.randomForest = randomForest
        self.isolationForest = isolationForest

        # n_estimators: number of trees in the full combined object
        self.n_estimators = self.randomForest.n_estimators
        # Store random forest into combine object
        self.randomForestEstimatorsIndices = [
            i for i in range(self.n_estimators)]
        self.estimators_ = [est for est in self.randomForest.estimators_]

        # ---- Isolation forest ----
        # Store and combine object with isolation forest
        if self.isolationForest:
            self.isolationForestEstimatorsIndices = [
                i + self.n_estimators for i in range(self.isolationForest.n_estimators)]
            self.n_estimators += self.isolationForest.n_estimators
            for est in self.isolationForest.estimators_:
                self.estimators_.append(est)
