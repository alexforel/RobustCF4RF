# Import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Import custom functions
from src.CfSimulator import CfSimulator
from src.binomial_confidence import p_star_threshold, robust_p_star_threshold
from src.robx import measure_stability_training_set, dutta_counterfactual


def run_single_experiment(datasetName, method, forestSize,
                          maxTreeDepth, batchSize, simIndex,
                          nbSimulations, gurobiEnv,
                          alpha=0.5, beta=0.0,
                          contam=0.0, lofWeight=0.0,
                          tau=0.0):
    """
    Run a single experiment of the counterfactual generation simulations
    """
    # ---- 0/ Initialise simulation setting ----
    # Create simulator object
    cfSimulator = CfSimulator(datasetName, batchSize, simIndex,
                              testSplitFactor=0.0,
                              verbose=False)
    # Create and train random forest classifier
    rfClassifier = RandomForestClassifier(n_estimators=forestSize,
                                          max_depth=maxTreeDepth,
                                          max_features="sqrt",
                                          random_state=simIndex)
    rfClassifier.fit(cfSimulator.x_train, cfSimulator.y_train)

    # ---- 1/ Generate counterfactuals ----
    if method == 'naive':
        # Target class constraint: h(x) ≥ 1/2
        cfSimulator.generate_robust_counterfactuals(rfClassifier, gurobiEnv)
    elif method == 'direct-saa':
        generate_direct_saa_counterfactuals(forestSize, alpha, rfClassifier,
                                            cfSimulator, gurobiEnv)
    elif method == 'robust-saa':
        generate_robust_saa_counterfactuals(forestSize, alpha, rfClassifier,
                                            cfSimulator, gurobiEnv,
                                            beta=beta)
    elif method == 'isolation-forest':
        cfSimulator.generate_robust_counterfactuals(rfClassifier, gurobiEnv,
                                                    useIsolationForest=True,
                                                    contamination=contam)
    elif method == 'lof':
        cfSimulator.generate_robust_counterfactuals(rfClassifier, gurobiEnv,
                                                    useLocalOutlierFactor=True,
                                                    lofWeight=lofWeight)
    elif method == 'robx':
        # Measure the stability of all training points
        stability_train = measure_stability_training_set(cfSimulator.x_train,
                                                         rfClassifier)
        # Get the set of stable points
        mask = (np.array(stability_train) > tau)
        x_stable = cfSimulator.x_train[mask, :]
        generate_dutta_counterfactual(rfClassifier,
                                      cfSimulator, gurobiEnv,
                                      batchSize, x_stable, tau)

    # ---- 2/ Validate counterfactual ----
    # Train a test classifier on very large datasets
    testClassifier = RandomForestClassifier(
        n_estimators=forestSize, max_depth=maxTreeDepth,
        max_features="sqrt", random_state=nbSimulations+simIndex)
    testClassifier.fit(cfSimulator.x_train,
                       cfSimulator.y_train)
    cfSimulator.verify_counterfactual_accuracy(testClassifier)
    # ---- 3/ Store results in dataframe ----
    partialResultDf = pd.DataFrame(
        {'datasetName': [datasetName] * batchSize,
         'simIndex': [simIndex] * batchSize,
         'alpha': [alpha] * batchSize,
         'beta': [beta] * batchSize,
         'contamination': [contam] * batchSize,
         'lofWeight': [lofWeight] * batchSize,
         'tau': [tau] * batchSize,
         'm_n': [forestSize] * batchSize,
         'method': [method] * batchSize,
         'x_init': [str(cfSimulator.x_init[i]) for i in range(batchSize)],
         'y_init': cfSimulator.y_init,
         'x_cf': [str(cfSimulator.x_cf[i]) for i in range(batchSize)],
         'y_test': cfSimulator.y_true_cf_label,
         'robustConsRelaxation': cfSimulator.robConstrRelax,
         'runTime': cfSimulator.runTime})
    return partialResultDf


def generate_direct_saa_counterfactuals(m_n, alpha, rfClassifier,
                                        cfSimulator, gurobiEnv,
                                        useIsolationForest=False):
    """
    Generate counterfactuals using a binomial probability model
    with naive estimation of the success rate p.
    """
    calibratedAlpha = p_star_threshold(m_n, alpha)
    # Solve OCEAN MILP to find min-cost counterfactual
    cfSimulator.generate_robust_counterfactuals(
        rfClassifier, gurobiEnv, useCalibratedAlpha=True,
        calibratedAlpha=calibratedAlpha, useIsolationForest=useIsolationForest)


def generate_robust_saa_counterfactuals(m_n, alpha, rfClassifier,
                                        cfSimulator, gurobiEnv, beta=0.05,
                                        useIsolationForest=False):
    """
    Generate counterfactuals using a binomial probability model
    with robust estimation of the success rate p based on
    Agresti-Coull confidence interval.
    """
    robustAlpha = robust_p_star_threshold(m_n, alpha, beta)
    # Solve OCEAN MILP to find min-cost robust counterfactual
    cfSimulator.generate_robust_counterfactuals(
        rfClassifier, gurobiEnv, useCalibratedAlpha=True,
        calibratedAlpha=robustAlpha, useIsolationForest=useIsolationForest)


def generate_dutta_counterfactual(rfClassifier, cfSimulator, gurobiEnv,
                                  batchSize, x_stable, tau):
    """
    Generate a robust counterfactual explanation using the method of
    Dutta et al. that measures a local stability of the prediction
    function and moves naive explanations toward training
    examples with high stability.
    """
    # Step 1: get a naive counterfactual
    print('- Generate naive explanations.')
    cfSimulator.generate_robust_counterfactuals(rfClassifier, gurobiEnv)

    # Step 2: improve locally if not locally stable
    for i in range(batchSize):
        x_cf = cfSimulator.x_cf[i, :]
        cf = dutta_counterfactual(x_cf, x_stable, rfClassifier,
                                  cfSimulator, tau)
        cfSimulator.x_cf[i, :] = cf
