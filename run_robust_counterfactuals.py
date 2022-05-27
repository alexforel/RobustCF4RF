"""
Generate counterfactual explanations that are robust to training uncertainty
for random forests. Run all methods and benchmarks on all data sets.
"""
# Import packages
import gurobipy as GBP
import pandas as pd
import os
from src.simulations import run_single_experiment

dirname = os.path.dirname(__file__)
# Create gurobi environment
gurobiEnv = GBP.Env()
gurobiEnv.setParam('OutputFlag', 0)
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
# Simulation setting parameters
NB_SIMULATIONS = 40
BATCH_SIZE = 5  # number of counterfactuals per simulation run
# Define list of datasets from all available:
datasetList = ['German-Credit', 'Students-Performance', 'Spambase',
               'COMPAS', 'Phishing', 'Credit-Card-Default',
               'OnlineNewsPopularity', 'Adult']
alphaList = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
MAX_TREE_DEPTH = 4
m_n = 100

# ---- Run simulations for naive method ----
method = 'naive'
for datasetName in datasetList:
    print("----- Naive method on dataset: {} -----".format(datasetName))
    resultDataframe = pd.DataFrame()
    for simIndex in range(NB_SIMULATIONS):
        print("- Simulation index: {} out of {}. -".format(
                simIndex+1, NB_SIMULATIONS))
        partialResultDf = run_single_experiment(
            datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
            simIndex, NB_SIMULATIONS, gurobiEnv)
        resultDataframe = pd.concat(
            [resultDataframe, partialResultDf],
            ignore_index=True, axis=0)
    # Save results fo csv file
    folderName = os.path.join(
            dirname, "output", "robustTrainingCounterfactuals", datasetName)
    os.makedirs(folderName, exist_ok=True)
    resultDataframe.to_csv(folderName+'/'+datasetName
                           + '_robustCounterfactuals_{}.csv'.format(
                               method))

# ---- Run simulations for Direct-SAA method ----
method = 'direct-saa'
for datasetName in datasetList:
    print("----- Direct-SAA method on dataset: {} -----".format(datasetName))
    for alpha in alphaList:
        print("--- alpha={} ---".format(alpha))
        resultDataframe = pd.DataFrame()
        for simIndex in range(NB_SIMULATIONS):
            print("- Simulation index: {} out of {}. -".format(
                    simIndex+1, NB_SIMULATIONS))
            partialResultDf = run_single_experiment(
                datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
                simIndex, NB_SIMULATIONS, gurobiEnv, alpha=alpha)
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # Save results fo csv file
        folderName = os.path.join(
                dirname, "output", "robustTrainingCounterfactuals", datasetName)
        os.makedirs(folderName, exist_ok=True)
        resultDataframe.to_csv(folderName+'/'+datasetName
                               + '_robustCounterfactuals_alpha={}_{}.csv'.format(
                                   alpha, method))

# ---- Run simulations for Robust-SAA method ----
method = 'robust-saa'
datasetList = ['Spambase']
betaList = [0.1, 0.05]
for datasetName in datasetList:
    print("----- Robust-SAA method on dataset: {} -----".format(datasetName))
    for alpha in alphaList:
        print("--- alpha={} ---".format(alpha))
        for beta in betaList:
            resultDataframe = pd.DataFrame()
            for simIndex in range(NB_SIMULATIONS):
                print("- Simulation index: {} out of {}. -".format(
                        simIndex+1, NB_SIMULATIONS))
                partialResultDf = run_single_experiment(
                    datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
                    simIndex, NB_SIMULATIONS, gurobiEnv, alpha=alpha, beta=beta)
                resultDataframe = pd.concat(
                    [resultDataframe, partialResultDf],
                    ignore_index=True, axis=0)
            # Save results fo csv file
            folderName = os.path.join(
                    dirname, "output", "robustTrainingCounterfactuals", datasetName)
            os.makedirs(folderName, exist_ok=True)
            resultDataframe.to_csv(
                folderName+'/'+datasetName
                + '_robustCounterfactuals_alpha={}_{}_beta={}.csv'.format(
                                       alpha, method, beta))

# ---- Run simulations for isolation forest method ----
method = 'isolation-forest'
datasetList = ['German-Credit', 'Students-Performance', 'Spambase',
               'COMPAS', 'Phishing', 'Credit-Card-Default',
               'OnlineNewsPopularity', 'Adult']
contaminationList = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
for datasetName in datasetList:
    print("----- Isolation forest method on dataset: {} -----".format(datasetName))
    for contamination in contaminationList:
        print("--- contamination={} ---".format(contamination))
        resultDataframe = pd.DataFrame()
        for simIndex in range(NB_SIMULATIONS):
            print("- Simulation index: {} out of {}. -".format(
                    simIndex+1, NB_SIMULATIONS))
            partialResultDf = run_single_experiment(
                datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
                simIndex, NB_SIMULATIONS, gurobiEnv,
                contamination=contamination)
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # Save results fo csv file
        folderName = os.path.join(
                dirname, "output", "robustTrainingCounterfactuals", datasetName)
        os.makedirs(folderName, exist_ok=True)
        resultDataframe.to_csv(folderName+'/'+datasetName
                               + '_robustCounterfactuals_c={}_{}.csv'.format(
                                   contamination, method))

# ---- Run simulations for Local Outlier Factor method ----
method = 'lof'
lofDatasetList = ['German-Credit', 'Students-Performance',
                  'COMPAS', 'Phishing', 'Credit-Card-Default',
                  'OnlineNewsPopularity', 'Adult', 'Spambase']
lofWeightList = [0.001, 0.01, 0.1, 1, 10, 100]
for datasetName in lofDatasetList:
    print("----- Local outlier factor method on dataset: {} -----".format(datasetName))
    for lofWeight in lofWeightList:
        print("--- lofWeight={} ---".format(lofWeight))
        resultDataframe = pd.DataFrame()
        for simIndex in range(NB_SIMULATIONS):
            if (datasetName == 'OnlineNewsPopularity') and (lofWeight in [0.1, 1, 10, 100]):
                # Reduce the relative MIPGap for the 'OnlineNewsPopularity' dataset
                gurobiEnv.setParam('MIPGap', 0.1)
            else:
                # Use default value for MIPGap
                gurobiEnv.setParam('MIPGap', 1e-4)
            print("- Simulation index: {} out of {}. -".format(
                    simIndex+1, NB_SIMULATIONS))
            partialResultDf = run_single_experiment(
                datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
                simIndex, NB_SIMULATIONS, gurobiEnv,
                lofWeight=lofWeight)
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # Save results fo csv file
        folderName = os.path.join(
                dirname, "output", "robustTrainingCounterfactuals", datasetName)
        os.makedirs(folderName, exist_ok=True)
        resultDataframe.to_csv(folderName+'/'+datasetName
                               + '_robustCounterfactuals_lambda={}_{}.csv'.format(
                                   lofWeight, method))
