"""
Generate counterfactual explanations that are robust to training uncertainty
for random forests using RobX.

Reference:
    Dutta, S., Long, J., Mishra, S., Tilli, C., & Magazzeni, D. (2022, June).
    Robust counterfactual explanations for tree-based ensembles.
    In International conference on machine learning (pp. 5742-5756). PMLR.
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
gurobiEnv.setParam('Threads', 1)
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
# Simulation setting parameters
NB_SIMULATIONS = 20
BATCH_SIZE = 5  # number of counterfactuals per simulation run
# Define list of datasets from all available:
datasetList = ['Spambase']
tauList = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
MAX_TREE_DEPTH = 4
m_n = 100

# ---- Run simulations for RobX method ----
method = 'robx'
for datasetName in datasetList:
    print("----- RobX method from Dutta et al. on dataset: {} -----".format(
        datasetName))
    for tau in tauList:
        print("--- tau={} ---".format(tau))
        resultDataframe = pd.DataFrame()
        for simIndex in range(NB_SIMULATIONS):
            print("- Simulation index: {} out of {}. -".format(
                    simIndex+1, NB_SIMULATIONS))
            partialResultDf = run_single_experiment(
                datasetName, method, m_n, MAX_TREE_DEPTH, BATCH_SIZE,
                simIndex, NB_SIMULATIONS, gurobiEnv, tau=tau)
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # Save results fo csv file
        folderName = os.path.join(
                dirname, "output", "robustTrainingCounterfactuals",
                datasetName)
        os.makedirs(folderName, exist_ok=True)
        resultDataframe.to_csv(folderName+'/'+datasetName
                               + '_robxCounterfactual_tau={}_{}.csv'.format(
                                   tau, method))
