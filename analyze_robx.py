"""
Analyze the results of the repeated experiments with
the method from Dutta et al.
- read simulations results from csv files,
- calculate kpis (cost and validity),
- plot relevant figures,
- export formatted results to csv files.
"""
# Import packages
import os
import numpy as np
import pandas as pd
# Import local functions
from src.result_analysis import read_cost_and_validity_from_results_df
from src.result_analysis import compute_kpis_and_confidence_intervals
from src.result_analysis import export_results_csv

# ------ Read simulations results ------
# Import result files and concatenate them into a single Dataframe
datasetList = ['Spambase']
# Specify subfolder for results files
projectPath = os.path.dirname(__file__)
experimentFolder = 'robustTrainingCounterfactuals'
subPath = projectPath+'/output/'+experimentFolder+'/'

# -------- RobX: from Dutta et al. --------
method = 'robx'
for datasetName in datasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath)
                 if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    # Read relevant parameters
    tauList = -np.sort(-pd.unique(methodDf['tau']))
    nbTaus = len(tauList)
    m_n_list = pd.unique(methodDf['m_n'])
    nbMns = len(m_n_list)
    nbSimulations = int(len(methodDf) / (nbTaus * nbMns))
    # Measure cost and validity of cf
    methoSensParameter = 'tau'
    cfValidity, cfCost = read_cost_and_validity_from_results_df(
        methodDf, methoSensParameter, tauList, nbTaus, m_n_list, nbMns,
        nbSimulations, datasetName)
    # ------ Analysis of simulations results ------
    avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
        cfValidity, cfCost, tauList, nbMns, nbSimulations)
    # Export formatted results to csv files --
    export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       tauList, experimentFolder, projectPath,
                       invert_alpha=False)
