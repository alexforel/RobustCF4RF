"""
Analyze the results of the repetated experiments to generate robust counterfactuals:
- read simulations results from csv files,
- calculate kpis (cost and validity),
- plot relevant figures,
- export formatted results to csv files.
"""
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
# Import local functions
from src.result_analysis import read_cost_and_validity_from_results_df
from src.result_analysis import compute_kpis_and_confidence_intervals
from src.result_analysis import export_results_csv
from src.result_analysis import plot_robust_counterfactuals_sensitivity
from src.result_analysis import plot_robust_constraint_relax_histogram

# ------ Read simulations results ------
# Import result files and concatenate them into a single Dataframe
datasetList = ['Students-Performance', 'German-Credit', 'Spambase',
               'COMPAS', 'Phishing', 'Credit-Card-Default',
               'OnlineNewsPopularity', 'Adult']
# Specify subfolder for results files
projectPath = 'C:/Code/robust-counterfactuals/'
experimentFolder = 'robustTrainingCounterfactuals'
subPath = projectPath+'/output/'+experimentFolder+'/'
SUBSTRACT_NAIVE_COSTS = False

# -------- Naive --------
# Calculate cost and validity with naive constraint: h(x) ≥ 1/2
method = 'naive'
for datasetName in datasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    # Read relevant parameters
    alphaList = -np.sort(-pd.unique(methodDf['alpha']))
    nbAlphas = len(alphaList)
    m_n_list = pd.unique(methodDf['m_n'])
    nbMns = len(m_n_list)
    nbSimulations = int(len(methodDf) / (nbAlphas * nbMns))
    # Measure cost and validity of cf
    methoSensParameter = 'alpha'
    cfValidity, cfCost = read_cost_and_validity_from_results_df(
        methodDf, methoSensParameter, alphaList, nbAlphas, m_n_list, nbMns,
        nbSimulations, datasetName)
    # ------ Analysis of simulations results ------
    avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
        cfValidity, cfCost, alphaList, nbMns, nbSimulations)
    # Export formatted results to csv files --
    export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       alphaList, experimentFolder, projectPath)

# -------- Direct-SAA --------
# Calculate cost and validity with SAA constraint: h(x) ≥ p^*_{N, α}
method = 'direct-saa'
for datasetName in datasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    # Read relevant parameters
    alphaList = -np.sort(-pd.unique(methodDf['alpha']))
    nbAlphas = len(alphaList)
    m_n_list = pd.unique(methodDf['m_n'])
    nbMns = len(m_n_list)
    nbSimulations = int(len(methodDf) / (nbAlphas * nbMns))
    # Measure cost and validity of cf
    methoSensParameter = 'alpha'
    cfValidity, cfCost = read_cost_and_validity_from_results_df(
        methodDf, methoSensParameter, alphaList, nbAlphas, m_n_list, nbMns,
        nbSimulations, datasetName)
    # ------ Analysis of simulations results ------
    avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
        cfValidity, cfCost, alphaList, nbMns, nbSimulations)
    # Export formatted results to csv files --
    export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       alphaList, experimentFolder, projectPath)
    # Plot trade-off between counterfactual cost and robustness
    plot_robust_counterfactuals_sensitivity(datasetName, avgVal, valConf,
                                            avgCost, costConf, alphaList,
                                            m_n_list, projectPath,
                                            experimentFolder)
    # Histogram of the relaxation of the robust constraint
    plot_robust_constraint_relax_histogram(datasetName, resultsDf, alphaList)

# -------- Robust-SAA --------
# Calculate cost and validity with SAA constraint: h(x) ≥ p^*_{N, α, β}
method = 'robust-saa'
robustDatasetList = ['Spambase']
betaList = [0.10, 0.05]
for datasetName in robustDatasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    for beta in betaList:
        betaDf = methodDf[methodDf['beta'] == beta]
        # Read relevant parameters
        alphaList = -np.sort(-pd.unique(betaDf['alpha']))
        nbAlphas = len(alphaList)
        m_n_list = pd.unique(betaDf['m_n'])
        nbMns = len(m_n_list)
        nbSimulations = int(len(betaDf) / (nbAlphas * nbMns))
        # Measure cost and validity of cf
        methoSensParameter = 'alpha'
        cfValidity, cfCost = read_cost_and_validity_from_results_df(
            betaDf, methoSensParameter, alphaList, nbAlphas, m_n_list, nbMns,
            nbSimulations, datasetName)
        # ------ Analysis of simulations results ------
        avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
            cfValidity, cfCost, alphaList, nbMns, nbSimulations)
        # Export formatted results to csv files --
        methodBeta = 'robust-saa-beta-'+str(beta)
        export_results_csv(methodBeta, datasetName, avgVal, valConf, avgCost, costConf,
                           alphaList, experimentFolder, projectPath)
        # Plot trade-off between counterfactual cost and robustness
        plot_robust_counterfactuals_sensitivity(datasetName, avgVal, valConf,
                                                avgCost, costConf, alphaList,
                                                m_n_list, projectPath,
                                                experimentFolder)
        # Histogram of the relaxation of the robust constraint
        plot_robust_constraint_relax_histogram(
            datasetName, resultsDf, alphaList)

# -------- Isolation Forest --------
method = 'isolation-forest'
for datasetName in datasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    # Read relevant parameters
    contaminationList = np.sort(pd.unique(methodDf['contamination']))
    nbCont = len(contaminationList)
    m_n_list = pd.unique(methodDf['m_n'])
    nbMns = len(m_n_list)
    nbSimulations = int(len(methodDf) / (nbCont * nbMns))
    # Measure cost and validity of cf
    methoSensParameter = 'contamination'
    cfValidity, cfCost = read_cost_and_validity_from_results_df(
        methodDf, methoSensParameter,  contaminationList, nbCont, m_n_list,
        nbMns, nbSimulations, datasetName)
    # ------ Analysis of simulations results ------
    avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
        cfValidity, cfCost, contaminationList, nbMns, nbSimulations)
    # Export formatted results to csv files --
    export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       contaminationList, experimentFolder, projectPath)
    # Plot trade-off between counterfactual cost and robustness
    plot_robust_counterfactuals_sensitivity(datasetName, avgVal, valConf,
                                            avgCost, costConf, contaminationList,
                                            m_n_list, projectPath,
                                            experimentFolder)
    # Histogram of the relaxation of the robust constraint
    plot_robust_constraint_relax_histogram(
        datasetName, resultsDf, contaminationList)

# -------- Local outlier factor --------
method = 'lof'
lofDatasetList = ['German-Credit', 'Students-Performance', 'Spambase',
                  'COMPAS', 'Phishing', 'Credit-Card-Default', 'Adult',
                  'OnlineNewsPopularity']
for datasetName in lofDatasetList:
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    # Filter to relevant counterfactual generation method
    methodDf = resultsDf[resultsDf['method'] == method]
    # Read relevant parameters
    lofWeightList = np.sort(pd.unique(methodDf['lofWeight']))
    nbWeights = len(lofWeightList)
    m_n_list = pd.unique(methodDf['m_n'])
    nbMns = len(m_n_list)
    nbSimulations = int(len(methodDf) / (nbWeights * nbMns))
    # Measure cost and validity of cf
    methoSensParameter = 'lofWeight'
    cfValidity, cfCost = read_cost_and_validity_from_results_df(
        methodDf, methoSensParameter,  lofWeightList, nbWeights, m_n_list,
        nbMns, nbSimulations, datasetName)
    # ------ Analysis of simulations results ------
    avgVal, valConf, avgCost, costConf = compute_kpis_and_confidence_intervals(
        cfValidity, cfCost, lofWeightList, nbMns, nbSimulations)
    # Export formatted results to csv files --
    export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       lofWeightList, experimentFolder, projectPath)
    # Plot trade-off between counterfactual cost and robustness
    plot_robust_counterfactuals_sensitivity(datasetName, avgVal, valConf,
                                            avgCost, costConf, lofWeightList,
                                            m_n_list, projectPath,
                                            experimentFolder)
    # Histogram of the relaxation of the robust constraint
    plot_robust_constraint_relax_histogram(
        datasetName, resultsDf, lofWeightList)

# -------- Computation times --------
# Compare the times of the different methods to obtain a counterfactual
methodList = ['naive', 'direct-saa', 'isolation-forest', 'lof', 'robust-saa']
runtimesDf = pd.DataFrame()
for method in methodList:
    for datasetName in datasetList:
        resultFolderPath = subPath+datasetName+'/'
        filepaths = [resultFolderPath+"/"
                     + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
        resultsDf = pd.concat(map(pd.read_csv, filepaths))
        methodDf = resultsDf[resultsDf['method'] == method]
        partialDf = methodDf[["method", "datasetName", "runTime"]]
        runtimesDf = pd.concat([runtimesDf, partialDf])
# Rename the methods
runtimesDf['method'][runtimesDf['method'] == 'naive'] = 'Naive'
runtimesDf['method'][runtimesDf['method'] == 'direct-saa'] = 'Direct SAA'
runtimesDf['method'][runtimesDf['method'] == 'isolation-forest'] = 'IForest'
runtimesDf['method'][runtimesDf['method'] == 'lof'] = '1-LOF'
runtimesDf['method'][runtimesDf['method'] == 'robust-saa'] = 'Robust SAA'
# Transform runtime to log seconds
runtimesDf['runTime'] = np.log10(runtimesDf['runTime'])
# - Plot runtimes as boxplot for each (dataset, method) -
# Set parameters of matplotlib to avoid Type 3 fonts as per NeurIPS guidelines
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(9, 4))
cubehelixPalette = sns.cubehelix_palette(
    n_colors=5, start=0, rot=0.4, gamma=1.0,
    hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=False)
ax = sns.boxplot(x="datasetName", y="runTime", hue="method", palette=cubehelixPalette,
                 data=runtimesDf, showmeans=True,
                 order=['COMPAS', 'Phishing', 'Students-Performance',
                        'German-Credit', 'Adult', 'Credit-Card-Default',
                        'OnlineNewsPopularity', 'Spambase'],
                 meanprops={"marker": "o", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "7"})
plt.xticks(rotation=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
ax.set_ylabel(r"CPU runtime [in $s$]")
ax.set_yticks(np.arange(-1, 4))
ax.set_yticklabels(10.0**np.arange(-1, 4))
minor_yticks = np.log10(
    np.concatenate((np.arange(1, 10) * 0.1,
                    np.arange(1, 10) * 1,
                    np.arange(1, 10) * 10,
                    np.arange(1, 10) * 100)).astype(np.float))
ax.set_yticks(minor_yticks, minor=True)
x_axis = ax.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)
fig.tight_layout()
plt.savefig('output/figs/runtimeBoxplot.pdf', bbox_inches='tight')
