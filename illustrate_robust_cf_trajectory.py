"""
Plot the trajectory of robust counterfactuals in two dimensions for a single
initial observation with varying robustness target (1-α).
"""
# Import packages
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.CfSimulator import CfSimulator
from src.result_analysis import re_encode_categorical_features

# Set parameters of matplotlib to avoid Type 3 fonts as per NeurIPS guidelines
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["font.family"] = "Times New Roman"


def get_cf_trajectory_as_array(cf, oneHotEncodedFeatures, paramList,
                               methodSensParam, cfDf):
    cfTrajectory = []
    # Append the initial point
    x = np.fromstring(cf[1: -1], dtype=float, sep=' ')
    x = re_encode_categorical_features(x, oneHotEncodedFeatures)
    cfTrajectory.append(x)
    # Append all the CF with increasing alpha
    for alpha in paramList:
        alphaDf = cfDf[cfDf[methodSensParam] == alpha]
        x = np.fromstring(alphaDf['x_cf'].values[0]
                          [1: -1], dtype=float, sep=' ')
        x = re_encode_categorical_features(x, oneHotEncodedFeatures)
        cfTrajectory.append(x - cfTrajectory[0])
    cfTrajArray = np.array(cfTrajectory)
    return cfTrajArray


# ------ Read simulations results ------
METHOD = 'direct-saa'  # 'direct-saa'
METHOD_SENS_PARAM = 'alpha'  # 'alpha'
SIM_INDEX = 1
# Import result files and concatenate them into a single Dataframe
# Datasets available:
# ['Adult', 'COMPAS', 'Credit-Card-Default', 'German-Credit',
# 'OnlineNewsPopularity', 'Phishing', 'Spambase', 'Students-Performance']
datasetList = ['Adult', 'COMPAS', 'Credit-Card-Default', 'German-Credit']
for datasetName in datasetList:
    # Specify path to results csv files
    projectPath = 'C:/Code/robust-counterfactuals/'
    experimentFolder = 'robustTrainingCounterfactuals'
    subFolder = None
    if subFolder:
        subPath = projectPath+'/output/' + experimentFolder + '/' + subFolder+'/'
    else:
        subPath = projectPath+'/output/' + experimentFolder + '/'
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    resultsDf = pd.concat(map(pd.read_csv, filepaths))
    methodDf = resultsDf[resultsDf['method'] == METHOD]

    # Set SIM_INDEX and filter dataframe
    simInstanceDf = methodDf[methodDf['simIndex'] == SIM_INDEX]
    nbCfs = len(np.unique(simInstanceDf['x_init']))
    initCfs = []
    for cf in np.unique(simInstanceDf['x_init']):
        initCfs.append(np.fromstring(cf[1: -1], dtype=float, sep=' '))

    if METHOD_SENS_PARAM == 'alpha':
        paramList = -np.sort(-pd.unique(methodDf[METHOD_SENS_PARAM]))
    else:
        paramList = np.sort(pd.unique(methodDf[METHOD_SENS_PARAM]))
    yLabels = [r'$x_{n+1}$']
    for a in paramList:
        yLabels.append(a)

    cfInits = np.unique(simInstanceDf['x_init'])
    nbCfs = len(cfInits)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6),
                            gridspec_kw={'width_ratios': [1, 1.2]})
    newcmp = LinearSegmentedColormap.from_list(
        "", ["indianred", "white", "royalblue"])
    # Create a CfSimulator object to track the data and counterfactuals
    cfSimulator = CfSimulator(
        datasetName, 1, 0, testSplitFactor=0.0, verbose=True)
    oneHotEncodedFeatures = cfSimulator.reader.oneHotEncoding
    for i in range(2):
        for j in range(2):
            cfIndex = i*2+j
            cf = cfInits[cfIndex]
            # Filter dataframe to this single cf
            cfDf = methodDf[methodDf['x_init'] == cf]
            cfTrajArray = get_cf_trajectory_as_array(cf, oneHotEncodedFeatures, paramList,
                                                     METHOD_SENS_PARAM, cfDf)
            # Plot cf heatmaps
            hm = sns.heatmap(cfTrajArray, linewidths=0, cmap=newcmp, vmin=-1, vmax=1,
                             xticklabels=(i == 1), cbar=(j == 1), ax=axs[i, j])
            hm.invert_yaxis()
            axs[i, j].set_yticklabels(yLabels, rotation=0)
            if j == 0:
                axs[i, j].set_ylabel('Robustness level ($1-\\alpha$)')
            axs[i, j].set_title(r'Counterfactual $i={}$'.format(cfIndex))
            axs[i, j].hlines([0, 1], *axs[i, j].get_xlim(), colors='black',
                             linestyles='dashed', linewidths=1)
    axs[1, 0].set_xticklabels(
        cfSimulator.reader.featureNames, rotation=45, ha='right')
    axs[1, 1].set_xticklabels(
        cfSimulator.reader.featureNames, rotation=45, ha='right')
    plt.savefig('output/figs/counterfactual_trajectory_illustration_'+datasetName+'.pdf',
                bbox_inches='tight')

    # --- Minimal figure for main body ---
    if datasetName == 'German-Credit':
        fig, ax = plt.subplots(figsize=(5, 3))
        j = 0
        cf = cfInits[0]
        # Filter dataframe to this single cf
        cfDf = methodDf[methodDf['x_init'] == cf]
        cfTrajArray = get_cf_trajectory_as_array(cf, oneHotEncodedFeatures, paramList,
                                                 METHOD_SENS_PARAM, cfDf)
        # Plot cf heatmaps
        hm = sns.heatmap(cfTrajArray, linewidths=0, cmap=newcmp, vmin=-1, vmax=1,
                         xticklabels=(i == 1), cbar=(j == 1), ax=ax)
        hm.invert_yaxis()
        ax.set_yticklabels(yLabels, rotation=0)
        ax.hlines([0, 1], *ax.get_xlim(), colors='black',
                  linestyles='dashed', linewidths=1)
        featureNames = cfSimulator.reader.featureNames
        xTicksFeatureNames = [str(featureNames[i])
                              for i in range(len(featureNames))]
        for i in range(len(xTicksFeatureNames)):
            name = xTicksFeatureNames[i]
            xTicksFeatureNames[i] = (
                name[:6] + '..') if len(name) > 6 else name
        ind = np.arange(len(featureNames))+0.5
        plt.setp(ax, xticks=ind, xticklabels=(xTicksFeatureNames))
        plt.savefig('output/figs/mini_cf_traj_'+datasetName+'.pdf',
                    bbox_inches='tight')
