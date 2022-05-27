"""
Measure the feature importance of the random forest classifiers
and compare to the average feature changed to
generate robust counterfactuals.
"""
# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import custom functions
from src.CfSimulator import CfSimulator
from src.feature_importance import feature_permutation_importance
from src.result_analysis import get_all_feature_changes, read_full_results_df
from src.result_analysis import re_encode_categorical_features_of_all_samples

# Set parameters of matplotlib to avoid Type 3 fonts as per NeurIPS guidelines
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["font.family"] = "Times New Roman"

# Specify subfolder for results files
projectPath = 'C:/Code/robust-counterfactuals/'
experimentFolder = 'robustTrainingCounterfactuals'
subPath = projectPath+'/output/'+experimentFolder+'/'
method = 'direct-saa'
datasetList = ['Students-Performance', 'COMPAS', 'Phishing', 'German-Credit']

featureImportance = dict()
featureChanges = dict()
featureNames = dict()
featurePossibleValues = dict()
for datasetName in datasetList:
    # Create a CfSimulator object to track the data and counterfactuals
    cfSimulator = CfSimulator(datasetName, 1, 0,
                              testSplitFactor=0.0, verbose=True)
    oneHotEncodedFeatures = cfSimulator.reader.oneHotEncoding
    featureNames[datasetName] = cfSimulator.reader.featureNames
    # Re-format one-hot encoded features to categorical
    x = re_encode_categorical_features_of_all_samples(
        cfSimulator.x_train, cfSimulator.reader.oneHotEncoding)
    featurePossibleValues[datasetName] = [
        np.unique(x[:, f]) for f in range(len(featureNames[datasetName]))]
    # --- Permutation feature importance ---
    nbRepeats = 10
    m_n = 100
    maxDepth = 4
    importance = feature_permutation_importance(
        x, cfSimulator.y_train, nbRepeats, maxDepth, m_n)
    featureImportance[datasetName] = np.mean(np.hstack(importance), axis=1)
    # --- Counterfactual changes ---
    fullResultsDf = read_full_results_df(datasetName, subPath)
    # Filter to relevant counterfactual generation method
    resultsDf = fullResultsDf[fullResultsDf['method'] == method]
    resultsDf = resultsDf[resultsDf['alpha'] == 0.5]
    changes = get_all_feature_changes(resultsDf, oneHotEncodedFeatures)
    absoluteChanges = np.abs(np.vstack(changes))
    featureChanges[datasetName] = np.mean(absoluteChanges, axis=0)

# --- Plot results ---
for datasetName in datasetList:
    # Bar plot with error bars
    ind = np.arange(len(featureNames[datasetName]))
    barWidth = 0.35  # the width of the bars
    fig, axis = plt.subplots(figsize=(6, 3))
    axis.bar(ind - barWidth/2, featureImportance[datasetName]/np.sum(featureImportance[datasetName]),
             barWidth, label='Feature importance')
    axis.bar(ind + barWidth/2, featureChanges[datasetName]/np.sum(featureChanges[datasetName]),
             barWidth, label='Avg. feature changes')
    axis.set_title("Feature importance and feature changes")
    axis.set_ylabel(r"Average [in $\%$]")
    axis.legend()
    xTicksFeatureNames = [str(featureNames[datasetName][i]) for i in ind]
    for i in range(len(xTicksFeatureNames)):
        name = xTicksFeatureNames[i].replace("_", "")
        xTicksFeatureNames[i] = (
            name[:10] + '...') if len(name) > 10 else name
    plt.setp(axis, xticks=ind, xticklabels=(xTicksFeatureNames))
    axis.tick_params(labelrotation=90)
    fig.tight_layout()
    plt.savefig('output/figs/featureImportance_'+datasetName+'.pdf',
                bbox_inches='tight')
    plt.show()

# -- Counterfactual sparsity: Number of features changed --
allDatasets = ['Students-Performance', 'German-Credit', 'Spambase', 'COMPAS',
               'Phishing', 'Credit-Card-Default', 'OnlineNewsPopularity', 'Adult']
alphaList = -np.sort(-pd.unique(fullResultsDf['alpha']))
dfToExport = pd.DataFrame({'confLevel': 1-alphaList})
fig, ax = plt.subplots()
for datasetName in allDatasets:
    # Read results from csv files
    fullResultsDf = read_full_results_df(datasetName, subPath)
    # Filter to relevant counterfactual generation method
    methodDf = fullResultsDf[fullResultsDf['method'] == method]
    cfSimulator = CfSimulator(
        datasetName, 1, 0, testSplitFactor=0.0, verbose=True)
    n = len(cfSimulator.x_train)
    oneHotEncodedFeatures = cfSimulator.reader.oneHotEncoding
    nbFeatureChanged = []
    for alpha in alphaList:
        # Filter to relevant counterfactual generation method
        resultsDf = methodDf[methodDf['alpha'] == alpha]
        featureChanges = get_all_feature_changes(
            resultsDf, oneHotEncodedFeatures)
        allChanges = np.vstack(featureChanges)
        isFeatureChanged = 1 - (allChanges == 0)
        featureChangeInEachCf = np.sum(isFeatureChanged, axis=1)
        nbFeatureChanged.append(
            np.mean(featureChangeInEachCf))
    plt.plot(1-alphaList, nbFeatureChanged, label=datasetName)
    # Add column to dataframe
    dfToExport[datasetName+'-nbFeat'] = nbFeatureChanged
fig.tight_layout()
ax.legend()
plt.show()
# Export to csv file
folderName = os.path.join(projectPath, "output", "csv")
os.makedirs(folderName, exist_ok=True)
dfToExport.to_csv(folderName+'/featureChangeWithAlpha.csv', index=False)
