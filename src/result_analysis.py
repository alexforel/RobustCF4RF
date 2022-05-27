# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.stats import t
from scipy.stats import sem
from ocean.dataProcessing import DatasetReader
from src.local_outlier_factor import l1_norm_with_scaling_factor


def read_full_results_df(datasetName, subPath):
    # Calculate cost and validity with naive constraint: h(x) ≥ 1/2
    resultFolderPath = subPath+datasetName+'/'
    filepaths = [resultFolderPath+"/"
                 + f for f in os.listdir(resultFolderPath) if f.endswith('.csv')]
    fullResultsDf = pd.concat(map(pd.read_csv, filepaths))
    return fullResultsDf


def read_cost_and_validity_from_results_df(resultsDf, methoSensParameter,
                                           alphaList, nbAlphas, m_n_list,
                                           nbMns, nbSimulations, datasetName):
    # -- Read feature types from data file --
    featuresType = get_feature_types(datasetName)
    # -- Compute cost and validity --
    cfCost = np.zeros((nbAlphas, nbMns, nbSimulations))
    cfValidity = np.zeros((nbAlphas, nbMns, nbSimulations))
    for aIndex in range(nbAlphas):
        alphaDf = resultsDf[resultsDf[methoSensParameter] == alphaList[aIndex]]
        for mIndex in range(nbMns):
            m_nDf = alphaDf[alphaDf['m_n'] == m_n_list[mIndex]]
            for x in range(nbSimulations):
                # Read counterfactual cost according to l2 norm
                x_init = np.fromstring(
                    m_nDf['x_init'].values[x][1:-1], dtype=float, sep=' ')
                x_cf = np.fromstring(
                    m_nDf['x_cf'].values[x][1:-1], dtype=float, sep=' ')
                cfCost[aIndex, mIndex, x] = compute_counterfactual_cost(
                    x_init, x_cf, featuresType)
            # Read counterfactual validity
            cfValidity[aIndex, mIndex, :] = (
                (1 - np.array(m_nDf['y_init'])) == np.array(m_nDf['y_test'])).astype(float)
    return cfValidity, cfCost


def read_naive_costs(naiveDf, nbSimulations, datasetName):
    # -- Read feature types from data file --
    featuresType = get_feature_types(datasetName)
    # -- Compute cost and validity --
    naiveCosts = np.zeros((nbSimulations))
    for x in range(nbSimulations):
        # Read counterfactual cost according to l2 norm
        x_init = np.fromstring(
            naiveDf['x_init'].values[x][1:-1], dtype=float, sep=' ')
        x_cf = np.fromstring(
            naiveDf['x_cf'].values[x][1:-1], dtype=float, sep=' ')
        naiveCosts[x] = compute_counterfactual_cost(
            x_init, x_cf, featuresType)
    return naiveCosts


def get_feature_types(datasetName):
    dirname = os.path.dirname(os.path.dirname(__file__))
    filename = dirname+'/datasets/'+datasetName+'.csv'
    return DatasetReader(filename).featuresType


def get_all_feature_changes(inputDf, oneHotEncodedFeatures):
    nbSimulations = len(inputDf)
    featureChanges = []
    for x in range(nbSimulations):
        # Read counterfactual cost according to l2 norm
        x_init = np.fromstring(
            inputDf['x_init'].values[x][1:-1], dtype=float, sep=' ')
        x_cf = np.fromstring(
            inputDf['x_cf'].values[x][1:-1], dtype=float, sep=' ')
        # Encode back to categorical features
        x_init = re_encode_categorical_features(x_init, oneHotEncodedFeatures)
        x_cf = re_encode_categorical_features(x_cf, oneHotEncodedFeatures)
        featureChanges.append(x_cf - x_init)
    return featureChanges


def re_encode_categorical_features_of_all_samples(xArray, oneHotEncodedFeatures):
    xList = []
    for i in range(len(xArray)):
        xList.append(re_encode_categorical_features(
            xArray[i], oneHotEncodedFeatures))
    return np.array(xList)


def re_encode_categorical_features(x, oneHotEncodedFeatures):
    # Encode the value of categorical features to new columns
    for f in oneHotEncodedFeatures:
        colIndices = oneHotEncodedFeatures[f]
        nbPossibleValues = len(colIndices)
        for i in range(nbPossibleValues):
            if x[colIndices[i]] == 1.0:
                # Add a feature at the end
                x = np.r_[x, i / (nbPossibleValues-1)]
    # Delete all columns corresponding to one-hot encodded features
    colsToDelete = list(oneHotEncodedFeatures.values())
    colsToDelete = [item for sublist in colsToDelete for item in sublist]
    x = np.delete(x, colsToDelete, axis=0)
    return x


def compute_counterfactual_cost(x_init, x_cf, featuresType):
    cost = l1_norm_with_scaling_factor(
        x_init, x_cf, featuresType, len(featuresType))
    return cost


def compute_kpis_and_confidence_intervals(cfValidity, cfCost, alphaList,
                                          nbMns, nbSimulations):
    """ Compute key performance indicators and confidence intervals of mean. """
    nbAlphas = len(alphaList)
    avgCost = np.zeros((nbAlphas, nbMns))
    costConf = np.zeros((nbAlphas, nbMns))
    avgVal = np.zeros((nbAlphas, nbMns))
    valConf = np.zeros((2, nbAlphas, nbMns))
    for a in range(nbAlphas):
        alpha = alphaList[a]
        for m in range(nbMns):
            # # Cost is relative to the cost of the counterfactual for α=50%
            # costReference = cfCost[0, m, :]
            costVector = cfCost[a, m, :]  # / costReference
            avgCost[a, m] = np.mean(costVector)
            if sem(costVector) > 0:
                costConf[a, m] = t.interval(0.95, len(costVector)-1,
                                            loc=np.mean(costVector),
                                            scale=sem(costVector))[1]
            else:
                costConf[a, m] = 0.0
            # Validity
            avgVal[a, m] = np.mean(cfValidity[a, m, :])
            # Hypothesis test: is target validity reached?
            successCount = int(sum(cfValidity[a, m, :]))
            robustTest = binomtest(successCount, nbSimulations,
                                   p=(1-alpha/100), alternative='two-sided')
            valConf[:, a, m] = [robustTest.proportion_ci().low,
                                robustTest.proportion_ci().high]
    return avgVal, valConf, avgCost, costConf


def export_results_csv(method, datasetName, avgVal, valConf, avgCost, costConf,
                       alphaList, experimentFolder, projectPath):
    dfToExport = pd.DataFrame({'confLevel': 1-alphaList})
    dfToExport[method+'-cost'] = avgCost[:, 0]
    dfToExport[method+'-cost-lb'] = avgCost[:, 0] - costConf[:, 0]
    dfToExport[method+'-cost-ub'] = avgCost[:, 0] + costConf[:, 0]
    dfToExport[method+'-valid'] = avgVal[:, 0]
    dfToExport[method+'-valid-lb'] = valConf[0, :, 0]
    dfToExport[method+'-valid-ub'] = valConf[1, :, 0]
    if experimentFolder == 'robustTrainingCounterfactuals':
        folderName = os.path.join(
            projectPath, "output", "csv", "algRobResults")
    elif experimentFolder == 'robustCounterfactuals':
        folderName = os.path.join(
            projectPath, "output", "csv", "robustResults")
    os.makedirs(folderName, exist_ok=True)
    dfToExport.to_csv(folderName+'/'+datasetName
                      + '_method='+method+'.csv', index=False)
    return None


def plot_robust_counterfactuals_sensitivity(datasetName, avgVal, valConf,
                                            avgCost, costConf, alphaList,
                                            m_n_list, projectPath,
                                            experimentFolder):
    targetValidity = 1-alphaList
    fig, axs = plt.subplots(2, figsize=(10, 10))
    # Cost plot
    fig.suptitle(datasetName)
    axs[0].set_title("Cost as a function of 1-alpha +/- one stdev")
    for m in range(len(m_n_list)):
        axs[0].plot(targetValidity, avgCost[:, m], marker=".")
        axs[0].fill_between(targetValidity, avgCost[:, m]+costConf[:, m],
                            avgCost[:, m]-costConf[:, m], alpha=0.1)
    axs[0].set_xlabel("1-alpha")
    axs[0].set_ylabel("l1-norm cost")
    axs[0].set_xticks(np.arange(0.5, 1.01, step=0.1))
    # Validity plot
    axs[1].plot(targetValidity, targetValidity, 'black',
                linestyle='dashed', label='_nolegend_')
    for m in range(len(m_n_list)):
        axs[1].plot(targetValidity, avgVal[:, m], marker=".")
        axs[1].fill_between(targetValidity, valConf[0, :, m],
                            valConf[1, :, m], alpha=0.1)
    axs[1].set_title("CF validity as a function of 1-alpha")
    axs[1].set_xlabel("1-alpha")
    axs[1].set_ylabel("Validity")
    # axs[1].set_xticks(np.arange(0.5, 1.01, step=0.1))
    if len(m_n_list) > 1:
        axs[1].legend(m_n_list)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    folderName = projectPath+'/output/figs/' + experimentFolder
    os.makedirs(folderName, exist_ok=True)
    fig.savefig(folderName + '/cost_robust_cf_'+datasetName+'.png')
    return None


def plot_robust_constraint_relax_histogram(datasetName, resultsDf, alphaList):
    fig, axs = plt.subplots(len(alphaList), figsize=(10, 10), sharex=True)
    fig.suptitle(datasetName+": histogram of robust constraint relaxation")
    for aIndex in range(len(alphaList)):
        alphaDf = resultsDf[resultsDf['alpha'] == alphaList[aIndex]]
        relaxVector = alphaDf['robustConsRelaxation']
        axs[aIndex].hist(relaxVector)
        axs[aIndex].set_title("alpha={}".format(alphaList[aIndex]))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return None
