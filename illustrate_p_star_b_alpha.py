"""
Plot the evolution of p*_{N, alpha} for varying N and alpha.
"""
from src.binomial_confidence import p_star_threshold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import binom

####################################################
# --- Forest probability and tree probability --- #
####################################################


def forest_probability(N, p):
    forestProba = 1 - binom.cdf(N/2, N, p)
    return forestProba


treeProba = np.linspace(0, 1, 100)
forestProba5 = [forest_probability(5, p) for p in treeProba]
forestProba100 = [forest_probability(100, p) for p in treeProba]
plt.figure()
plt.plot(treeProba, forestProba5, label='N={}'.format(5))
plt.plot(treeProba, forestProba100, label='N={}'.format(100))
plt.plot(treeProba, treeProba, '--k')
plt.legend()
plt.savefig('output/figs/forest_tree_proba.png')

##############################
# --- Varying α, fixed N --- #
##############################


def get_1_minus_alpha_and_p_star_vectors(B):
    """Find the minimum p that satisfies the robustness criterion"""
    alphaVect = np.linspace(start=0.5, stop=0.01, num=50)
    p_star = [p_star_threshold(B, alpha)[0] for alpha in alphaVect]
    return [1-alphaVect, p_star]


def plot_min_p_with_guaranteed_target_class(B):
    [oneMinusAlphaVect,
        minThreshVect] = get_1_minus_alpha_and_p_star_vectors(B)
    plt.plot(oneMinusAlphaVect, minThreshVect, label='B={}'.format(B))
    plt.legend()
    plt.title("Min tree success rate p* for target class with (1-α) guarantee")
    plt.xlabel("Robustness (1-α)")
    plt.ylabel("Min tree success rate p")


plt.figure()
plot_min_p_with_guaranteed_target_class(50)
plot_min_p_with_guaranteed_target_class(100)
plot_min_p_with_guaranteed_target_class(500)
plt.plot(np.linspace(0.50, 0.675, 20), np.linspace(0.50, 0.675, 20), '--k')

plt.savefig('output/figs/min_p_with_guaranteed_target_class.png')

# Export figure data to csv files
data = np.array((get_1_minus_alpha_and_p_star_vectors(50)[0],
                get_1_minus_alpha_and_p_star_vectors(50)[1],
                get_1_minus_alpha_and_p_star_vectors(100)[1],
                get_1_minus_alpha_and_p_star_vectors(200)[1]))

pStarDataframe = pd.DataFrame(data).T
pStarDataframe.rename(columns={
                      0: '1MinusAlpha', 1: 'pstart_50', 2: 'pstart_100', 3: 'pstart_200'},
                      inplace=True)
folderName = 'C:\\output\\csv\\'
os.makedirs(folderName, exist_ok=True)
pStarDataframe.to_csv(folderName+'p_star_sensitivity.csv',
                      index=False)

##############################
# --- Varying N, fixed α --- #
##############################
nbTreesVector = np.linspace(start=1, stop=100, num=100)
p_star_vector_alpha1 = [p_star_threshold(N, 0.1)[0] for N in nbTreesVector]
p_star_vector_alpha2 = [p_star_threshold(N, 0.25)[0] for N in nbTreesVector]
p_star_vector_alpha3 = [p_star_threshold(N, 0.5)[0] for N in nbTreesVector]
p_star_vector_alpha4 = [p_star_threshold(N, 0.75)[0] for N in nbTreesVector]

plt.plot(nbTreesVector, p_star_vector_alpha1, label='α={}'.format(0.1))
plt.plot(nbTreesVector, p_star_vector_alpha2, label='α={}'.format(0.25))
plt.plot(nbTreesVector, p_star_vector_alpha3, label='α={}'.format(0.5))
plt.plot(nbTreesVector, p_star_vector_alpha4, label='α={}'.format(0.75))
plt.legend()
plt.xlabel("Nb trees")
plt.ylabel("p_star")

data = np.array((nbTreesVector,
                p_star_vector_alpha1,
                p_star_vector_alpha2,
                p_star_vector_alpha3,
                p_star_vector_alpha4))

pStarDataframe = pd.DataFrame(data).T
pStarDataframe.rename(columns={
                      0: 'N', 1: 'pstar_alpha_0p1', 2: 'pstar_alpha_0p25',
                      3: 'pstar_alpha_0p8', 4: 'pstar_alpha_0p75'},
                      inplace=True)
pStarDataframe.to_csv(folderName+'p_star_sensitivity_N.csv',
                      index=False)
