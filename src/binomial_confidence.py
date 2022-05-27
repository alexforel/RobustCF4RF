import numpy as np
from scipy.stats import norm, binom


def p_star_threshold(m_n, alpha):
    """
    Returns the tree success probability such that the classification
    score of a random forest with m_n trees follows:
    Prob(h(x) >= 1/2) = 1 - F_Bin(m_n,p)(mn/2).
    The random forest class prediction is seen as a binomial distribution
    where each tree follows a Bernoulli(p) random variable.
    """
    assert(alpha <= 1)
    if alpha <= 0.5:
        xMin = 0.5
    else:
        xMin = 0.0
    pVector = np.linspace(xMin, 1, 10000)
    p_star = next((p for p in pVector if (
            binom.cdf(m_n/2, m_n, p) <= alpha)), None)
    return [p_star, p_star]


def robust_p_star_threshold(m_n, alpha, beta):
    """ Robust estimation of success rate p with Agresti-Coull conf. interval """
    assert(alpha <= 1)
    pVector = np.linspace(0, 1, 20000)
    # A conservative estimate of p using Agresti-Coull interval
    # Note the (2*) factor
    robustPVector = [robust_agresti_coull_p_estimate(m_n, p, beta=2*beta)
                     for p in pVector]
    p_idx = next((i for i in range(len(robustPVector)) if (
            binom.cdf(m_n/2, m_n, robustPVector[i]) <= alpha)), None)
    p_star = pVector[p_idx]
    return [p_star, p_star]


def p_agresti_coull(T, p):
    """
    Returns the Agresti-Coull success rate for binomial distribution with
    T observations and empirical sucess rate p.
    """
    return (T*p+2)/(T+4)


def robust_agresti_coull_p_estimate(T, p, beta=0.05):
    """
    Returns the lower bound of the C(α) confidence interval around p_ac.
    """
    p_ac = p_agresti_coull(T, p)
    return binomial_ci_agresti_coull(T, p_ac, beta)[0]


def binomial_ci_agresti_coull(T, p_ac, beta):
    """
    Returns the upper and lower bounds of the
    Agresti-Coull confidence interval of the success rate estimate of
    a binomial distribution.
    """
    low = p_ac - norm.ppf(1-beta/2) * np.sqrt(p_ac*(1-p_ac)/T)
    upp = p_ac + norm.ppf(1-beta/2) * np.sqrt(p_ac*(1-p_ac)/T)
    return [low, upp]
