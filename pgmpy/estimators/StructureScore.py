#!/usr/bin/env python
from math import lgamma, log

import numpy as np
import statsmodels.formula.api as smf
from scipy.special import gammaln
from scipy.stats import multivariate_normal

from pgmpy.estimators import BaseEstimator


class StructureScore(BaseEstimator):
    """
    Abstract base class for structure scoring classes in pgmpy. Use any of the
    derived classes K2Score, BDeuScore, BicScore or AICScore. Scoring classes
    are used to measure how well a model is able to describe the given data
    set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    Reference
    ---------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3
    """

    def __init__(self, data, **kwargs):
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a score to measure how well the given `BayesianNetwork` fits
        to the data set.  (This method relies on the `local_score`-method that
        is implemented in each subclass.)

        Parameters
        ----------
        model: BayesianNetwork instance
            The Bayesian network that is to be scored. Nodes of the BayesianNetwork need to coincide
            with column names of data set.

        Returns
        -------
        score: float
            A number indicating the degree of fit between data and model

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import K2Score
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> K2Score(data).score(BayesianNetwork([['A','B'], ['A','C']]))
        -24242.367348745247
        >>> K2Score(data).score(BayesianNetwork([['A','B'], ['B','C']]))
        -16273.793897051042
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, list(model.predecessors(node)))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        """A (log) prior distribution over models. Currently unused (= uniform)."""
        return 0

    def structure_prior_ratio(self, operation):
        """Return the log ratio of the prior probabilities for a given proposed change to the DAG.
        Currently unused (=uniform)."""
        return 0


class K2Score(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with Dirichlet priors.
    The K2 score is the result of setting all Dirichlet hyperparameters/pseudo_counts to 1.
    The `score`-method measures how well a model is able to describe the given data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 806)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(K2Score, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=float)

        # Compute log(gamma(counts + 1))
        gammaln(counts + 1, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        # Adjustments when using reindex=False as it drops columns of 0 state counts
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1]) * var_cardinality * gammaln(1)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(
            var_cardinality
        )

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(var_cardinality)
        )

        return score


class BDeuScore(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with Dirichlet priors.
    The BDeu score is the result of setting all Dirichlet hyperparameters/pseudo_counts to
    `equivalent_sample_size/variable_cardinality`.
    The `score`-method measures how well a model is able to describe the given data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    equivalent_sample_size: int (default: 10)
        The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters.
        The score is sensitive to this value, runs with different values might be useful.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 806)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super(BDeuScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score


class BDsScore(BDeuScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The BDs score is the result of setting all Dirichlet
    hyperparameters/pseudo_counts to
    `equivalent_sample_size/modified_variable_cardinality` where for the
    modified_variable_cardinality only the number of parent configurations
    where there were observed variable counts are considered.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    equivalent_sample_size: int (default: 10)
        The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet
        hyperparameters.
        The score is sensitive to this value, runs with different values might be useful.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Scutari, Marco. An Empirical-Bayes Score for Discrete Bayesian Networks.
    Journal of Machine Learning Research, 2016, pp. 438–48

    """

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        super(BDsScore, self).__init__(data, equivalent_sample_size, **kwargs)

    def structure_prior_ratio(self, operation):
        """Return the log ratio of the prior probabilities for a given proposed change to
        the DAG.
        """
        if operation == "+":
            return -log(2.0)
        if operation == "-":
            return log(2.0)
        return 0

    def structure_prior(self, model):
        """
        Implements the marginal uniform prior for the graph structure where each arc
        is independent with the probability of an arc for any two nodes in either direction
        is 1/4 and the probability of no arc between any two nodes is 1/2."""
        nedges = float(len(model.edges()))
        nnodes = float(len(model.nodes()))
        possible_edges = nnodes * (nnodes - 1) / 2.0
        score = -(nedges + possible_edges) * log(2.0)
        return score

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / state_counts.shape[1]
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + state_counts.shape[1] * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score


class BicScore(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The BIC/MDL score ("Bayesian Information Criterion",
    also "Minimal Descriptive Length") is a log-likelihood score with an
    additional penalty for network complexity, to avoid overfitting.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 802)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(BicScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score


class BicScoreGauss(StructureScore):
    def __init__(self, data, **kwargs):
        super(BicScoreGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(
                formula=f"{variable} ~ {' + '.join(parents)}", data=self.data
            ).fit()
        # Adding +2 to model df to compute the likelihood df.
        return glm_model.llf - (
            ((glm_model.df_model + 2) / 2) * np.log(self.data.shape[0])
        )


class AICScore(StructureScore):
    """
    Class for Bayesian structure scoring for BayesianNetworks with
    Dirichlet priors.  The AIC score ("Akaike Information Criterion) is a log-likelihood score with an
    additional penalty for network complexity, to avoid overfitting.  The
    `score`-method measures how well a model is able to describe the given
    data set.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    References
    ---------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3.4-18.3.6 (esp. page 802)
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
    http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    def __init__(self, data, **kwargs):
        super(AICScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= num_parents_states * (var_cardinality - 1)

        return score


class AICScoreGauss(StructureScore):
    def __init__(self, data, **kwargs):
        super(AICScoreGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(
                formula=f"{variable} ~ {' + '.join(parents)}", data=self.data
            ).fit()
        # Adding +2 to model df to compute the likelihood df.
        return glm_model.llf - (glm_model.df_model + 2)


class CondGauss(StructureScore):
    def __init__(self, data, **kwargs):
        super(CondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        df = self.data.loc[:, [variable] + parents]

        # If variable is continuous, the probability is computed as:
        # P(C1 | C2, D) = p(C1, C2 | D) / p(C2 | D)
        if self.dtypes[variable] == "N":
            C1 = variable
            C2 = [var for var in parents if self.dtypes[var] == "N"]
            D = list(set(parents) - set(C2))

            if len(D) == 0:
                if len(C2) == 0:
                    return np.log(
                        multivariate_normal.pdf(
                            x=df, mean=df.mean(axis=1), cov=df.cov()
                        )
                    ).sum()
                else:
                    return np.log(
                        multivariate_normal.pdf(
                            x=df, mean=df.mean(axis=1), cov=df.cov()
                        )
                        / multivariate_normal.pdf(
                            x=df.loc[:, C2],
                            mean=df.loc[:, C2].mean(axis=1),
                            cov=df.loc[:, C2].cov(axis=1),
                        )
                    ).sum()

            else:
                log_like = 0
                for D_states, df_strat in df.groupby(D):
                    df_C1C2 = df.loc[:, [C1] + C2]
                    p_C1C2_D = multivariate_normal.pdf(
                        x=df_C1C2, mean=df_C1C2.mean(axis=1), cov=df_C1C2.cov()
                    )
                    if len(C2) == 0:
                        p_C2_D = 1
                    else:
                        df_C2 = df.loc[:, C2]
                        p_C2_D = multivariate_normal.pdf(
                            x=df_C2, mean=df_C2.mean(axis=1), cov=df_C2.cov()
                        )
                    log_like += np.log(p_C1C2_D / p_C2_D).sum()
                return log_like

        # If variable is discrete, the probability is computed as:
        # P(D1 | C1, C2, D2) = (p(C1, C2 | D1, D2) p(D1, D2)) / (p(C1, C2 | D2) p(D2))
        else:
            all_discrete = [variable] + [
                var for var in parents if self.dtypes[var] in ("C", "O")
            ]
            all_cont = list(set(parents) - set(all_discrete))

            p_all_discrete = (
                df.loc[:, all_discrete].groupby(all_discrete).counts() / df.shape[0]
            )
            if all_discrete == [variable]:
                p_d2 = 1
            else:
                d2 = set(all_discrete) - {variable}
                p_d2 = df.loc[:, d2].groupby(d2).counts() / df.shape[0]

            for z_state, df_strat in df.groupby(all_discrete):
                p_c1_c2 = multivariate_normal(
                    df.loc[:, all_cont],
                    mean=df.loc[:, all_cont].mean(axis=1),
                    cov=df.loc[:, all_cont].cov(),
                )
                p_d1_d2 = p_all_discrete[df_strat]
                df_d2 = df.loc[df.loc[:, d2] == df_strat[1:]]
                p_c1_c2_d2 = multivariate_normal(
                    df_d2.loc[:, all_cont],
                    df_d2.loc[:, all_cont].mean(axis=1),
                    df_d2.loc[:, all_cont].cov(),
                )
                p_d2 = p_d2[d2_state]
