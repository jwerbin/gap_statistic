'''
Created: 2020-06-25
Author: Jeffrey L. Werbin
Licence: MIT

This module contains methods for calculating the Gap statistic for determining optimal clustering.
Tibshirani et al. (2001)
https://web.stanford.edu/~hastie/Papers/gap.pdf

This module was extracted from code used in the paper
"Non-invasive profiling of advanced prostate cancer via multi-parametric liquid biopsy and radiomic analysis"
Authors: Gareth Morrison, Jonathan Buckley, Dejerianne Ostrow, Bino Varhese,
Steven Yong Cen, Jeffrey Werbin, Nolan Ericson, Alexander Cunha, Yi-Tsung Lu,
Thaddeus George, Jeffrey Smith, David Quinn, Vinay Duddalwar, Timothy Triche,
Amir Goldkorn
Molecular Pathology, Diagnostics, and Therapeutics 2022
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.base import clone
from typing import Callable, Tuple


def calculate_cluster_D(data: pd.DataFrame) -> np.ndarray:
    # cluster label column should also add 0 to the distance
    # Using the square form here double counts the pairs. I think that is what the Gap stat paper calls for
    # If not it will be off by a factor of 2
    dist = squareform(pdist(data.to_numpy(dtype=np.float64)))
    return np.sum(dist)


def calculate_W(data: pd.DataFrame) -> np.ndarray:
    # Note data is assumed to already have been clustered and the cluster id is stored in the 'cluster_id' column
    cluster_ids = np.unique(data['cluster_id'])
    w_r = []
    for curr_id in cluster_ids:
        cluster = data.loc[data['cluster_id'] == curr_id, :]
        w_r.append(calculate_cluster_D(cluster) / (2 * cluster.shape[0]))
    return np.sum(w_r)


def random_unif_sampling(data: pd.DataFrame or np.ndarray, original_cluster_labels) -> pd.DataFrame:
    # As described in the above paper the simplest approach is to randomly sample a value for each feature
    # from unif[feature_min, feature_max]
    shape = data.shape
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=1)
    return pd.DataFrame(np.hstack([np.random.uniform(mi, ma, size=(shape[0], 1)) for mi, ma in zip(mins, maxs)]))


def hyperbox_sampling(data: pd.DataFrame, original_cluster_labels) -> pd.DataFrame:
    # As described in Tibshirani.
    # select points randomly in the plane of the two principal components and then project back into real space
    # Todo modify to allow for n components
    transformer = PCA(n_components=2)
    projected = transformer.fit_transform(data)
    projected_rand = random_unif_sampling(projected)
    return pd.DataFrame(transformer.inverse_transform(projected_rand))


def abc_sampling(data: pd.DataFrame, original_cluster_labels) -> pd.DataFrame:
    # A modification of hyperbox sampling from Tibshirani by SAS
    # https://documentation.sas.com/?docsetId=casstat&docsetTarget=casstat_kclus_details05.htm&docsetVersion=8.4&locale=en
    # performs hyperbox sampling around each cluster
    hyperbox_samples = [hyperbox_sampling(data.loc[original_cluster_labels == clust, :])
                        for clust in np.unique(original_cluster_labels)]
    return pd.concat(hyperbox_samples, axis=0, ignore_index=True)


def calculate_reference_W(data: pd.DataFrame, clusterer,
                          original_cluster_labels: np.ndarray,
                          resampling_method: Callable[[pd.DataFrame, np.ndarray], pd.DataFrame] = random_unif_sampling):
    random_data = resampling_method(data, original_cluster_labels)
    cluster_labels = clusterer.fit_predict(random_data)
    random_data['cluster_id'] = cluster_labels
    return calculate_W(random_data)


def gap_n(data: pd.DataFrame, clusterer, n_estimates) -> Tuple[float, float, np.ndarray]:
    """
    Calculates the gap statistic
    inputs:
      data,        i samples x j feature data set stored as a pandas dataframe

      clusterer,   an object that can cluster the data. All parameters should already be set
                   and must have a fit_predict method

      n_estimates, is the number of random sampling used to estimate the reference distribution

    outputs:
      gap,            the value of the gap statistic

      cluster_labels, the cluster_labels for the data
    """

    cluster_labels = clusterer.fit_predict(data)
    clusterer_copy = clone(clusterer)
    data_copy = data.copy()
    data_copy['cluster_id'] = cluster_labels
    sample_log_W = np.log(calculate_W(data_copy))

    # The paper says to take expectation of the logW not log(Expectation(W)) pleas confirm that this is correct.
    reference_log_W = [np.log(calculate_reference_W(data, clusterer_copy, cluster_labels)) for _ in range(n_estimates)]
    mean_reference_log_W = np.mean(reference_log_W)
    std_reference_log_W = float(np.std(reference_log_W))

    gap = float(mean_reference_log_W - sample_log_W)
    return gap, std_reference_log_W, cluster_labels


def calculate_optimal_clusters_by_gap(data: pd.DataFrame, clusterer_objs: dict, n_estimations=5,
                                      resampling_method: Callable = hyperbox_sampling):
    """
     Clusters data using KMeans clustering
     Chooses the number of clusters by finding the maximum of the gap statistic

     inputs:
        data,           i samples x j features dataframe with no extraneous columns (e.g. 'dataset' column)
        clusterer_objs, a dict of clustering object (parameters preset) where the key is the number of clusters
        n_estimations,  sets number of data samplings used to estimate the reference distribution

     outputs:
        cluster_labels, for data from k clusters that maximize the gap statistic
        k,              the optimal number of clusters
        gap_scores,     a 3 x n_clusterer array where gap_scores[0,:] is the number of clusters,
                        gap_scores[1,:] is the gap score, gap_scores[2,:] is the std dev of gap_scores.
    """

    gap_scores = []
    gap_std = []
    n_clusters = []

    cluster_labels = {}
    for n_clust, clusterer in clusterer_objs.items():
        print(f'Work on {n_clust}')
        gap, g_std, labels = gap_n(data, clusterer, n_estimates=n_estimations)
        gap_scores.append(gap)
        gap_std.append(g_std)
        n_clusters.append(n_clust)
        cluster_labels[n_clust] = labels

    # Best K is the smallest k where gap(k) > gap(k+1) - sd(k+1)
    # where sd(k) = std( logWref(k)) * sqrt(1+1/n_estimations)
    # using argmax of bools
    # It returns first value that meets the criteria
    # If nothing meets the criteria it will return 0.
    sdk = np.array(gap_std[1:]) * np.sqrt(1.0 + 1.0 / n_estimations)
    criteria = -1*np.diff(gap_scores) + sdk >= 0
    best_idx = int(np.argmax(criteria))

    gap_scores = np.vstack((np.atleast_2d(n_clusters), np.atleast_2d(gap_scores), np.atleast_2d(gap_std)))

    if best_idx == 0 and not criteria[0]:
        # No cluster met the criteria for best k. Return num_clusters = 1
        return np.zeros(cluster_labels[n_clusters[best_idx]].shape), 1, gap_scores

    return cluster_labels[n_clusters[best_idx]], n_clusters[best_idx], gap_scores
