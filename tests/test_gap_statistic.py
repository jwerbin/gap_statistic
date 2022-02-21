"""
Created: 2020-07-02
Author: Jeffrey L. Werbin
Licence: MIT

Tests for the gap_statistics module.

These tests were extracted from code used in the paper
"Non-invasive profiling of advanced prostate cancer via multi-parametric liquid biopsy and radiomic analysis"
Authors: Gareth Morrison, Jonathan Buckley, Dejerianne Ostrow, Bino Varhese,
Steven Yong Cen, Jeffrey Werbin, Nolan Ericson, Alexander Cunha, Yi-Tsung Lu,
Thaddeus George, Jeffrey Smith, David Quinn, Vinay Duddalwar, Timothy Triche,
Amir Goldkorn
Molecular Pathology, Diagnostics, and Therapeutics 2022
"""

import numpy as np
import pandas as pd
from sklearn import cluster
import pytest

from typing import List, Tuple, Union

from gap_statistic import gap_statistic

def make_gaussian_test_data(cluster_info: List[Tuple[np.ndarray, np.ndarray, int]]):
    data_points = []
    for means, stds, num in cluster_info:
        data_points.append(np.random.normal(means, stds, size=num))
    return np.vstack(data_points)


def test_calculate_cluster_D():
    # D is just the sum of all pairwise distances (in both directionss i,j and j,i)
    points = np.array([[0, 0, 0], [-2, 0, 4], [0, 0, 5]])
    d = gap_statistic.calculate_cluster_D(pd.DataFrame(points))
    assert np.isclose(d, 2*(np.sqrt(4+16) + 5 + np.sqrt(4+1)))

    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    d = gap_statistic.calculate_cluster_D(pd.DataFrame(points))
    assert np.isclose(d, 2*(1 + 0 + 1))


def test_calculate_W():
    # W is the sum of all the cluster Ds divided by two * number of points
    points = np.array([[0, 0, 0], [-2, 0, 4], [0, 0, 5]])
    df = pd.DataFrame(np.vstack((points, points + np.atleast_2d([0, 0, 100]))))
    d = gap_statistic.calculate_cluster_D(pd.DataFrame(points))
    with pytest.raises(KeyError):
        gap_statistic.calculate_W(df)

    df['cluster_id'] = [0, 0, 0, 1, 1, 1]
    w = gap_statistic.calculate_W(df)
    assert np.isclose(w, 2 * (d / (2 * 3)))


def test_random_unif_sampling():
    points = np.array([[0, 0, 0], [-2, 0, 4], [0, 0, 5]])
    rand_points = gap_statistic.random_unif_sampling(points, np.zeros((3,)))

    assert points.shape == rand_points.shape


def make_principal_comp_data(size):
    means = np.arrange(size[1])[::-1]
    data = np.random.normal(loc=means, scale=0.1, size=size)
    data[:, :2] = np.random.normal(loc=means[:2], scale=[10.0, 25.0], size=(size[0], 2))
    return data


@pytest.mark.skip(reason='Unsure how to test functionality')
def test_hyperbox_sampling():
    pytest.skip
    # Trying to think of a good way to test this
    points = np.array([[0, 0, 0], [-2, 0, 4], [0, 0, 5]])


@pytest.mark.skip(reason='Unsure how to test functionality')
def test_abc_sampling():
    # Trying to think of a good way to test this
    points = np.array([[0, 0, 0], [-2, 0, 4], [0, 0, 5]])


def make_clustered_data(means: List[np.ndarray], stdevs: List[np.ndarray], num_points: List[int]):
    points = []
    labels = []
    for i, (ms, stds, num) in enumerate(zip(means, stdevs, num_points)):
        points.append(np.random.normal(loc=ms, scale=stds, size=(num, ms.size)))
        labels.append(i * np.ones(num))
    return np.vstack(points), np.hstack(labels)


def test_calculate_reference_W():
    np.random.seed(1) # Set seed to make the test predictable.
    points, cluster_ids = make_clustered_data([np.array([0, 0, 0, 0, 0, 0, 0]), np.array([10, 10, 10, 0, 0, 0, 0])],
                                              [np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.1]),
                                              np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.1])],
                                              [50, 80])
    df = pd.DataFrame(points)
    df_copy = df.copy()
    df_copy['cluster_id'] = cluster_ids
    w = gap_statistic.calculate_W(df_copy)
    clusterer = cluster.KMeans(n_clusters=2)
    w_ref = gap_statistic.calculate_reference_W(df, clusterer, cluster_ids)

    assert w_ref >= w


def test_gap_n():
    np.random.seed(1) # Set seed to make the test predictable.
    points, cluster_ids = make_clustered_data([np.array([0, 0, 0, 0, 0, 0, 0]), np.array([10, 10, 10, 0, 0, 0, 0])],
                                         [np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.1]), np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.1])],
                                         [50, 80])
    df = pd.DataFrame(points)
    clusterer1 = cluster.KMeans(n_clusters=2)
    clusterer2 = cluster.KMeans(n_clusters=5)

    gap1, std1, labels1 = gap_statistic.gap_n(df, clusterer1, 100)
    gap2, std2, labels2 = gap_statistic.gap_n(df, clusterer2, 100)

    assert gap1 > gap2


def test_calculate_optimal_clusters_by_gap():
    # An addtional/improved test to make would be to reproduce the test simulations for the original paper

    np.random.seed(1)  # Set seed to make the test predictable.
    points, expected_labels = make_clustered_data([np.array([0, 0, 0, 0, 0, 0, 0]), np.array([10, 10, 10, 0, 0, 0, 0]),
                                                   np.array([0, 0, 0, 0, 50, 25, 10])],
                                                  [np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                                   np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                                   np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])],
                                                  [50, 80, 50])
    df = pd.DataFrame(points)
    clusterers = {i: cluster.KMeans(n_clusters=i) for i in range(2, 10)}

    labels, n_clust, gap_scores = gap_statistic.calculate_optimal_clusters_by_gap(df, clusterers, n_estimations=10)

    # presumably either group could be labeled 0 or 1
    assert n_clust == 3

    # To confirm correct cluster labels. We Note that the points are ordered by cluster
    # So for 3 clusters there will only be 2 changes in clusterId between adjacent points.
    assert np.sum(np.diff(labels) != 0) == 2

    assert np.max(gap_scores[1, :]) == gap_scores[1, gap_scores[0, :] == n_clust]
