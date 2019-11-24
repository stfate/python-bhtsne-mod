
import os
import unittest
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from bhtsne import tsne
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

PLOTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/plots'

def mean_shift(X):
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels_unique = np.unique(ms.labels_)
    return len(labels_unique), ms.cluster_centers_

class TestTsne(unittest.TestCase):

    def test_iris(self):
        iris = load_iris()
        X = iris.data
        self.assertEqual(mean_shift(X)[0], 2)
        Y = tsne(X)
        plt.scatter(Y[:, 0], Y[:, 1], c=iris.target)

        num_clusters, cluster_centers = mean_shift(Y)
        self.assertTrue(num_clusters > 1)
        self.assertTrue(num_clusters < 4)
        for k in range(num_clusters):
            cluster_center = cluster_centers[k]
            plt.plot(cluster_center[0], cluster_center[1], 'x', markerfacecolor='r',
                     markeredgecolor='r', markersize=16)

        plt.savefig(PLOTS_DIR + '/iris.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_iris_sklearn(self):
        iris = load_iris()
        X = iris.data
        self.assertEqual(mean_shift(X)[0], 2)

        sklearn_tsne = TSNE(learning_rate=100)
        Y = sklearn_tsne.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], c=iris.target)

        num_clusters, cluster_centers = mean_shift(Y)
        for k in range(num_clusters):
            cluster_center = cluster_centers[k]
            plt.plot(cluster_center[0], cluster_center[1], 'x', markerfacecolor='r',
                     markeredgecolor='r', markersize=16)

        plt.savefig(PLOTS_DIR + '/iris_sklearn.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_set_rand_seed(self):
        iris = load_iris()
        X = iris.data
        Y_a = tsne(X, rand_seed=999)
        Y_b = tsne(X, rand_seed=999)

        self.assertEqual(round(Y_a[0][0] / 5), round(Y_b[0][0] / 5))
        self.assertEqual(round(Y_a[0][1] / 5), round(Y_b[0][1] / 5))

        plt.scatter(Y_a[:, 0], Y_a[:, 1], c='b')
        plt.scatter(Y_b[:, 0], Y_b[:, 1], c='r')
        plt.savefig(PLOTS_DIR + '/iris_set_rand_seed.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_without_seed_positions(self):
        iris = load_iris()
        X_a = load_iris().data[:-10]
        X_b = load_iris().data
        Y_a = tsne(X_a, rand_seed=999)
        Y_b = tsne(X_b, rand_seed=999)

        plt.scatter(Y_a[:, 0], Y_a[:, 1], c='b')
        plt.scatter(Y_b[:-10, 0], Y_b[:-10, 1], c='r')
        plt.scatter(Y_b[-10:, 0], Y_b[-10:, 1], c='g')

        plt.savefig(PLOTS_DIR + '/iris_without_seed_positions.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_seed_positions(self):
        iris = load_iris()
        X_a = load_iris().data[:-10]
        X_b = load_iris().data
        Y_a = tsne(X_a, rand_seed=999)
        # Generate random positions for last 10 items
        remainder_positions = np.array([
            [(random.uniform(0, 1) * 0.0001), (random.uniform(0, 1) * 0.0001)]
                for x in range(X_b.shape[0] - Y_a.shape[0])
            ])
        # Append them to previous TSNE output and use as seed_positions in next plot
        seed_positions = np.vstack((Y_a, remainder_positions))
        Y_b = tsne(X_b, seed_positions=seed_positions)

        self.assertEqual(round(Y_a[0][0] / 20), round(Y_b[0][0] / 20))
        self.assertEqual(round(Y_a[0][1] / 20), round(Y_b[0][1] / 20))

        plt.scatter(Y_a[:, 0], Y_a[:, 1], c='b')
        plt.scatter(Y_b[:-10, 0], Y_b[:-10, 1], c='r')
        plt.scatter(Y_b[-10:, 0], Y_b[-10:, 1], c='g')

        plt.savefig(PLOTS_DIR + '/iris_seed_positions.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()
