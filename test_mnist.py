#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package
@brief
@author Dan SASAI (RDD)
"""

import scipy as sp
from sklearn.datasets import fetch_mldata
from bhtsne_interface import BHTSNE
import matplotlib.pyplot as plot


if __name__ == "__main__":
    mnist = fetch_mldata("MNIST original", data_home="./")
    mnist_data = mnist["data"]
    mnist_labels = mnist["target"]

    model = BHTSNE(n_components=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=1000)
    mnist_tsne = model.fit_transform(mnist_data)

    xmin = mnist_tsne[:,0].min()
    xmax = mnist_tsne[:,0].max()
    ymin = mnist_tsne[:,1].min()
    ymax = mnist_tsne[:,1].max()

    plot.figure( figsize=(16,12) )
    for _y,_label in zip(mnist_tsne[::20],mnist_labels[::20]):
        plot.text(_y[0], _y[1], _label)
    plot.axis([xmin,xmax,ymin,ymax])
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    plot.title("MNIST t-SNE visualization")
    plot.savefig("mnist_tsne.png")
