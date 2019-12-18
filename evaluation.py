# coding:utf-8
from sklearn import datasets
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from time import time
from tsne import bh_sne
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue as PQ
import heapq as hq
from sklearn.decomposition import PCA




FILENAME = 'terr'
VERSION = '_0.3_0.01'

settings = {
    #'skiprows': [0],
    'index_col': 0,
    'sep': ' ',
}

def sample_visualization_v2():
    # take ids of the first 100 points
    ids = pd.read_csv("./dataset/node2vec/terr_0.3_0.01.content", header=None, sep='\t').iloc[:20, 0].values

    VERSIONS = ["terr_0.3_0.01_da-node2vec_{}.emd".format(i) for i in [1, 5, 9] ]

    X = None
    Y = None
    last_vis_data = None
    for version in VERSIONS:
        feature_df = pd.read_csv("./emb/{}.emd".format(version), header=None, **settings).iloc[:, :]
        feature_df = feature_df.loc[ids, :]
        # print(feature_df.iloc[:, :3])
        # draw the first four nodes

        zero_data = np.ones(shape=(len(ids), 1))
        for i in range(0, 4):
            zero_data[i] = i + 2
        labels = pd.DataFrame(zero_data)
        if X:
            X = feature_df.values
        else:
            X += feature_df.values
        if Y:
            Y = labels.values
        else:
            Y += labels.values

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        vis_data = pd.DataFrame(data = principalComponents)

        arrow_data = []
        ax = plt.axes()
        if last_vis_data:
            for i in range(len(vis_data.shape[0])):
                ax.arrow(last_vis_data[i, 0], last_vis_data[i, 1], vis_data[i, 0], vis_data[i, 1])

        last_vis_data = vis_data
        print(vis_data)

        # vis_data = bh_sne(X, random_state=np.random.RandomState(1), perplexity=1)
        vis_x = vis_data.iloc[:, 0]
        vis_y = vis_data.iloc[:, 1]

        codes = labels[0].astype('category').cat.codes

        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(vis_x, vis_y, c=[ code * 2 for code in codes ], s=100, cmap=plt.cm.get_cmap("jet", 5))
#     plt.scatter(vis_x, vis_y)
    # plt.colorbar(ticks=[2, 4, 6, 8, 10])
    plt.clim(-0.5, 9.5)
    plt.show()
    # fig.savefig('{}_{}.png'.format(method, ts))


if __name__ == "__main__":
    sample_visualization_v2()
