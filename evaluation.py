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

settings_label = {
    'sep': '\t',
    'index_col': 0
}

l_rate = {
    'cora': 0.07,
    'citeseer': 0.3,
    'terr': 0.06
}

def evaluate_classification(settings, settings_label, l_rate):
    # 读取数据文件
    feature_df = pd.read_csv("./emb/{}{}.emd".format(FILENAME, VERSION), header=None, **settings).iloc[:, :]
    label_df = pd.read_csv("./dataset/ori/{}/group.txt".format(FILENAME), header=None, sep='\t', index_col=0)#, index_col=0)


    df = pd.concat([feature_df, label_df], axis=1)

    df = df.dropna(axis=0, how='any')

    X = df.values[:, :-1]
    Y = df.values[:,-1:].ravel()

    Y = Y.astype(int)

    t = dict()
    for i in Y:
        if i not in t:
            t[i] = i

    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    # 设置随机数种子，以便比较结果。

    # 3.标准化特征值
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    X_test_std = X_test
    X_train_std = X_train


    # 4. 训练逻辑回归模型
    logreg = linear_model.LogisticRegression(C=l_rate[FILENAME])
    logreg.fit(X_train_std, Y_train)

    # 5. 预测
    prepro = logreg.predict_proba(X_test_std)
    predict = logreg.predict(X_test_std)

    acc = logreg.score(X_test_std,Y_test)
    recall = recall_score(Y_test, predict, average='weighted')
    precision = precision_score(Y_test, predict, average='weighted')
    return recall, precision

def load_graph_from_file():
    graph = [[]]
    ts = 0
    with open('./dataset/node2vec/{}{}.cites'.format(FILENAME, VERSION)) as f:
        line = f.readline()
        while line:
            if line == '\n':
                ts += 1
                graph.append([])
            else:
                edge = line.strip().split('\t')
                if len(edge) < 2:
                    edge = edge[0].split(' ')
                graph[ts].append(edge)
            line = f.readline()
    return graph

def evaluate_link_predict(ts, settings):
    feature_df = pd.read_csv("./emb/{}{}.emd".format(FILENAME, VERSION), header=None, **settings).iloc[:, :128]
    graph = load_graph_from_file()

    existed_edges_dict = {}

    # build adjacent matrix
    for edges in graph[:ts]:
        for edge in edges:
            for i in range(2):
                if edge[i] not in existed_edges_dict:
                    existed_edges_dict[edge[i]] = {}
            existed_edges_dict[edge[0]][edge[1]] = 1
            existed_edges_dict[edge[1]][edge[0]] = 1


    # choose the timestamp we want to examinate
    predicted_edges_dict = {}
    predicted_edges = []
    for edges in graph[ts:]:
        for edge in edges:
            if edge[0] not in existed_edges_dict or edge[1] not in existed_edges_dict:
                continue
            predicted_edges.append(edge)
            for i in range(2):
                if edge[i] not in predicted_edges_dict:
                    predicted_edges_dict[edge[i]] = {}
            predicted_edges_dict[edge[0]][edge[1]] = 1
            predicted_edges_dict[edge[1]][edge[0]] = 1

    # randomly create unexisted edges
    unexisted_edges_dict = {}
    unexisted_edges = []
    nodes = list(feature_df.index.values)
    node2index = {}


    for i in range(len(nodes)):
        nodes[i] = str(nodes[i])
        node2index[nodes[i]] = i


    i = 0

    while len(unexisted_edges) < len(predicted_edges) and i < len(nodes):
        for q in range(i + 1, len(nodes)):
            if len(unexisted_edges) >= len(predicted_edges):
                break
            if nodes[i] in existed_edges_dict and nodes[q] in existed_edges_dict[nodes[i]]:
                continue
            if nodes[i] in predicted_edges and nodes[q] in predicted_edges[nodes[i]]:
                continue
            if nodes[i] not in existed_edges_dict or nodes[q] not in existed_edges_dict:
                continue
            unexisted_edges.append([nodes[i], nodes[q]])
            if nodes[i] not in unexisted_edges_dict:
                unexisted_edges_dict[nodes[i]] = {}
            unexisted_edges_dict[nodes[i]][nodes[q]] = 1
            if nodes[q] not in unexisted_edges_dict:
                unexisted_edges_dict[nodes[q]] = {}
            unexisted_edges_dict[nodes[q]][nodes[i]] = 1
        i += 1


    # if not len(predicted_edges):
    #     return 0


    features = feature_df.values

    # 3.标准化特征值
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(features)
    features_std = sc.transform(features)
    # features_std = features


    X = []

    for edge in predicted_edges + unexisted_edges:
        if edge[1] not in node2index:
            print(edge)
        node1_feas = features_std[node2index[edge[0]]]
        node2_feas = features_std[node2index[edge[1]]]
        X.append([cosine_similarity([ node1_feas, node2_feas ])[0][1]])
    Y = [1] * len(predicted_edges) + [0] * len(unexisted_edges)

    # plt.figure()
    # plt.title('this is title')
    # plt.xlabel('x label')
    # plt.ylabel('y label')
    # plt.grid(True)
    # plt.plot(X, Y, 'k.')
    # plt.show()

    score = roc_auc_score(Y, X)
    print(score)
    return score



    # 2.拆分测试集、训练集。
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    # 设置随机数种子，以便比较结果。

    # 4. 训练逻辑回归模型
    # model = linear_model.LinearRegression()
    # model.fit(X_train, Y_train)

    # 5. 预测
    # predict = model.predict(X_test)
    # print X_test

def sample_visualization_v2():
    # take ids of the first 100 points
    ids = pd.read_csv("./dataset/node2vec/terr_0.3_0.01.content", header=None, sep='\t').iloc[:20, 0].values

    VERSIONS = ["terr_0.3_0.01_da-node2vec_{}.emd".format(i) for i in [1, 5, 9] ]

    X = None
    Y = None
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
    print(vis_data)

    # vis_data = bh_sne(X, random_state=np.random.RandomState(1), perplexity=1)
    vis_x = vis_data.iloc[:, 0]
    vis_y = vis_data.iloc[:, 1]

    codes = labels[0].astype('category').cat.codes

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=[ code * 2 for code in codes ], s=100, cmap=plt.cm.get_cmap("jet", 5))
#     plt.scatter(vis_x, vis_y)
    # plt.colorbar(ticks=[2, 4, 6, 8, 10])
    plt.clim(-0.5, 9.5)
    plt.show()
    # fig.savefig('{}_{}.png'.format(method, ts))



def sample_visualization(method, ts, settings, sample_indices):
    # take ids of the first 100 points
    ids = pd.read_csv("./dataset/node2vec/terr_0.3_0.01.content", header=None, sep='\t').iloc[:20, 0].values

    feature_df = pd.read_csv("./emb/{}{}.emd".format(FILENAME, VERSION), header=None, **settings).iloc[:, :]
    feature_df = feature_df.loc[ids, :]
    print(feature_df.iloc[:, :3])
    # draw the first four nodes

    zero_data = np.ones(shape=(len(ids), 1))
    for i in range(0, 4):
        zero_data[i] = i + 2
    labels = pd.DataFrame(zero_data)
    Y = labels.values
    X = feature_df.values

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    vis_data = pd.DataFrame(data = principalComponents)
    print(vis_data)

    # vis_data = bh_sne(X, random_state=np.random.RandomState(1), perplexity=1)
    vis_x = vis_data.iloc[:, 0]
    vis_y = vis_data.iloc[:, 1]

    codes = labels[0].astype('category').cat.codes

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=[ code * 2 for code in codes ], s=100, cmap=plt.cm.get_cmap("jet", 5))
#     plt.scatter(vis_x, vis_y)
    # plt.colorbar(ticks=[2, 4, 6, 8, 10])
    plt.clim(-0.5, 9.5)
    plt.show()
    # fig.savefig('{}_{}.png'.format(method, ts))

def visualization(method, ts, settings, settings_label):
    feature_df = pd.read_csv("./emb/{}{}.emd".format(FILENAME, VERSION), header=None, **settings).iloc[:, :]
    label_df = pd.read_csv("./dataset/ori/{}/group.txt".format(FILENAME))#, header=None)

    df = pd.concat([feature_df, label_df], axis=1)

    df = df.dropna(axis=0, how='any')

    X = df.values[:, :-1]
    Y = df.values[:,-1:].ravel()

    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # sc.fit(X)
    # X = sc.transform(X)

    X = df.iloc[:, :-1]

    vis_data = bh_sne(X)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    codes = df.iloc[:,-1].astype('category').cat.codes

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=codes, s=1, cmap=plt.cm.get_cmap("jet", 10))
#     plt.scatter(vis_x, vis_y)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
#     plt.show()
    fig.savefig('{}_{}.png'.format(method, ts))


if __name__ == "__main__":
    pass
    #evaluate_classification(settings, settings_label,)
    # global VERSION

    sample_visualization(method=None, ts=None, settings=settings, sample_indices=None)

    # evaluate_classification(settings, settings_label, l_rate)
    # visualization('tadw_terr', 0)
    # evaluate_link_predict(1)
