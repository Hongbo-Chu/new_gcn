from tkinter import N
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score #用的是不是cosine distance?????
from sklearn.cluster import SpectralClustering
import pandas as pd
import numpy as np
from evaluation.evaluation import evaluate
import copy
from collections import Counter
def bi_partition(clusters:list, grand_truth:list, i):
    '''
    
    output:
    a list:[[cluter1],[cluster2]]
    
    '''
    if len(clusters[i]) == 0:
      return clusters
    all_cluster = copy.deepcopy(clusters)
    all_label = copy.deepcopy(grand_truth)
    input_clusters = copy.deepcopy(clusters[i])
    input_label = copy.deepcopy(grand_truth[i])
    Scluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',n_neighbors=10)
    temp_list = []
    input_clusters = np.array(input_clusters)
    input_label = np.array(input_label)
    print(f"当前cluster中真实标签统计:{Counter(input_label)}")
    # print(input_clusters.shape)
    # print(input_clusters)
    # print("hhhhhhh")
    # print("inf检查")
    # print(np.isposinf(input_clusters).any())
    # print(np.isneginf(input_clusters).any())
    # tt = np. random.randn(*input_clusters.shape)
    # print(f"max:{np.max(input_clusters)}min:{np.min(input_clusters)}, absmin:{np.min(abs(input_clusters))},avg: {np.mean(input_clusters)}")
    # print(input_clusters[1])
    bi_partition_label = Scluster.fit_predict(input_clusters)
    # print("cocococo")
    # print(bi_partition_label)
    bi_paritition_0 = []
    bi_paritition_1 = []
    bi_partition_0_true_label = []
    bi_partition_1_true_label = []
    clusters_temp = []
    for k, val in enumerate(bi_partition_label):
        if val == 0:
            bi_paritition_0.append(input_clusters[k])
            bi_partition_0_true_label.append(input_label[k])#真实标签，最后验证用
        else:
            bi_paritition_1.append(input_clusters[k]) 
            bi_partition_1_true_label.append(input_label[k])#真实标签，最后验证用
    print(f"cluster0:{len(bi_paritition_0)},cluster1:{len(bi_paritition_1)}")
    all_cluster.pop(i)
    all_cluster.append(bi_paritition_0)
    all_cluster.append(bi_paritition_1)
    all_label.pop(i)
    all_label.append(bi_partition_0_true_label)
    all_label.append(bi_partition_1_true_label)
    return all_cluster, all_label

def clusterToList(cluster):
    all_temp = []
    label_temp = []
    for i in range(len(cluster)):
        all_temp.extend(cluster[i])
        label_temp.extend([i for _ in range(len(cluster[i]))])
    return all_temp, label_temp

def mergeClusters(clusters:list, labels:list, clus_num):
    """
    merge the clusters after refinement
    input: clusters:list
    output:merged clusters:list
            label of each cluster
    """
    #统计每个cluster的种类
    clu_labels = []
    final_clusters = [[] for _ in range(clus_num)]
    final_truelabel = [[] for _ in range(clus_num)]
    for clus_label in labels:
        temp = Counter(clus_label)
        clas = max(temp.keys(), key=temp.get)
        clu_labels.append(clas)
    
    idx = [] #format:[[index of clus1],[...2],[...3],[],[]]
    for i in range(clus_num):#根据标签进行聚类
        #找到对应的标签
        idx.append([j for j, x in enumerate(clu_labels) if x == i])
    
    #final merge
    for k in range(clus_num):
        for j in idx[k]:
            final_clusters[k].extend(clusters[j])
            final_truelabel[k].extend(labels[j])

    return final_clusters, final_truelabel
    
    


def Refinement(input, clus_label, grand_truth, clus_nums):
    '''
    params:
    input: 每个图片的feature vector
    clus_label: clusters of each image. 
        dim = 1 X N
    clus_num: total nums of the clusters
    ''' 
    sc_pre = silhouette_score(input, clus_label.reshape(-1,1), metric='cosine')
    sc_best = sc_pre
    i = 0
    i_best = i
    classes = clus_nums
    #先按照clusters分类
    clusters = [[] for _ in range(clus_nums)]
    true_label = [[] for _ in range(clus_nums)]
    for k, (vec, tru) in enumerate(zip(input, grand_truth.tolist())): 
        clusters[clus_label[k]].append(vec)
        true_label[clus_label[k]].append(tru)
    while(i_best != -1): # 证明有更新       #sc_best >= sc_pre):
        i_best =-1
        #遍历，找最大的sc
        i = 0
        print(f"当前cluster数量{classes}, sc_best={sc_best}")
        while(i < classes):
            # Bi-partition i-th cluster with spectral clustering

            clusters_temp, _ = bi_partition(clusters, true_label, i)
            all_temp, label_temp = clusterToList(clusters_temp)
            # print(all_temp)
            # print(label_temp)
            sc_new = silhouette_score(all_temp, label_temp, metric='cosine')
            print(f"第{i}个cluster, sc={sc_new} ")
            if sc_new > sc_best:
                sc_best = sc_new
                i_best = i
                # #更新clusters
                # clusters.pop[i]
                # clusters.append(bi_paritition_0)
                # clusters.append(bi_paritition_1)
            i += 1

        if i_best != -1:#证明有更大的sc
            #更新cluster
            classes +=1
            print("i_best:")
            print(i_best)
            print("****************************************")
            for i in clusters:
              print(len(i))
            print("*************************************")
            clusters, true_label = bi_partition(clusters, true_label, i_best)
            print("after bi-partition")
            print("****************************************")
            for i in clusters:
              print(len(i))
            print("*************************************")
            all_temp, label_temp = clusterToList(clusters)
            sc_pre = silhouette_score(all_temp, label_temp, metric='cosine')
            sc_best = sc_pre
            print("sc_best")
            print(sc_best)
            #compute acc
            test_cluster = copy.deepcopy(clusters)
            test_ture_label = copy.deepcopy(true_label)
            final_clus, final_truelabel = mergeClusters(test_cluster,test_ture_label, 9)
            cluster_list, pre_label_list = clusterToList(final_clus)
            #将聚类标签和真实标签变成list
            true_label_list = []
            for i in final_truelabel:
                true_label_list.extend(i)
            nmi, ari, f, acc = evaluate(true_label_list, pre_label_list)
            print('after refinement NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
        else:
            test_cluster = copy.deepcopy(clusters)
            test_ture_label = copy.deepcopy(true_label)
            final_clus, final_truelabel = mergeClusters(test_cluster,test_ture_label, 9)
            cluster_list, pre_label_list = clusterToList(final_clus)
            #将聚类标签和真实标签变成list
            true_label_list = []
            for i in final_truelabel:
                true_label_list.extend(i)
            nmi, ari, f, acc = evaluate(true_label_list, pre_label_list)
            print('after refinement NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
            break
    return clusters
