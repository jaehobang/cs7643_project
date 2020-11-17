"""
Objective of this file
1. See the insides of agglomerative clustering, which function does which (we need to extract the overall loss that is the result of agglomerative clustering)


"""

#from sklearn.cluster import AgglomerativeClustering
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_memory, check_array, _deprecate_positional_args
from loaders.seattle_loader import SeattleLoader
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
from heapq import heapify, heappop, heappush, heappushpop
from sklearn.cluster import AgglomerativeClustering

##### functions to derive the loss for agglomerative clustering
###############################################################

def merge_cost(A, B):
    n = len(A)
    m = len(B)

    m_A = np.mean(A, axis=0)
    m_B = np.mean(B, axis=0)
    linalg_result = np.linalg.norm(m_A - m_B)
    overall = linalg_result * (n * m) / (n + m)
    print(f"{A} {B} | {m_A} {m_B} | {linalg_result} | {overall}")
    print('---------')
    return overall


def get_elements(original_data, translator, child):
    ### here we just add the necessary data to list

    if child < len(original_data):
        element = original_data[child]
    else:
        element = translator[child]


def create_translator(children):
    translator = {}
    n_samples = children.shape[0] + 1
    n_operations = children.shape[0]
    for i in range(n_samples):
        translator[i] = np.array([i])
    newest_number = n_samples
    for i in range(n_operations):
        left, right = children[i]
        translator[newest_number] = np.concatenate([translator[left], translator[right]])
        newest_number += 1
    return translator


def agg_compute_losses(original_data, children):
    n_operations, _ = children.shape
    losses = []
    ## you can use np.cumsum to get cumulative loss if interested
    translator = create_translator(children)

    for i in range(n_operations):
        ## first form the A and B
        left_child = children[i][0]
        right_child = children[i][1]
        A = original_data[translator[left_child]]
        B = original_data[translator[right_child]]
        losses.append(merge_cost(A, B))

    cum_loss = np.cumsum(losses)
    return losses, cum_loss


############ end
###########################################################






if __name__ == "__main__":
    loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_short.mp4'
    images = loader.load_images(directory)
    images = images[:100]
    images = np.reshape(images, (100, -1))
    #num_clusters = 10
    distance_threshold = 10
    ac = AgglomerativeClustering(n_clusters = None, distance_threshold = distance_threshold, linkage =  'ward', compute_full_tree = True)
    ac.fit(images)
    print(ac.distances_)