import json
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def dist2similarity(dist_matrix):
    max_val = np.max(dist_matrix)
    sim_matrix = np.zeros(dist_matrix.shape)
    for i in range(0, sim_matrix.shape[0]):
        for j in range(0, sim_matrix.shape[1]):
            sim_matrix[i][j] = 1.0 - dist_matrix[i][j]/max_val
    return sim_matrix


def plot_base_hierarchy(dist_clustering_dict, x, clustering_xlist_dict, init_order, npoints, use_distances=True):
    base_dist = sorted(dist_clustering_dict, reverse=not use_distances)[0]
    clustering = dist_clustering_dict[base_dist]

    order = str(init_order)
    order = order.replace('(', '')
    order = order.replace(')', '')
    order = order.replace('[', '')
    order = order.replace(']', '')
    order = order.replace(',,', ',')
    order = order.replace(' ', '').strip()
    order = order.split(',')
    order = [o for o in order if o != '']
    order_dict = {int(x): i for i, x in enumerate(order)}

    val = 0.0
    if not use_distances:
        val = 1.0

    xticks_list = ['P'] * npoints
    for cluster in clustering:
        x_list = list()
        for pidx in cluster:
            xreal = order_dict[pidx]
            plt.plot([xreal, xreal], [val, base_dist], linewidth=2, color='blue')
            xticks_list[xreal] = 'P%d' % pidx
            x_list.append(xreal)
            x += 1
        plt.plot([min(x_list), max(x_list)], [base_dist, base_dist], linewidth=3, zorder=10)
        clustering_xlist_dict[tuple(cluster)] = x_list

    return x, clustering_xlist_dict, xticks_list


def plot_up_hierarchy(dist_clustering_dict, clustering_xlist_dict, i, use_distances=True):
    dist_1 = sorted(dist_clustering_dict, reverse=not use_distances)[i - 1]
    dist = sorted(dist_clustering_dict, reverse=not use_distances)[i]
    clustering = dist_clustering_dict[dist]
    for cluster in clustering:
        x_list2 = list()
        x_list2id = list()
        for c in cluster:
            if type(c) is tuple:
                x_list = clustering_xlist_dict[c]
            else:
                x_list = clustering_xlist_dict[tuple([c])]
            x = np.mean(x_list)
            plt.plot([x, x], [dist_1, dist], linewidth=2, color='blue')
            x_list2.append(x)
            x_list2id.append(c)
        plt.plot([min(x_list2), max(x_list2)], [dist, dist], linewidth=3, zorder=10)
        clustering_xlist_dict[tuple(x_list2id)] = x_list2
    if (use_distances and dist < max(dist_clustering_dict)) \
            or (not use_distances and dist > min(dist_clustering_dict)):
        plot_up_hierarchy(dist_clustering_dict, clustering_xlist_dict, i + 1, use_distances)


def plot_dendogram(dist_clustering_dict, npoints, use_distances=True, title=None):
    agg_fun = np.max
    if not use_distances:
        agg_fun = np.min

    x = 0
    clustering_xlist_dict = dict()
    plt.subplots(figsize=(6, 4))

    init_order = dist_clustering_dict[agg_fun(list(dist_clustering_dict.keys()))]

    x, clustering_xlist_dict, xticks_list = plot_base_hierarchy(dist_clustering_dict, x, clustering_xlist_dict,
                                                                init_order, npoints, use_distances)
    if len(dist_clustering_dict) > 1:
        plot_up_hierarchy(dist_clustering_dict, clustering_xlist_dict, 1, use_distances)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([0, max(dist_clustering_dict) + 1])
    if not use_distances:
        plt.ylim([1.0, min(dist_clustering_dict) - 0.1])
    plt.xticks(range(0, npoints), xticks_list)
    plt.xlim([-1, npoints])

    if title is not None:
        plt.title(title, fontsize=14)
    plt.show()


class DidatticHierarchical:

    def __init__(self):
        np.set_printoptions(precision=2, suppress=True)
        self.jdata = None

    def fit(self, dataset, link_criteria='single', use_distances=True, step_by_step=False, distance_type='euclidean',
            plot_figures=True):

        if link_criteria == 'single' or link_criteria == 'min':
            link_fun = np.min if use_distances else np.max
            title_alg = 'Single'
        elif link_criteria == 'complete' or link_criteria == 'max':
            link_fun = np.max if use_distances else np.min
            title_alg = 'Complete'
        elif link_criteria == 'average' or link_criteria == 'mean':
            link_fun = np.mean
            title_alg = 'Average'
        else:
            print("Unknown link criteria, please specify 'single', 'complete' or 'average'")
            return

        self.jdata = dict()
        self.jdata['data'] = dataset.tolist()
        self.jdata['parameters'] = {
            'link_criteria': link_criteria,
            'distance': distance_type,
            'matrix_type': 'distance' if use_distances else 'similarity',
        }
        self.jdata['iterations'] = list()

        dist_matrix = squareform(pdist(dataset, distance_type))
        agg_fun = np.min
        val = 0
        mask_fun = np.ma.less_equal

        if not use_distances:
            dist_matrix = dist2similarity(dist_matrix)
            agg_fun = np.max
            val = 1
            mask_fun = np.ma.greater_equal

        npoints = dist_matrix.shape[0]
        clusters_labels = [tuple([i]) for i in range(0, npoints)]
        clusters_labels_map = {i: i for i in range(0, npoints)}

        iterid = 0
        dist_merge = None
        plot_levels = None

        dist_merge = 0.0 if use_distances else 1.0
        dist_clustering_dict = dict()
        while True:
            hierarchical_iteration = dict()
            if iterid > 0:
                dist_clustering_dict[dist_merge] = clusters_labels
                if plot_figures:
                    plot_dendogram(dist_clustering_dict, npoints, use_distances,
                                   '%s-Linkage - Iteration %d' % (title_alg, iterid))

            if not use_distances and dist_merge == 0.0:
                return

            if plot_figures:
                print('iter', iterid)
                print('%s merge' % ('distance' if use_distances else 'similarity'), '%.2f' % dist_merge)
                print(clusters_labels)
                print(dist_matrix)
            hierarchical_iteration['dist_merge'] = dist_merge
            hierarchical_iteration['clusters_labels'] = json.loads(json.dumps(clusters_labels))
            hierarchical_iteration['dist_matrix'] = dist_matrix.tolist()

            if dist_matrix.shape[0] == 1 and dist_matrix.shape[1] == 1:
                break

            # fase di merge
            masked_dist_matrix = np.ma.masked_array(dist_matrix, mask_fun(dist_matrix, val))
            dist_merge = agg_fun(masked_dist_matrix)
            min_pts = np.where(masked_dist_matrix == dist_merge)

            point_aggregatewith = defaultdict(list)
            points_involved = dict()
            for p1, p2 in zip(min_pts[0], min_pts[1]):
                point_aggregatewith[p1].append(p2)
                points_involved[p1] = 0
                points_involved[p2] = 0

            if plot_figures:
                print(default_to_regular(point_aggregatewith))

            hierarchical_iteration['point_aggregatewith'] = {str(k): [int(vv) for vv in v] for k, v in point_aggregatewith.items()}
            
            new_clusters_tmp = dict()
            for p, plist in point_aggregatewith.items():
                lista = set()
                lista.add(p)
                for p2 in plist:
                    lista.add(p2)
                    for p3 in point_aggregatewith[p2]:
                        lista.add(p3)
                clusterid = tuple(sorted(lista))
                new_clusters_tmp[clusterid] = 0

            reverse_clusters_labels_map = {v: k for k, v in clusters_labels_map.items()}
            clusters_labels_map = {i: k for i, k in enumerate(sorted(reverse_clusters_labels_map))}
            
            new_clusters = dict()
            clusters_labels = list()
            new_clusters_labels_map = dict()
            new_cluster_id = 0
            for clusterid1 in sorted(new_clusters_tmp):
                flag = True
                items_to_add = list()
                for clusterid2 in sorted(new_clusters_tmp):
                    if set(clusterid1) < set(clusterid2):
                        flag = False
                        break
                if flag:
                    new_clusters[clusterid1] = 0
                    cluster_label = list()
                    for pid in sorted(clusterid1):
                        cluster_label.append(clusters_labels_map[pid])
                    clusters_labels.append(tuple(sorted(cluster_label)))
                    new_clusters_labels_map[new_cluster_id] = tuple(sorted(cluster_label))
                    new_cluster_id += 1

            for pidx in range(0, dist_matrix.shape[0]):
                if pidx not in points_involved:
                    new_clusters[tuple([pidx])] = 0
                    clusters_labels.append(tuple([clusters_labels_map[pidx]]))
                    new_clusters_labels_map[new_cluster_id] = tuple([clusters_labels_map[pidx]])
                    new_cluster_id += 1

            clusters_labels = sorted(clusters_labels)

            # fase di ricalcolo matrice
            new_dist_matrix = np.zeros((len(new_clusters), len(new_clusters))) if use_distances else np.ones((len(new_clusters), len(new_clusters)))
            for i, clusterid1 in enumerate(sorted(new_clusters.keys())):
                for j, clusterid2 in enumerate(sorted(new_clusters.keys())):
                    if i == j:
                        continue

                    new_dist_matrix[i][j] = link_fun(dist_matrix[np.ix_(clusterid1, clusterid2)])

            dist_matrix = new_dist_matrix
            clusters_labels_map = new_clusters_labels_map
            iterid += 1
            print('')

            if step_by_step:
                ret = input('')

            self.jdata['iterations'].append(hierarchical_iteration)

    def get_jdata(self):
        return self.jdata

