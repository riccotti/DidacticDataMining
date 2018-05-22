import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def plot_dbscan(dataset, maxvalue_x=10, maxvalue_y=10, title=None, radius=1.9,
                core_points=None, border_points=None, noise_points=None, clusters=None):

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.axis('equal')

    any_core = False
    any_border = False
    any_noise = False

    plt.plot(maxvalue_x + np.ceil(radius), maxvalue_y + np.ceil(radius), marker='o', color='yellow', zorder=10)

    circle = plt.Circle((maxvalue_x + np.ceil(radius), maxvalue_y + np.ceil(radius)),
                        radius, color='orange', alpha=0.7)
    ax.add_artist(circle)

    for i, p in enumerate(dataset):
        if clusters is None:
            plt.plot(p[0], p[1], marker='o', color='blue', markersize=6)
        else:
            colors = ['blue', 'green', 'orange', 'black']
            markers = ['o', 's', 'd', 'v', '^']
            for j, cluster in enumerate(clusters):
                if list(p) in [list(c) for c in cluster]:
                    plt.plot(p[0], p[1], marker=markers[j], color=colors[j], markersize=6)

        if core_points is not None and i in core_points:
            if not any_core:
                any_core = True
                plt.plot(p[0], p[1], marker='o', color='blue', markersize=18, alpha=0.3, label='Core')
            else:
                plt.plot(p[0], p[1], marker='o', color='blue', markersize=18, alpha=0.3)

        if border_points is not None and i in border_points:
            if not any_border:
                any_border = True
                plt.plot(p[0], p[1], marker='s', color='green', markersize=18, alpha=0.3, label='Border')
            else:
                plt.plot(p[0], p[1], marker='s', color='green', markersize=18, alpha=0.3)

        if noise_points is not None and i in noise_points:
            if not any_noise:
                any_noise = True
                plt.plot(p[0], p[1], marker='d' if clusters is None else 'x', color='red',
                         markersize=18 if clusters is None else 10, alpha=0.8, label='Noise')
            else:
                plt.plot(p[0], p[1], marker='d' if clusters is None else 'x', color='red',
                         markersize=18 if clusters is None else 10, alpha=0.8)

        plt.text(p[0] + 0.1, p[1] + 0.1, 'P%d' % i, fontsize=12)

    plt.grid()

    plt.xlim([-0.2, maxvalue_x + np.ceil(radius) + 1 + 0.2])
    plt.ylim([-0.2, maxvalue_y + np.ceil(radius) + 1 + 0.2])

    plt.xticks(np.arange(-2, maxvalue_x + np.ceil(radius) + 1 + 2, 1))
    plt.yticks(np.arange(0, maxvalue_y + np.ceil(radius) + 1 + 2, 1))

    plt.tick_params(axis='both', which='major', labelsize=14)

    if title is not None:
        plt.title(title, fontsize=14)

    if any_core or any_border or any_noise:
        plt.legend(fontsize=12, loc='lower right', handlelength=0.5)

    plt.show()


class DidatticDbscan:
    def __init__(self, eps=1.8, min_pts=3):
        self.eps = eps
        self.min_pts = min_pts

    def fit(self, dataset, step_by_step=False):

        maxvalue_x = np.max(dataset, 0)[0]  # np.max(dataset) #np.max(dataset, 0)[0]
        maxvalue_y = np.max(dataset, 0)[1]  # np.max(dataset) #np.max(dataset, 0)[1]

        plot_dbscan(dataset, title='Dbscan - Init', radius=self.eps,
                    maxvalue_x=maxvalue_x, maxvalue_y=maxvalue_y)
        if step_by_step:
            ret = input('')

        dist_matrix = squareform(pdist(dataset))

        core_points = dict()
        pidx_neighbors = dict()
        for pidx, distances in enumerate(dist_matrix):
            pidx_neighbors[pidx] = np.where(distances <= self.eps)[0]
            nbr_neighbors = len(pidx_neighbors[pidx])

            if nbr_neighbors >= self.min_pts:
                core_points[pidx] = 1

                # print pidx, np.where(distances <= eps), ) #distances

        plot_dbscan(dataset, title='Dbscan - Core Points', radius=self.eps, core_points=core_points,
                    maxvalue_x=maxvalue_x, maxvalue_y=maxvalue_y)
        print('Core Points', core_points.keys())
        if step_by_step:
            ret = input('')

        border_points = dict()
        noise_points = dict()
        for pidx, distances in enumerate(dist_matrix):

            if pidx in core_points:
                continue

            is_border = False
            for pidx2 in pidx_neighbors[pidx]:
                if pidx2 in core_points:
                    border_points[pidx] = 0
                    is_border = True
                    break

            if not is_border:
                noise_points[pidx] = 0

        plot_dbscan(dataset, title='Dbscan - Border Points', radius=self.eps,
                    core_points=core_points, border_points=border_points,
                    maxvalue_x=maxvalue_x, maxvalue_y=maxvalue_y)
        print('Border Points', border_points.keys())
        if step_by_step:
            ret = input('')

        plot_dbscan(dataset, title='Dbscan - Noise Points', radius=self.eps,
                    core_points=core_points, border_points=border_points, noise_points=noise_points,
                    maxvalue_x=maxvalue_x, maxvalue_y=maxvalue_y)
        print('Noise Points', noise_points.keys())
        if step_by_step:
            ret = input('')

        clusters = defaultdict(set)
        pidx_clusterid = dict()
        clusterid = 0
        for pidx, distances in enumerate(dist_matrix):
            pidx_neighbors[pidx] = np.where(distances <= self.eps)[0]
            nbr_neighbors = len(pidx_neighbors[pidx])

            if nbr_neighbors >= self.min_pts:
                if pidx not in pidx_clusterid:
                    pidx_clusterid[pidx] = clusterid
                    clusterid += 1
                    clusters[pidx_clusterid[pidx]].add(pidx)

                for pidx2 in pidx_neighbors[pidx]:
                    if pidx2 not in pidx_clusterid:
                        pidx_clusterid[pidx2] = pidx_clusterid[pidx]
                    clusters[pidx_clusterid[pidx]].add(pidx2)

        clusters_new = list()
        for clusterid in clusters:
            lista = list()
            for pidx in clusters[clusterid]:
                lista.append(dataset[pidx])
            clusters_new.append(lista)

        plot_dbscan(dataset, title='Dbscan - Result', clusters=clusters_new, noise_points=noise_points,
                    maxvalue_x=maxvalue_x, maxvalue_y=maxvalue_y)
        print(clusters)

