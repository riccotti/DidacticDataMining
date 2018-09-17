import json
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_kmeans(dataset, maxvalue=10, centroids=None, bisecting_line=False, clusters=None,
                title=None):
    plt.subplots(figsize=(6, 6))
    for i, p in enumerate(dataset):
        if clusters is None:
            plt.plot(p[0], p[1], marker='o', color='blue', markersize=6)
        else:
            colors = ['blue', 'green', 'orange', 'black']
            markers = ['o', 's', 'd', 'v', '^']
            for j, cluster in enumerate(clusters):
                if list(p) in [list(c) for c in cluster]:
                    plt.plot(p[0], p[1], marker=markers[j], color=colors[j], markersize=6)
        plt.text(p[0] + 0.1, p[1] + 0.1, 'P%d' % i, fontsize=12)

    plt.grid()

    plt.xlim([-0.2, maxvalue])
    plt.ylim([-0.2, maxvalue])

    plt.xticks(np.arange(0, maxvalue + 1, 1))
    plt.yticks(np.arange(0, maxvalue + 1, 1))

    if centroids is not None:
        for i, c in enumerate(centroids):
            plt.plot(c[0], c[1], marker='x', markersize=20, color='red', zorder=10,
                     label='C%d - (%.2f, %.2f)' % (i, c[0], c[1]))
            plt.text(c[0] + 0.1, c[1] - 0.3, 'C%d' % i, fontsize=12)

        plt.legend(fontsize=12, handlelength=0, handleheight=0, markerscale=0, loc='best')

        if bisecting_line:
            for c1, c2 in itertools.combinations(centroids, 2):
                plt.plot([c1[0], c2[0]], [c1[1], c2[1]], color='green')

                p1x = (c1[0] + c2[0]) / 2.0
                p1y = (c1[1] + c2[1]) / 2.0

                if c1[0] == c2[0] or c1[1] == c2[1]:
                    plt.axvline(x=p1x, color='red')
                else:
                    m1 = 1.0 * (c1[1] - c2[1]) / (c1[0] - c2[0])

                    m2 = -1.0 / m1

                    p2x = 0.0
                    p2y = m2 * (p2x - p1x) + p1y

                    p3x = maxvalue
                    p3y = m2 * (p3x - p1x) + p1y

                    plt.plot([p2x, p3x], [p2y, p3y], color='red')

    plt.tick_params(axis='both', which='major', labelsize=14)

    if title is not None:
        plt.title(title, fontsize=14)

    plt.show()


def euclidean_distance(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


class DidatticKMeans:
    def __init__(self, K=2, centroid_indexs=None, dist='euclidean'):

        self.K = K
        self.centroid_indexs = centroid_indexs
        self.dist_str = dist
        if self.dist_str == 'euclidean':
            self.dist = euclidean_distance
        elif self.dist_str == 'manhattan':
            self.dist = manhattan_distance
        self.jdata = None

    def __continue__(self, c_old, c_new):
        return (np.all(c_old[0] == c_new[0]) and np.all(c_old[1] == c_new[1])) \
               or (np.all(c_old[0] == c_new[1]) and np.all(c_old[1] == c_new[0]))

    def __calculate_clusters(self, dataset, centroids):
        clusters = [[] for k in range(0, len(centroids))]
        labels = []
        for p in dataset:
            dist_c = list()
            for c in centroids:
                dist_c.append(self.dist(p, c))

            cid = int(np.argmin(dist_c))
            clusters[cid].append(p)
            labels.append(cid)
        return clusters, labels

    def __calculate_centroids(self, clusters):
        centroids = list()
        for points in clusters:
            centroids.append(np.mean(np.asarray(points), axis=0))
        return centroids

    def __print_centroids(self, centroids):
        for i, c in enumerate(centroids):
            print('C%d' % i, '(%.2f, %.2f)' % (c[0], c[1]))

    def __init_centroids(self, dataset, indexes):
        centroids = list()
        for idx in indexes:
            centroids.append(dataset[idx])
        return centroids

    def fit(self, dataset, step_by_step=False, plot_figures=True):

        self.jdata = dict()
        self.jdata['data'] = dataset.tolist()
        self.jdata['iterations'] = list()
        kmeans_iteration = dict()

        if plot_figures:
            plot_kmeans(dataset, title='K-Means - Dataset', maxvalue=np.max(dataset)+1)

        npoints = len(dataset)
        if self.centroid_indexs is None:
            self.centroid_indexs = np.random.randint(0, npoints, self.K)

        self.jdata['parameters'] = {
            'nbr_clusters': self.K,
            'centroid_indexs': list(self.centroid_indexs),
            'distance': self.dist_str,
        }

        centroids = self.__init_centroids(dataset, indexes=self.centroid_indexs)
        clusters, labels = self.__calculate_clusters(dataset, centroids)

        if plot_figures:
            plot_kmeans(dataset, centroids=centroids, bisecting_line=True,
                    title='K-Means - Iteration 0', clusters=clusters, maxvalue=np.max(dataset)+1)
            self.__print_centroids(centroids)

        kmeans_iteration['centers'] = [c.tolist() for c in centroids]
        kmeans_iteration['labels'] = labels[:]
        self.jdata['iterations'].append(kmeans_iteration)

        clusters, labels = self.__calculate_clusters(dataset, centroids)
        new_centroids = self.__calculate_centroids(clusters)

        if step_by_step:
            val = input('Continue')

        iteration = 1
        while not self.__continue__(centroids, new_centroids):
            centroids = new_centroids
            if plot_figures:
                plot_kmeans(dataset, centroids=centroids, bisecting_line=True,
                            title='K-Means - Iteration %d' % iteration, clusters=clusters, maxvalue=np.max(dataset)+1)
                self.__print_centroids(centroids)

            clusters, labels = self.__calculate_clusters(dataset, centroids)
            kmeans_iteration = dict()
            kmeans_iteration['centers'] = [c.tolist() for c in centroids]
            kmeans_iteration['labels'] = labels[:]
            self.jdata['iterations'].append(kmeans_iteration)
            new_centroids = self.__calculate_centroids(clusters)
            iteration += 1

            if step_by_step:
                val = input('')

        if plot_figures:
            plot_kmeans(dataset, centroids=centroids, bisecting_line=True,
                        title='K-Means - Iteration %d' % iteration, clusters=clusters, maxvalue=np.max(dataset)+1)
            self.__print_centroids(centroids)
            plot_kmeans(dataset, clusters=clusters, title='K-Means - Result', maxvalue=np.max(dataset)+1)

    def get_jdata(self):
        return self.jdata
