import numpy as np

from main import get_image_array
from snippets import GaussMVD
from PIL import Image


def draw_array(pixels_by_clusters):
    im_array = get_image_array(pixels_by_clusters)
    res_im = Image.fromarray(im_array, "L")
    Image._show(res_im)


class CMixedGauss:
    _dim = 3
    _cluster_count = 2
    _convergence_threshold = 0.00001

    def __init__(self, rnd):
        self._gausses = list()
        for i in range(self._cluster_count):
            g = GaussMVD()
            means = rnd.rand(g.dim)
            s = rnd.rand() * 0.3
            cov = np.eye(g.dim, dtype=np.float64) * s
            g.modify(means, cov)
            self._gausses.append(g)

    def compute_probs(self, arr_values):
        sh = np.shape(arr_values)
        row_count = sh[0]
        col_count = sh[1]

        d = np.zeros((self._cluster_count, row_count, col_count), dtype=np.float64)

        for k in range(self._cluster_count):
            d[k] = self._gausses[k].compute_probs(arr_values)
        return d

    def find_clusters(self, arr_values):
        sh = np.shape(arr_values)
        row_count = sh[0]
        col_count = sh[1]

        for k in range(self._cluster_count):
            print "start k=", k
            self._gausses[k].write()

        value_count = row_count * col_count
        w = np.full(self._cluster_count, 1.0 / self._cluster_count)  # 2L
        pixels_by_clusters = np.zeros((self._cluster_count, row_count, col_count), dtype=np.int)
        diff = 1

        while diff != 0:
            px = self.compute_probs(arr_values)  # (2L, 289L, 250L)
            # draw_array(px[0])
            gsum = np.zeros((self._cluster_count, row_count, col_count), dtype=np.float64)
            for k in range(self._cluster_count):
                gsum[k] = px[k] * w[k]
            gsum = np.sum(gsum, 0)

            g = np.zeros((self._cluster_count, row_count, col_count), dtype=np.float64)

            for k in range(self._cluster_count):
                g[k] = px[k] * w[k]
                g[k] = g[k] / gsum
                # draw_array(g[k])

                w[k] = np.sum(g[k]) / value_count

                self._gausses[k].estimate(arr_values, g[k])

                # if k == 0:
                #     print "k=", k
                #     self._gausses[k].write()

            cs = np.sum(w)

            pc = np.argmax(g, 0)
            # draw_array(pc)
            dm = pc - pixels_by_clusters
            pixels_by_clusters = pc
            dm = dm * dm
            diff = np.sum(dm)
            print "diff=", diff

        return pixels_by_clusters
