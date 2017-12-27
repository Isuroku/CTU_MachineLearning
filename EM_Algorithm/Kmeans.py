import numpy as np


class CKMean:
    _dim = 3
    _cluster_count = 2
    _convergence_threshold = 0.00001

    def __init__(self, rnd):
        self._means = np.zeros((self._cluster_count, self._dim), dtype=np.float64)
        for i in range(self._cluster_count):
            self._means[i] = rnd.rand(1, self._dim)

    def init_means_from_array(self, array_pixels, rng, rnd):
        x = rnd.randint(0, rng)
        y = rnd.randint(0, rng)
        self._means[0] = array_pixels[x][y]

        sh = np.shape(array_pixels)
        row_count = sh[0]
        col_count = sh[1]

        x = col_count / 2 - rnd.randint(-rng, rng)
        y = row_count / 2 - rnd.randint(-rng, rng)
        self._means[1] = array_pixels[x][y]

    @staticmethod
    def vec_distance_p2(v1, v2):
        v = v1 - v2
        v = v * v
        d = np.sum(v)
        return d

    @staticmethod
    def arr_distance_p2(v1, v2):
        v = v1 - v2
        v = v * v
        d = np.sum(v, 2)
        return d

    def distance_means_p2(self, array_pixels):
        sh = np.shape(array_pixels)
        row_count = sh[0]
        col_count = sh[1]

        d = np.zeros((self._cluster_count, row_count, col_count), dtype=np.float64)

        # apply a function to all elements of the array
        # parr = np.log(arr / (1.0 - arr))

        for k in range(self._cluster_count):
            arr_mean = np.full_like(array_pixels, self._means[k], dtype=None, order='K', subok=False)
            d[k] = self.arr_distance_p2(array_pixels, arr_mean)
        return d

    def select_by_clusters(self, array_pixels):
        d = self.distance_means_p2(array_pixels)
        k = np.argmin(d, 0)
        return k

    def calc_new_means(self, array_pixels, pixels_by_clusters):
        new_means = np.zeros((self._cluster_count, self._dim), dtype=np.float64)

        sh = np.shape(array_pixels)
        row_count = sh[0]
        col_count = sh[1]

        for k in range(self._cluster_count):
            total = 0
            pixel_sum = np.zeros(self._dim)
            for l in range(row_count):
                for p in range(col_count):
                    if pixels_by_clusters[l][p] == k:
                        total += 1
                        pixel_sum += array_pixels[l][p]
            new_means[k] = pixel_sum / total
        return new_means

    def find_clusters(self, array_pixels):
        sh = np.shape(array_pixels)
        row_count = sh[0]
        col_count = sh[1]
        step = 1
        pixels_by_clusters = np.zeros((self._cluster_count, row_count, col_count), dtype=np.int)
        while step > self._convergence_threshold:
            pixels_by_clusters = self.select_by_clusters(array_pixels)
            new_means = self.calc_new_means(array_pixels, pixels_by_clusters)
            means_shift = np.zeros(self._cluster_count, dtype=np.float64)
            for k in range(self._cluster_count):
                means_shift[k] = self.vec_distance_p2(new_means[k], self._means[k])
            step = means_shift.max()
            # print step
            self._means = new_means
        return pixels_by_clusters

    def get_image_array(self, pixels_by_clusters):
        sh = np.shape(pixels_by_clusters)
        row_count = sh[0]
        col_count = sh[1]
        gray_scale_step = 1 / (self._cluster_count - 1)
        im_array = np.zeros((row_count, col_count), dtype=np.uint8)
        for l in range(row_count):
            for p in range(col_count):
                value = 255 * int(round(gray_scale_step * pixels_by_clusters[l][p]))
                im_array[l][p] = np.full(1, value, dtype=np.uint8)
        return im_array
