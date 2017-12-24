import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# Example 1
# load a colour image and put it into a 3D numpy array
# name = "foo.png"
# im = Image.open(name).convert("RGB")
# arr = np.array(im, dtype=np.float64) / 255.0

# -------------------------------------------------
# Example 2
# assume that arr is an n-dimensional numpy array

# apply a function to all elements of the array
# parr = np.log(arr / (1.0 - arr))


# truncate the values in arr by some maxval
# arr[arr > maxval] = maxval

# ----------------------------------------------------
# Example 3
# class definition of a multivariate Gaussian (fixed dimension = 3)


class GaussMVD:
    """ Multivariate normal distribution """
    dim = 3

    # ===============================================
    def __init__(self):
        self.mean = np.zeros(GaussMVD.dim, dtype=np.float64)
        self.cov = np.eye(GaussMVD.dim, dtype=np.float64)
        self.det = np.linalg.det(np.matrix(self.cov))
        self.normc = 1.0 / np.sqrt(np.power((2.0 * np.pi), GaussMVD.dim) * self.det)

    # ===============================================
    def modify(self, mean, cov):
        if not ((mean.shape == (GaussMVD.dim,)) and (cov.shape == (GaussMVD.dim, GaussMVD.dim))):
            raise Exception("Gaussian: shape mismatch!")

        self.mean = np.array(mean, dtype=np.float64)
        self.cov = np.array(cov, dtype=np.float64)

        self.det = np.linalg.det(np.matrix(self.cov))
        self.normc = 1.0 / np.sqrt(np.power((2.0 * np.pi), GaussMVD.dim) * self.det)

        return None

    # ===============================================
    def compute_probs(self, arr_values):
        """ compute probabilities for an array of values """

        inv_cov = np.asarray(np.linalg.inv(np.matrix(self.cov)))
        darr = arr_values - self.mean
        varr = np.sum(darr * np.inner(darr, inv_cov), axis=-1)
        varr = - varr * 0.5
        varr = np.exp(varr) * self.normc

        return varr

    # ===============================================
    def estimate(self, arr_values, weight):
        """ estimate parameters from data (array of values & array of weights) """
        eweight = weight[..., np.newaxis]
        wsum = np.sum(weight)

        dimlist = list(range(len(arr_values.shape) - 1))
        dimtup = tuple(dimlist)

        # estimate mean
        mean = np.sum(eweight * arr_values, axis=dimtup) / wsum

        # estimate covariance
        darr = arr_values - mean
        cov = np.tensordot(darr, darr * eweight, axes=(dimlist, dimlist)) / wsum

        self.modify(mean, cov)

        return None

    # ===============================================
    def compute_distance(self, mvgd):
        """ Bhattacharyya distance """

        ccov = (self.cov + mvgd.cov) / 2.0
        inv_ccov = np.asarray(np.linalg.inv(np.matrix(ccov)))
        d_ccov = np.linalg.det(np.matrix(ccov))
        cmean = self.mean - mvgd.mean
        v1 = np.dot(cmean, np.tensordot(inv_ccov, cmean, 1)) / 8.0
        v2 = np.log(d_ccov / np.sqrt(self.det * mvgd.det)) / 2.0

        return v1 + v2

    # ===============================================
    def write(self):
        print("Mean, covariance:")
        print(repr(self.mean))
        print(repr(self.cov))


def test_pixel_array(arr_pixels):
    line_count = len(arr_pixels)
    for line_index in range(line_count):
        line = arr_pixels[line_index]
        pixel_count = len(line)
        for pixel_index in range(pixel_count):
            print(line[pixel_index])

class K_Mean:
    _dim = 3
    _cluster_count = 2

    def __init__(self, arr_pixels):
        # d = self.distance_p2(np.array([1, 1, 1]), np.array([3, 3, 3]))
        sh = np.shape(arr_pixels)
        self._row_count = sh[0]
        self._col_count = sh[1]
        self._array_pixels = arr_pixels

        self._means = np.zeros((self._cluster_count, self._dim))

        rng = np.random.RandomState(1234)
        for i in range(self._cluster_count):
            self._means[i] = rng.rand(self._dim)

        # self.distance_means_p2()
        pixels_by_clusters = self.select_by_clusters()
        self.calc_new_means(pixels_by_clusters)

    @staticmethod
    def distance_p2(v1, v2):
        v = v1 - v2
        v = v * v
        d = np.sum(v)
        return d

    def distance_means_p2(self):
        d = np.zeros((self._cluster_count, self._row_count, self._col_count))

        for l in range(self._row_count):
            for p in range(self._col_count):
                for k in range(self._cluster_count):
                    d[k][l][p] = self.distance_p2(self._array_pixels[l][p], self._means[k])
        return d

    def select_by_clusters(self):
        d = self.distance_means_p2()
        k = np.argmin(d, 0)
        return k

    def calc_new_means(self, pixels_by_clusters):
        for k in range(self._cluster_count):
            total = 0
            pixel_sum = np.zeros(self._dim)
            for l in range(self._row_count):
                for p in range(self._col_count):
                    if pixels_by_clusters[l][p] == k:
                        total += 1
                        pixel_sum += self._array_pixels[l][p]
            self._means[k] = pixel_sum / total

    def k_mean(self):

        print(self._means)


def load_all_images():
    img_count = 50
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    for img_index in range(img_count):
        if img_index < 10:
            image_name = curr_dir + "\\hand_0%s.png" % img_index
        else:
            image_name = curr_dir + "\\hand_%s.png" % img_index
        image = Image.open(image_name).convert("RGB")
        # Image._show(image)
        arr_pixels = np.array(image, dtype=np.float64) / 255.0
        K_Mean(arr_pixels)


if __name__ == '__main__':
    load_all_images()
