import numpy as np
from PIL import Image
import os
from Kmeans import CKMean
import MixedGauss


def get_image_array(pixels_by_clusters):
    sh = np.shape(pixels_by_clusters)
    row_count = sh[0]
    col_count = sh[1]
    im_array = np.zeros((row_count, col_count), dtype=np.uint8)
    for l in range(row_count):
        for p in range(col_count):
            value = 255 * int(round(pixels_by_clusters[l][p]))
            im_array[l][p] = np.full(1, value, dtype=np.uint8)
    return im_array


def calculate(image_name, check_image_name, calulator):
    image = Image.open(image_name).convert("RGB")
    # Image._show(image)

    arr_pixels = np.array(image, dtype=np.float64) / 255.0
    pixels_by_clusters = calulator.find_clusters(arr_pixels)

    im_array = get_image_array(pixels_by_clusters)

    check_image = Image.open(check_image_name).convert("L")
    Image._show(check_image)
    check_arr_pixels = np.array(check_image, dtype=np.int32)

    err = check_arr_pixels - im_array
    err_shape = np.shape(err)
    avr_err = np.sum(np.abs(err)) / (err_shape[0] * err_shape[1]) / 255.0
    if avr_err > 0.5:
        avr_err = 1 - avr_err
    avr_err = avr_err / 0.5

    res_im = Image.fromarray(im_array, "L")
    Image._show(res_im)

    return avr_err


def calculate_k_mean(image_name, check_image_name, rnd):
    k_mean = CKMean(rnd)
    return calculate(image_name, check_image_name, k_mean)


def get_image_name(img_index):
    curr_dir = os.path.dirname(os.path.abspath(__file__)) + "\\Images\\"
    if img_index < 10:
        image_name = curr_dir + "hand_0%s.png" % img_index
    else:
        image_name = curr_dir + "hand_%s.png" % img_index
    return image_name


def get_check_image_name(img_index):
    curr_dir = os.path.dirname(os.path.abspath(__file__)) + "\\Images\\"
    if img_index < 10:
        image_name = curr_dir + "hand_0%s_seg.png" % img_index
    else:
        image_name = curr_dir + "hand_%s_seg.png" % img_index
    return image_name


def calculate_gauss(image_name, check_image_name, rnd):
    mix_gauss = MixedGauss.CMixedGauss(rnd)
    return calculate(image_name, check_image_name, mix_gauss)


def test(calc_type):
    rnd = np.random.RandomState(1234)
    index = 47
    image_name = get_image_name(index)
    check_image_name = get_check_image_name(index)
    for i in range(1):
        avr_err = 10000
        if calc_type == "k_mean":
            avr_err = calculate_k_mean(image_name, check_image_name, rnd)
        if calc_type == "mix_gauss":
            avr_err = calculate_gauss(image_name, check_image_name, rnd)
        print "error %f" % avr_err

    # img_count = 50
    # for img_index in range(img_count):
    #     image_name = get_image_name(img_index)
    #     calculate_image(image_name)


if __name__ == '__main__':
    test("mix_gauss")
    test("k_mean")
