import numpy as np
from PIL import Image
import os
from Kmeans import KMean


def calculate_image(image_name, check_image_name, rnd):
    image = Image.open(image_name).convert("RGB")
    # Image._show(image)

    arr_pixels = np.array(image, dtype=np.float64) / 255.0
    k_mean = KMean(rnd)
    pixels_by_clusters = k_mean.find_clusters(arr_pixels)

    im_array = k_mean.get_image_array(pixels_by_clusters)

    check_image = Image.open(check_image_name).convert("L")
    # Image._show(check_image)
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


def load_all_images():
    rnd = np.random.RandomState(1234)
    index = 0
    image_name = get_image_name(index)
    check_image_name = get_check_image_name(index)
    for i in range(1):
        avr_err = calculate_image(image_name, check_image_name, rnd)
        print "error %f" % avr_err

    # img_count = 50
    # for img_index in range(img_count):
    #     image_name = get_image_name(img_index)
    #     calculate_image(image_name)


if __name__ == '__main__':
    load_all_images()
