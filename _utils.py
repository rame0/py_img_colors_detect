import sys

import cv2
import numpy as np


def quantify_colors(img, k=32, attempts=5):
    """

    :param img: An array of N-Dimensional points with int coordinates is needed.
    :param k: int
    :param attempts: int
    """
    float_pixels = np.float32(img)
    float_pixels = float_pixels.reshape((float_pixels.shape[0] * float_pixels.shape[1], 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(float_pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    ret = center[label.flatten()]
    ret = ret.reshape(img.shape)
    return ret


# https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def plot_colors(data):
    """

    :param data: (cluster names, cluster colors, cluster histogram)
    :return:
    """
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (_, color, percent) in data:
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar


# https://stackoverflow.com/a/47542304/513723
# definitely read this ^^ answer it's very useful!
def distance_sq(left, right):
    """ Returns the square of the distance between left and right. """
    return (
            ((int(left[0]) - int(right[0])) ** 2) +
            ((int(left[1]) - int(right[1])) ** 2) +
            ((int(left[2]) - int(right[2])) ** 2)
    )


# https://stackoverflow.com/a/47542304/513723
# definitely read this ^^ answer it's very useful!
def distance(left, right):
    """ Returns the distance between left and right. """
    return (((int(left[0]) - int(right[0])) ** 2) +
            ((int(left[1]) - int(right[1])) ** 2) +
            ((int(left[2]) - int(right[2])) ** 2)
            ) ** .5


# https://stackoverflow.com/a/34325723/513723
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def new_image_size(orig_width, orig_height, smallest_side_size):
    new_w = orig_width
    new_h = orig_height
    if smallest_side_size <= 0:
        return new_w, new_h

    if orig_width > orig_height:
        new_h = smallest_side_size
        new_w = round(orig_width * new_h / orig_height)
    else:
        new_w = smallest_side_size
        new_h = round(orig_height * new_w / orig_width)

    return new_w, new_h
