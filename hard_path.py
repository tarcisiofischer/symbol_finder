from math import ceil
from scipy.misc.pilutil import toimage

from imageio.core.functions import imread, imsave
from joblib import Parallel, delayed
from skimage import feature

import numpy as np

if __name__ == '__main__':
    im = imread('imgs/i1.png', 'png')
    edge_img = feature.canny(im[:, :, 0])
    imsave('edges.png', edge_img)

    a_img = np.array(
        imread('imgs/letras/a.png', 'png')[:, :, 0], dtype=np.bool)
    xx, yy = edge_img.shape
    letter_xx, letter_yy = a_img.shape

    def process_pixel(full_image, img_mx, img_my):
        ii_b = img_mx - ceil(letter_xx / 2)
        ii_e = img_mx + ceil(letter_xx / 2) + 1
        jj_b = img_my - ceil(letter_yy / 2)
        jj_e = img_my + ceil(letter_yy / 2) + 1
        return np.sum(full_image[ii_b:ii_e, jj_b:jj_e] == a_img)

    result = np.zeros(dtype=int, shape=edge_img.shape)

    def process_line(i):
        for j in xrange(letter_yy // 2, yy - letter_yy // 2):
            result[i, j] = process_pixel(edge_img, i, j)
        print(i)
    Parallel(n_jobs=8, backend='threading')(delayed(process_line)(i)
                                            for i in xrange(letter_xx // 2, xx - letter_xx // 2))

    letter_xx, letter_yy = a_img.shape

# Uncomment to render colored heatmap
#     cmap = plt.get_cmap('magma')
#     heat_img = cmap(result)
#     imsave('heatmap.png', heat_img)

    imsave('heatmap.png', result)