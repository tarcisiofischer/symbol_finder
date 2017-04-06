from itertools import izip
from scipy.misc.pilutil import imread, imsave
import numpy as np

a_img = imread('imgs/letras/a.png', 'png')
letter_xx, letter_yy = a_img.shape


r = imread('heatmap.png')
p = np.where(r > 190)
s = imread('imgs/i1.png')
ii, jj = p
r = 5
y, x = np.ogrid[-r: r + 1, -r: r + 1]
mask_xx, mask_yy = np.where(x**2 + y**2 <= r**2)
for (i, j) in izip(ii, jj):
    s[i + letter_xx // 2, j - letter_yy // 2:j + letter_yy // 2][:, 0:3] = 0
    s[i + letter_xx // 2, j - letter_yy // 2:j + letter_yy // 2][:, 0] = 255

    s[i - letter_xx // 2, j - letter_yy // 2:j + letter_yy // 2][:, 0:3] = 0
    s[i - letter_xx // 2, j - letter_yy // 2:j + letter_yy // 2][:, 0] = 255

    s[i - letter_xx // 2:i + letter_xx // 2, j - letter_yy // 2][:, 0:3] = 0
    s[i - letter_xx // 2:i + letter_xx // 2, j - letter_yy // 2][:, 0] = 255

    s[i - letter_xx // 2:i + letter_xx // 2, j + letter_yy // 2][:, 0:3] = 0
    s[i - letter_xx // 2:i + letter_xx // 2, j + letter_yy // 2][:, 0] = 255

imsave('final_result.png', s)
