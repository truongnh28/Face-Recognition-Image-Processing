import math
import cv2
import numpy as np
from PIL import Image


def gauss_kernel(size, k, sigma):
    gauss_kernel_matrix = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            norm = math.pow(i - k, 2) + pow(j - k, 2)
            gauss_kernel_matrix[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2))) / 2 * math.pi * pow(sigma, 2)
    sum = np.sum(gauss_kernel_matrix)
    kernel = gauss_kernel_matrix / sum
    return kernel


def gauss_filter(img_gray, kernel):
    h, w = img_gray.shape
    k_h, k_w = kernel.shape
    for i in range(int(k_h / 2), h - int(k_h / 2)):
        for j in range(int(k_h / 2), w - int(k_h / 2)):
            sum = 0
            for k in range(0, k_h):
                for l in range(0, k_h):
                    sum += img_gray[i - int(k_h / 2) + k, j - int(k_h / 2) + l] * kernel[k, l]
            img_gray[i, j] = sum
    return img_gray


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))


def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x = row - (diameter / 2 - k)
                    n_y = col - (diameter / 2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(np.round(filtered_image))
    return new_image


if __name__ == '__main__':
    img = cv2.imread("src/data/img.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_g = img_gray.copy()
    # k = 1
    # size = 2 * k + 1
    # kernel = gauss_kernel(size, k, 1.5)
    #
    # img_B, img_G, img_R = cv2.split(img)
    # img_gauss_B = gauss_filter(img_B, kernel)
    # img_gauss_G = gauss_filter(img_G, kernel)
    # img_gauss_R = gauss_filter(img_R, kernel)
    # img_gauss = cv2.merge([img_gauss_B, img_gauss_G, img_gauss_R])
    # print(img_g)
    # img_median = median_filter(img_g, 3)
    # # img_comp = np.hstack((img, img_median))
    # cv2.imshow("gauss", img_median)
    # cv2.waitKey(0)
    # img = Image.open("data/median.png").convert(
    #     "L")
    # arr = np.array(img)
    # removed_noise = median_filter(img_gray, 5)
    img_bilateral = bilateral_filter(img, 7, 20.0, 20.0)
    cv2.imshow("cc", img_bilateral)
    img = Image.fromarray(img_bilateral)
    img.show()
    cv2.waitKey(0)
