from __future__ import division
import numpy as np
from scipy.misc import imread, imresize, imsave
from collections import OrderedDict

color_path = '1.jpeg'
content_path = 'content_text.jpg'
unit_pixel = 16
pixel_num = 256 // unit_pixel
max_size = 512


def channel_to_rgb(image):
    height = image.shape[0]
    width = image.shape[1]
    channel_num = image.shape[2]
    image_list = image.reshape([-1, channel_num])
    image_rgb = np.array([0] * (height * width))
    for idx in range(height * width):
        channel = image_list[idx]
        channel_r = int(channel[0]) // unit_pixel
        channel_g = int(channel[1]) // unit_pixel
        channel_b = int(channel[2]) // unit_pixel
        rgb = channel_r * pixel_num * pixel_num + channel_g * pixel_num + channel_b
        image_rgb[idx] = rgb
    return image_rgb.reshape([height, width])


def rgb_to_channel(image_rgb):
    height = image_rgb.shape[0]
    width = image_rgb.shape[1]
    image_rgb = image_rgb.reshape([image_rgb.size])
    image = np.array([0, 0, 0] * height * width).reshape([height * width, 3])
    for idx in range(height * width):
        rgb = image_rgb[idx]
        channel_r = int(rgb) // (pixel_num * pixel_num)
        channel_g = (int(rgb) - (channel_r * pixel_num * pixel_num)) // pixel_num
        channel_b = int(rgb) - (channel_r * pixel_num * pixel_num) - (channel_g * pixel_num)
        # image[idx] = np.array([channel_r * unit_pixel, channel_g * unit_pixel, channel_b * unit_pixel])
        image[idx][0] = channel_r * unit_pixel
        image[idx][1] = channel_g * unit_pixel
        image[idx][2] = channel_b * unit_pixel
    return image.reshape([height, width, 3])


def load_image(image_path):
    image = imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    if height >= width:
        width = int(float(width) / float(height) * max_size)
        height = max_size
    else:
        height = int(float(height) / float(width) * max_size)
        width = max_size
    image = imresize(image, [height, width])

    image_rgb = channel_to_rgb(image)
    color_dict = dict()
    color_idx = dict()
    color_idx_reverse = dict()

    image_list = np.reshape(image_rgb, [image_rgb.size])
    # print image_list[0]
    for idx in range(image_list.size):
        rgb = image_list[idx]

        if rgb in color_dict.keys():
            color_dict[rgb] += 1
        else:
            color_dict[rgb] = 1

        if idx % 20000 == 0:
            print "%d pixels complete. " % idx

    color_dict = OrderedDict(sorted(color_dict.items(), key=lambda x: x[1], reverse=True))

    count = 0
    for rgb in color_dict:
        # print rgb
        color_idx[rgb] = count
        color_idx_reverse[count] = rgb
        count += 1

    return image_rgb, color_idx, color_idx_reverse


def pixel_trans(image_rgb, image_idx, color_idx_reverse):
    height = image_rgb.shape[0]
    width = image_rgb.shape[1]
    image_rgb = image_rgb.reshape([image_rgb.size])
    generate_image = np.array([0] * image_rgb.size)

    image_num = len(image_idx)
    color_num = len(color_idx_reverse)
    num = min(image_num, color_num)
    for idx in range(image_rgb.size):
        rgb = int(image_rgb[idx])
        index = image_idx[rgb]
        if index >= num:
            generate_image[idx] = rgb
            continue
        generate_image[idx] = color_idx_reverse[index]
    generate_image = np.reshape(generate_image, [height, width])

    return generate_image


def transform():
    color_rgb, color_idx, color_idx_reverse = load_image(color_path)
    content_rgb, content_idx, content_idx_reverse = load_image(content_path)

    generate_image = pixel_trans(content_rgb, content_idx, color_idx_reverse)
    generate_image = rgb_to_channel(generate_image)
    imsave('./generate_content.jpg', generate_image)


def test():
    a = np.array([[[1,2,3],[11, 12, 13]],[[21,22,23],[31,32,33]]])
    b = channel_to_rgb(a)
    c = rgb_to_channel(b)
    print a
    print b
    print c


transform()