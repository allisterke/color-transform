from __future__ import division
import numpy as np
from scipy.misc import imread, imresize, imsave
from collections import OrderedDict

color_path = 'image2.jpeg'
content_path = 'image1.jpeg'
color_unit_pixel = 8
content_unit_pixel = 8
max_size = 512


def channel_to_rgb(image, unit_pixel):
    pixel_num = 256 // unit_pixel
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


def rgb_to_channel(rgb, unit_pixel):
    pixel_num = 256 // unit_pixel

    channel_r = int(rgb) // (pixel_num * pixel_num)
    channel_g = (int(rgb) - (channel_r * pixel_num * pixel_num)) // pixel_num
    channel_b = int(rgb) - (channel_r * pixel_num * pixel_num) - (channel_g * pixel_num)
    # image[idx] = np.array([channel_r * unit_pixel, channel_g * unit_pixel, channel_b * unit_pixel])
    return np.array([channel_r, channel_g, channel_b])


def load_image(image_path, unit_pixel):
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

    image_rgb = channel_to_rgb(image, unit_pixel)
    color_dict = dict()
    color_perc = list()
    color_pixel = list()

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

    color_dict = to_percent(color_dict, image_rgb.size)
    color_dict = OrderedDict(sorted(color_dict.items(), key=lambda x: x[1], reverse=True))

    count = 0
    for rgb in color_dict:
        # print rgb
        color_perc.append(color_dict[rgb])
        color_pixel.append(rgb)
        count += 1

    return image_rgb, color_dict, color_perc, color_pixel


def to_percent(image_dict, size):
    for (d, x) in image_dict.items():
        image_dict[d] = float(x) / float(size)
    return image_dict


def pixel_trans(image_rgb, image_dict, color_perc, color_pixel):
    height = image_rgb.shape[0]
    width = image_rgb.shape[1]
    image_rgb = image_rgb.reshape([image_rgb.size])
    generate_image = np.array([0, 0, 0] * image_rgb.size).reshape(image_rgb.size, 3)

    for idx in range(image_rgb.size):
        rgb = int(image_rgb[idx])
        perc = image_dict[rgb]
        index = 0
        while index < len(color_perc):
            cur_perc = color_perc[index]
            if perc == cur_perc:
                break
            elif perc > cur_perc:
                if index != 0:
                    pre_perc = color_perc[index - 1]
                    if pre_perc - perc < perc - cur_perc:
                        index -= 1
                break
            else:
                index += 1
        if index == len(color_perc):
            pre_perc = color_perc[index - 1]
            if perc < pre_perc - perc:
                channel = rgb_to_channel(rgb, content_unit_pixel)
                generate_image[idx] = channel
                continue
            else:
                index -= 1

        new_pixel = color_pixel[index]
        channel = rgb_to_channel(new_pixel, color_unit_pixel)
        generate_image[idx] = channel
    generate_image = np.reshape(generate_image, [height, width, 3])

    return generate_image


def transform():
    color_rgb, color_dict, color_perc, color_pixel = load_image(color_path, color_unit_pixel)
    content_rgb, content_dict, content_perc, content_pixel = load_image(content_path, content_unit_pixel)

    generate_image = pixel_trans(content_rgb, content_dict, color_perc, color_pixel)
    # generate_image = rgb_to_channel(generate_image, 1)
    imsave('./generate_content.jpg', generate_image)


def test():
    a = np.array([[[1,2,3],[11, 12, 13]],[[21,22,23],[31,32,33]]])
    b = channel_to_rgb(a, 1)
    c = rgb_to_channel(b, 1)
    print a
    print b
    print c


transform()