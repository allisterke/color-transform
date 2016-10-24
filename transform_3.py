import numpy as np
from scipy.misc import imread, imresize, imsave
from collections import OrderedDict

color_path = '3.jpg'
content_path = 'content_text.jpg'
unit_pixel = 4
max_size = 512


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

    color_dict_r = dict()
    color_dict_g = dict()
    color_dict_b = dict()

    color_perc_r = list()
    color_perc_g = list()
    color_perc_b = list()

    color_pixel_r = list()
    color_pixel_g = list()
    color_pixel_b = list()

    size = height * width
    pixels = image.reshape([size, -1])
    for idx in range(size):
        pixel_r = pixels[idx][0] // unit_pixel
        pixel_g = pixels[idx][1] // unit_pixel
        pixel_b = pixels[idx][2] // unit_pixel

        if pixel_r in color_dict_r.keys():
            color_dict_r[pixel_r] += 1
        else:
            color_dict_r[pixel_r] = 1

        if pixel_g in color_dict_g.keys():
            color_dict_g[pixel_g] += 1
        else:
            color_dict_g[pixel_g] = 1

        if pixel_b in color_dict_b.keys():
            color_dict_b[pixel_b] += 1
        else:
            color_dict_b[pixel_b] = 1

        if idx % 10000 == 0:
            print "%d pixels complete. " % idx

    color_dict_r = to_percent(color_dict_r, size)
    color_dict_g = to_percent(color_dict_g, size)
    color_dict_b = to_percent(color_dict_b, size)

    color_dict_r = OrderedDict(sorted(color_dict_r.items(), key=lambda x: x[1], reverse=True))
    color_dict_g = OrderedDict(sorted(color_dict_g.items(), key=lambda x: x[1], reverse=True))
    color_dict_b = OrderedDict(sorted(color_dict_b.items(), key=lambda x: x[1], reverse=True))

    for color_r in color_dict_r:
        color_perc_r.append(color_dict_r[color_r])
        color_pixel_r.append(color_r)

    for color_g in color_dict_g:
        color_perc_g.append(color_dict_g[color_g])
        color_pixel_g.append(color_g)

    for color_b in color_dict_b:
        color_perc_b.append(color_dict_b[color_b])
        color_pixel_b.append(color_b)

    return image, color_dict_r, color_perc_r, color_pixel_r, color_dict_g, color_perc_g, color_pixel_g, color_dict_b, color_perc_b, color_pixel_b


def to_percent(image_dict, size):
    for (d, x) in image_dict.items():
        image_dict[d] = float(x) / float(size)
    return image_dict


def pixel_trans(image_channel, image_dict, color_perc, color_pixel):
    height = image_channel.shape[0]
    width = image_channel.shape[1]
    image_channel = image_channel.reshape([image_channel.size])

    for idx in range(image_channel.size):
        pixel = int(image_channel[idx]) // unit_pixel
        perc = image_dict[pixel]
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
                continue
            else:
                index -= 1
        new_pixel = color_pixel[index]
        image_channel[idx] = new_pixel * unit_pixel

    return image_channel.reshape([height, width, -1])


_, _, color_perc_r, color_pixel_r, _, color_perc_g, color_pixel_g, _, color_perc_b, color_pixel_b = load_image(color_path)
content_image, content_dict_r, _, _, content_dict_g, _, _, content_dict_b, _, _ = load_image(content_path)

generate_image_r = pixel_trans(content_image[:, :, 0], content_dict_r, color_perc_r, color_pixel_r)
generate_image_g = pixel_trans(content_image[:, :, 1], content_dict_g, color_perc_g, color_pixel_g)
generate_image_b = pixel_trans(content_image[:, :, 2], content_dict_b, color_perc_b, color_pixel_b)

generate_image = np.concatenate([generate_image_r, generate_image_g, generate_image_b], 2)
imsave('./generate2.jpg', generate_image)

