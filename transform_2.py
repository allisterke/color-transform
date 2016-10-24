import numpy as np
from scipy.misc import imread, imresize, imsave
from collections import OrderedDict

color_path = 'style_text.jpg'
content_path = 'content_text.jpg'
unit_pixel = 4
max_size = 128


def load_image(image_path):
    image = imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    if height >= width:
        width = float(width) / float(height) * max_size
        height = max_size
    else:
        height = float(height) / float(width) * max_size
        width = max_size
    image = imresize(image, [int(height), int(width)])

    color_dict_r = dict()
    color_dict_g = dict()
    color_dict_b = dict()

    color_idx_r = dict()
    color_idx_g = dict()
    color_idx_b = dict()

    color_idx_reverse_r = dict()
    color_idx_reverse_g = dict()
    color_idx_reverse_b = dict()

    height, width = image.shape[0], image.shape[1]
    print height, width

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

    color_dict_r = OrderedDict(sorted(color_dict_r.items(), key=lambda x: x[1], reverse=True))
    color_dict_g = OrderedDict(sorted(color_dict_g.items(), key=lambda x: x[1], reverse=True))
    color_dict_b = OrderedDict(sorted(color_dict_b.items(), key=lambda x: x[1], reverse=True))

    count = 0
    for color_r in color_dict_r:
        color_idx_r[color_r] = count
        color_idx_reverse_r[count] = color_r
        count += 1

    count = 0
    for color_g in color_dict_g:
        color_idx_g[color_g] = count
        color_idx_reverse_g[count] = color_g
        count += 1

    count = 0
    for color_b in color_dict_b:
        color_idx_b[color_b] = count
        color_idx_reverse_b[count] = color_b
        count += 1

    return image, color_idx_r, color_idx_reverse_r, color_idx_g, color_idx_reverse_g, color_idx_b, color_idx_reverse_b


def pixel_trans(image_channel, image_idx, color_idx_reverse):
    height = image_channel.shape[0]
    width = image_channel.shape[1]
    image_channel = image_channel.reshape([image_channel.size])
    image_num = len(image_idx)
    color_num = len(color_idx_reverse)
    num = min(image_num, color_num)
    for idx in range(image_channel.size):
        pixel = int(image_channel[idx]) // unit_pixel
        index = image_idx[pixel]
        if index >= num:
            continue
        image_channel[idx] = color_idx_reverse[index] * unit_pixel

    return image_channel.reshape([height, width, -1])


color_image, color_idx_r, color_idx_reverse_r, color_idx_g, color_idx_reverse_g, color_idx_b, color_idx_reverse_b = load_image(color_path)
content_image, content_idx_r, content_idx_reverse_r, content_idx_g, content_idx_reverse_g, content_idx_b, content_idx_reverse_b = load_image(content_path)

generate_image_r = pixel_trans(content_image[:, :, 0], content_idx_r, color_idx_reverse_r)
generate_image_g = pixel_trans(content_image[:, :, 1], content_idx_g, color_idx_reverse_g)
generate_image_b = pixel_trans(content_image[:, :, 2], content_idx_b, color_idx_reverse_b)

generate_image = np.concatenate([generate_image_r, generate_image_g, generate_image_b], 2)
imsave('./generate2.jpg', generate_image)

