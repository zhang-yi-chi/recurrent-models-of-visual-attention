import matplotlib
matplotlib.use('agg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

def get_glimpse(x, l, output_size, k):
    """Transform image to retina representation

        Assume that width = height and channel = 1
    """
    batch_size, input_size = x.size(0), x.size(2) - 1
    assert output_size * 2**(k - 1) <= input_size, \
        "output_size * 2**(k-1) should smaller than or equal to input_size"

    # construct theta for affine transformation
    theta = torch.zeros(batch_size, 2, 3)
    theta[:, :, 2] = l

    scale = output_size / input_size
    osize = torch.Size([batch_size, 1, output_size, output_size])

    output = torch.zeros(batch_size, output_size * output_size * k)

    for i in range(k):
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        grid = F.affine_grid(theta, osize)
        glimpse = F.grid_sample(x, grid).view(batch_size, -1)
        output[:, i * output_size *
               output_size:(i + 1) * output_size * output_size] = glimpse
        scale *= 2
    return output.detach()


def draw_locations(image, locations, size=8, epoch=0):
    locations = list(locations)
    fig, ax = plt.subplots(1, len(locations))
    for i, location in enumerate(locations):
        if len(locations) == 1:
            subplot = ax
        else:
            subplot = ax[i]
        subplot.axis('off')
        subplot.imshow(image, cmap='gray')
        loc = ((location[0] + 1) * image.shape[1] / 2 - size / 2,
           (location[1] + 1) * image.shape[0] / 2 - size / 2)
        # print(location, loc)
        rect = patches.Rectangle(
            loc, size, size, linewidth=1, edgecolor='r', facecolor='none')
        subplot.add_patch(rect)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.savefig('results/glimpse_%d.png'%epoch, bbox_inches='tight')
    plt.close()
