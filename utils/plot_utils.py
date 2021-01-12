import torch
import numpy as np
from matplotlib import cm
import torchvision.utils as vutils


def add_spect_image(writer, spect, name, num, step, colorful=True):
    # input state: bb x tt x dd
    if len(spect.size()) == 4:
        spect = spect.permute(0, 2, 1, 3).reshapse(spect.size(0), spect.size[1], -1)
    bb, tt, dd = spect.size()

    num_per_row = min(bb, max(int(np.round(np.sqrt(bb * dd / tt))), 1))
    num = min(bb, int(np.round(num_per_row * tt / dd)) * num_per_row)
    x = spect.cpu().data.float()
    x = (x - x.min()) / (x.max() - x.min())
    if colorful:
        x = torch.Tensor(np.transpose(cm.get_cmap('jet')(x)[:num, :, :, :3], [0, 3, 2, 1]))
        # bb x 3 x dd x tt
    else:
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        # bb x 1 x dd x tt
    x = torch.flip(x, [2])
    plot_image = vutils.make_grid(x, num_per_row, padding=4, normalize=True, scale_each=True)
    writer.add_image(name, plot_image, step)