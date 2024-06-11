
import torch
import torch.nn as nn

import numpy as np
import os
import glob
import spectral

import torch.nn.functional as F
import random

from collections import OrderedDict

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)


category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4, "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,"rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}


class HsiMaterial():
    def __init__(self):

        category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                            "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                            "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}
        
        materials = glob.glob('materials_numpy/*.npy')
        materials = sorted(materials)


        self.materials = np.zeros((len(materials), 31))
        for i,m in enumerate(materials):
            self.materials[i,:] = np.load(m)

        self.code2material = {v: k for k, v in zip(materials, category2code.values())}
        self.num_bands = 31

    def convert(self, cube):
        """
        Convert a hyperspectral cube to a material cube

        Params:
            cube -> Hyperspectral cube (batch_size, height, width, 31)

        Returns:
            material_cube: Material cube (batch_size, height, width, 1)
        """
        cube = cube.transpose(1,2,0)
        #print(cube.shape, self.materials.shape)
        assert cube.shape[-1] == self.num_bands

        result_sam = spectral.algorithms.spectral_angles(cube, self.materials / 100)
        return np.argmin(result_sam, axis=2)
    

def new_fig():
    """Create a new matplotlib figure containing one axis"""
    fig = Figure()
    FigureCanvas(fig)
    axes = []  # Lista para almacenar los subplots

    # Agregar cada subplot a la lista
    axes.append(fig.add_subplot(131))
    axes.append(fig.add_subplot(132))
    axes.append(fig.add_subplot(133))

    return fig, axes

def make_plot_train(inputs, outputs, labels):
    fig, ax = new_fig()


    input_ = inputs[0].cpu().numpy().transpose(1,2,0)
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())

    ax[0].imshow(input_)
    ax[0].set_title("Input")

    ax[1].imshow(labels[0].cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[1].set_title("Ground Truth")

    out = F.softmax(outputs[0], dim=1)
    out = torch.argmax(out, dim=0)

    ax[2].imshow(out.detach().cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[2].set_title("Prediction")


    return fig


def make_plot_val(inputs, outputs, labels):
    fig, ax = new_fig()

    idx = random.randint(0, inputs.shape[0]-1)

    input_ = inputs[idx].cpu().numpy().transpose(1,2,0)
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())

    ax[0].imshow(input_)
    ax[0].set_title("Input")

    ax[1].imshow(labels[idx].cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[1].set_title("Ground Truth")

    out = F.softmax(outputs[idx], dim=0)
    out = out.argmax(dim=0)

    ax[2].imshow(out.detach().cpu().numpy().squeeze(), vmin=0, vmax=46)
    ax[2].set_title("Prediction")

    return fig


def process_statedict_dataparallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict