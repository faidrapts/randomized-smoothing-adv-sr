"""
This script was used to generate series of figures (comparison of images) for the thesis 
in order to demonstrate randomized smoothing's performance under adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm, trange
import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from models.unet_sr import UNetSR
from src.dataset import ImageNet1000
from smoothened_estimate import SmoothenedModel
from adv import AttackerModel


def test(dataloader, model, device, loss_fn, make_adv, **attack_kwargs):
    model.eval()
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        _, pred = model(X, target=y, make_adv=make_adv, **attack_kwargs)
        test_loss += loss_fn(pred, y).item()
    return test_loss / len(dataloader)


def perform_evaluation(model_name, test_adversaraially, dataset_test_loader, device, sigma, **attack_kwargs):
    test_loss_fn = torch.nn.MSELoss()
    model = torch.load(model_name).to(device)

    smoothing_model = SmoothenedModel(model, sigma)
    attackerModel = AttackerModel(model)

    return test(dataset_test_loader, attackerModel, device, test_loss_fn, test_adversaraially, **attack_kwargs)


def seek_perturbation(model, test_img, target_img, make_adv, **attack_kwargs):
    test_loss_fn = torch.nn.MSELoss()
    attackerModel = AttackerModel(model)
    new_input, prediction = attackerModel(
        test_img, target=target_img, make_adv=make_adv, **attack_kwargs)
    return new_input, prediction


def tensor_to_imshow(x):
    normalize_inverse = transforms.Normalize(mean=[-0.4882/0.2777, -0.4431/0.2665, -0.3946/0.2739],
                                             std=[1/0.2777, 1/0.2665, 1/0.2739])
    x_numpy = np.array(normalize_inverse(x).detach().cpu().permute(1,2,0).clamp(0, 1))
    return x_numpy


if __name__ == "__main__":
    os.chdir('/faidra/project')
    test_data = ImageNet1000('./imagenet-mini/val', use_cache=True)
    test_dataloader = DataLoader(
        test_data, batch_size=1, pin_memory=True, shuffle=False)

    device = 'cuda:1'
    model = torch.load('model_01_50epochs_lr13.pt').to(device)
    sigma = 0.1

    smoothing_model = SmoothenedModel(model, sigma)
    normalize_inverse = transforms.Normalize(mean=[-0.4882/0.2777, -0.4431/0.2665, -0.3946/0.2739],
                                             std=[1/0.2777, 1/0.2665, 1/0.2739])

    lr, hr = test_data[0]
    # the dimensionality is later used to scale the adversarial perturbations.
    #print(lr.shape)
    n = np.prod(lr.shape)

    # will be multiplied by math.sqrt(n), where n is the image dimension (e.g. 3*32*32)
    eps_rel = 0.1
    # number of PGD iterations; with value 1 it is a variant of FastFGSM (not necessarily sufficiently strong attack)
    adv_iterations = 10

    # specific attack settings defined below
    # needs to be adapted for different settings (e.g. for classification one usually needs restarts and more iterations)
    eps = eps_rel * math.sqrt(n) # eps_rel * 277
    attack_kwargs01 = {
        'constraint': "2",
        'eps': eps,
        'step_size': 2.5 * (eps / adv_iterations),
        'iterations': adv_iterations,
        'random_start': True,
        'random_restarts': 0,
        'use_best': False,
        'random_mode': "uniform_in_sphere"
    }

    model.eval()
    clean_psnr = []
    base_psnr = []
    smooth_psnr = []

    clean_ssim = []
    base_ssim = []
    smooth_ssim = []

    sr_imgs = []
    smooth_sr_imgs = []
    lr_imgs = []
    hr_imgs = []
    low_raw_imgs = []
    low_perts = []

    for idx, (low_img, high_img) in enumerate(test_dataloader):
        low_img = low_img.to(device)
        high_img = high_img.to(device)

        adv_perturbed_input, base_model_pred = seek_perturbation(
            model, low_img, high_img, make_adv=True, **attack_kwargs01)
        #adv_perturbed_input = adv_perturbed_input.clamp(-1.7580, 2.2103)
        
        print("PSNR of original and perturbed input: ", peak_signal_noise_ratio(adv_perturbed_input.detach().cpu(), low_img.detach().cpu()))
        print("SSIM of original and perturbed input: ", structural_similarity_index_measure(adv_perturbed_input.detach().cpu(), low_img.detach().cpu()))
        no_adv_output = model(low_img)

        psnr = peak_signal_noise_ratio(
            no_adv_output.detach().cpu(), high_img.detach().cpu())
        ssim = structural_similarity_index_measure(
            no_adv_output.detach().cpu(), high_img.detach().cpu())
        clean_psnr.append(psnr)
        clean_ssim.append(ssim)

        smooth = smoothing_model.smoothened_prediction(
            adv_perturbed_input, num_e=50)
        #smooth = torch.clamp(smooth,-1.75,2.2)

        psnr = peak_signal_noise_ratio(
            base_model_pred.detach().cpu(), high_img.detach().cpu())
        base_psnr.append(psnr)
        psnr_smooth = peak_signal_noise_ratio(
            smooth.detach().cpu(), normalize_inverse(high_img).detach().cpu())
        smooth_psnr.append(psnr_smooth)

        ssim = structural_similarity_index_measure(
            base_model_pred.detach().cpu(), high_img.detach().cpu())
        base_ssim.append(ssim)
        ssim_smooth = structural_similarity_index_measure(
            smooth.detach().cpu(), normalize_inverse(high_img).detach().cpu())
        smooth_ssim.append(ssim_smooth)

        if idx < 15:
            lr_imgs.append(low_img[0])
            low_perts.append(adv_perturbed_input[0])
            hr_imgs.append(high_img[0])
            sr_imgs.append(base_model_pred[0])
            smooth_sr_imgs.append(smooth[0])

        else:
            break

    print(f"Clean base model output PSNR:",
          sum(clean_psnr)/len(clean_psnr))
    print(f"Clean base model output SSIM:",
          sum(clean_ssim)/len(clean_ssim))
    print("Base PSNR:", sum(base_psnr)/len(base_psnr))
    print("Base SSIM:", sum(base_ssim)/len(base_ssim))
    print("PSNR smooth:", sum(smooth_psnr)/len(smooth_psnr))
    print("SSIM smooth:", sum(smooth_ssim)/len(smooth_ssim))

    fig = plt.figure(figsize=(20, 100))
    grid = gridspec.GridSpec(15, 5)

    for i in range(15):
        ax1 = plt.subplot(grid[i, 0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2 = plt.subplot(grid[i, 1])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax3 = plt.subplot(grid[i, 2])
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax4 = plt.subplot(grid[i, 3])
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax5 = plt.subplot(grid[i, 4])
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)

        ax1.set_title('Low resolution', fontsize=10)
        ax2.set_title(f'Low resolution adversarially perturbed', fontsize=10)
        ax3.set_title('Super resolution', fontsize=10)
        ax4.set_title('Smoothened SR', fontsize=10)
        ax5.set_title('High resolution', fontsize=10)
        
        ax1.imshow(tensor_to_imshow(lr_imgs[i]))
        ax2.imshow(tensor_to_imshow(low_perts[i]))
        ax3.imshow(tensor_to_imshow(sr_imgs[i]))
        ax4.imshow(np.array(smooth_sr_imgs[i].detach().cpu().permute(1,2,0).clip(0.,1.)))
        ax5.imshow(tensor_to_imshow(hr_imgs[i]))

    plt.savefig('visual01_50_01.pdf')
