import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import os
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter

from models.unet_sr import UNetSR
from adv import AttackerModel
from src.dataset import ImageNet1000


def train(dataloader, model, device, loss_fn, optimizer, make_adv, scaler, **attack_kwargs):
    model.train()
    loss_total = 0
    for _, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            _, pred = model(X, target=y,
                            make_adv=make_adv, **attack_kwargs)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_total += loss.item()
        
    return loss_total / len(dataloader)


def test(dataloader, model, device, loss_fn, make_adv, **attack_kwargs):
    model.eval()
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            _, pred = model(X, target=y, make_adv=make_adv,
                            **attack_kwargs)
            test_loss += loss_fn(pred, y).item()
    return test_loss / len(dataloader)


def perform_training(epochs, train_adversarially, **attack_kwargs):

    writer = SummaryWriter('runs/adv005_lr1_3_batch1')
    os.chdir('/faidra/project')

    device = 'cuda:1'
    dataset_train = ImageNet1000(
        './imagenet-mini/train', scaling_factor=2, use_cache=True)
    dataset_test = ImageNet1000('./imagenet-mini/val', scaling_factor=2, use_cache=True)
    dataset_train_loader = DataLoader(dataset_train, batch_size=128, pin_memory=True)
    dataset_test_loader = DataLoader(dataset_test, batch_size=128, pin_memory=True)

    train_loss_fn = torch.nn.MSELoss()
    test_loss_fn = torch.nn.MSELoss()

    model = UNetSR().to(device)
    attackerModel = AttackerModel(model)
    optimizer = optim.Adam(attackerModel.parameters(), lr=1e-3)
    loss_train_values = []
    loss_test_values = []
    loss_test_values_adv = []
    
    scaler = torch.cuda.amp.GradScaler()
    
    for t in tqdm.trange(epochs, desc=f"Training", unit="epoch", position=0):
        torch.cuda.empty_cache()

        train_loss = train(dataset_train_loader, attackerModel, device,
                                 train_loss_fn, optimizer, train_adversarially, scaler, **attack_kwargs)
        print("Training loss: ", train_loss)
    
        writer.add_scalar('Training loss', train_loss, t)
        # one could do experiment tracking here (e.g. tensorboard)
        loss_train_values.append(train_loss)
        test_loss = test(
            dataset_test_loader, attackerModel, device, test_loss_fn, False, **attack_kwargs)
        loss_test_values.append(test_loss)
        print("Test loss: ", test_loss)
        writer.add_scalar('Validation loss', test_loss, t)
        test_loss_adv = test(
            dataset_test_loader, attackerModel, device, test_loss_fn, True, **attack_kwargs)
        loss_test_values_adv.append(test_loss_adv)
        print("Adversarial test loss: ", test_loss_adv)
        writer.add_scalar('Adversarial test loss', test_loss_adv, t)

    writer.close()
    return model, loss_train_values, loss_test_values, loss_test_values_adv


if __name__ == "__main__":

    epochs = 50

    os.chdir('/faidra/project')
    dataset_test_loader = ImageNet1000('./imagenet-mini/val', scaling_factor=2, use_cache=True)
    lr, hr = dataset_test_loader[0]
    
    # the dimensionality is later used to scale the adversarial perturbations.
    n = np.prod(lr.shape)
    
    # will be multiplied by math.sqrt(n), where n is the image dimension (e.g. 3*32*32)
    eps_rel = 0.05
    
    # number of PGD iterations; with value 1 it is a variant of FastFGSM (not necessarily sufficiently strong attack)
    adv_iterations = 5

    # specific attack settings defined below
    # needs to be adapted for different settings (e.g. for classification one usually needs restarts and more iterations)
    eps = eps_rel * math.sqrt(n)  # eps_rel * 277

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

    model_adv, train_values_adv, test_values_adv_std, test_values_adv_adv = perform_training(
        epochs=epochs, train_adversarially=True, **attack_kwargs01)
    
    torch.save(model_adv, f"model_adv_{eps_rel}.pt")
