import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm
import numpy as np
from models.unet_sr import UNetSR
from src.dataset import ImageNet1000
from torch.utils.tensorboard import SummaryWriter


def train(train_data, val_data, batchsize, epochs=50, init_lr=10e-4, sigma=0):

    device = 'cuda:1'
    writer = SummaryWriter('runs/smoothtest_lr1_3_sub4')

    model = UNetSR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr, eps=0.00001)
    criterion = nn.MSELoss()
    
    scaler = torch.cuda.amp.GradScaler()
    
    train_dataloader = DataLoader(train_data, batch_size=batchsize, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, pin_memory=True)

    print("Starting training.\n")
    for epoch in range(epochs):
        loop = tqdm(train_dataloader)
        torch.cuda.empty_cache()
    
        model.train()

        train_loss = []
        train_psnr = []
        train_ssim = []

        for low_img, high_img in loop:
            optimizer.zero_grad()
            
            low_img = low_img.to(device)
            high_img = high_img.to(device)
            
            low_noisy = low_img + torch.randn_like(low_img, device='cuda:1') * sigma
            #low_noisy = torch.clip(low_noisy, 0., 1.)
                
            with torch.cuda.amp.autocast():
                output = model(low_noisy)
                loss = criterion(high_img, output)
                
                #output = torch.clip(output, 0., 1.)
                train_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            psnr = peak_signal_noise_ratio(output.detach().cpu(), high_img.detach().cpu())
            train_psnr.append(psnr)
            
            output = output.type(torch.float32)
            ssim = structural_similarity_index_measure(output.detach().cpu(), high_img.detach().cpu())
            train_ssim.append(ssim)

        # evaluate using validation set
        model.eval()
        with torch.no_grad():
            val_loss = []
            val_psnr = []
            val_ssim = []

            for low_img, high_img in val_dataloader:
                low_img = low_img.to(device)
                high_img = high_img.to(device)
                
                low_noisy = low_img + torch.randn_like(low_img, device='cuda:1') * sigma
                #low_noisy = torch.clip(low_noisy, 0., 1.)
                
                with torch.cuda.amp.autocast():
                    output = model(low_noisy)
                    loss = criterion(high_img, output)
                    
                    #output = torch.clip(output, 0., 1.)
                    val_loss.append(loss.item())
                
                psnr = peak_signal_noise_ratio(output.detach().cpu(), high_img.detach().cpu())
                val_psnr.append(psnr) 
                
                output = output.type(torch.float32)
                ssim = structural_similarity_index_measure(output.detach().cpu(), high_img.detach().cpu())
                val_ssim.append(ssim)
        
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        
        epoch_training_loss = sum(train_loss)/len(train_loss)
        epoch_training_psnr = sum(train_psnr)/len(train_psnr)
        epoch_training_ssim= sum(train_ssim)/len(train_ssim)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        epoch_val_psnr = sum(val_psnr)/len(val_psnr)
        epoch_val_ssim = sum(val_ssim)/len(val_ssim)
        
        writer.add_scalar('Training loss', epoch_training_loss, epoch)
        writer.add_scalar('Validation loss', epoch_val_loss, epoch)
        
        print(f"Epoch {epoch}\n \
            Training loss: {epoch_training_loss}\n \
            Training PSNR: {epoch_training_psnr}\n \
            Training SSIM: {epoch_training_ssim}\n \
            Validation loss: {epoch_val_loss}\n \
            Validation PSNR: {epoch_val_psnr}\n \
            Validation SSIM: {epoch_val_ssim}")
    
    writer.close()
    
    return model


if __name__=="__main__":
    os.chdir('/faidra/project')
    print(os.getcwd())
    train_dataset = ImageNet1000('./imagenet-mini/train', scaling_factor=2, use_cache=True)
    print(train_dataset.__len__())
    val_dataset = ImageNet1000('./imagenet-mini/val', scaling_factor=2, use_cache=True)
    print(val_dataset.__len__())
    
    model = train(train_dataset, val_dataset, batchsize=128, epochs=50, init_lr=1e-3, sigma=0.5)
    
    torch.save(model, "model_05_50.pt")
    