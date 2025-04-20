import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np


class SmoothenedModel(nn.Module):

    def __init__(self, base_model: torch.nn.Module, sigma, device) -> None:
        super(SmoothenedModel, self).__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.device = device

    def forward(self, x, num_e=50):
        torch.cuda.empty_cache()
        
        # use mean and std of ImageNet-1000 dataset
        normalize_inverse = transforms.Normalize(mean=[-0.4882/0.2777, -0.4431/0.2665, -0.3946/0.2739],
                                                 std=[1/0.2777, 1/0.2665, 1/0.2739])

        smooth_pred = torch.zeros_like(x, device=self.device)
        x = x.to(device=self.device)

        with torch.cuda.amp.autocast():
            for _ in range(num_e):
                torch.cuda.empty_cache()
                e = torch.randn_like(x, device=self.device) * self.sigma
                input = x + e
                pred = self.base_model(input)
                pred = torch.clamp(pred, -1.75, 2.2)
                smooth_pred += normalize_inverse(pred)

        smooth_pred = torch.div(smooth_pred, num_e)
        smooth_pred = torch.clamp(smooth_pred, 0., 1.)

        return smooth_pred
