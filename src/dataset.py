import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image


class ImageNet1000(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scaling_factor, use_cache=False) -> None:
        super(ImageNet1000, self).__init__()
        self.dataset_path = dataset_path
        self.use_cache = use_cache

        image_paths = []
        for folder in os.listdir(dataset_path):
            for img in os.listdir(dataset_path+'/'+folder):
                image_paths.append(dataset_path+'/'+folder+'/'+img)

        self.image_paths = pd.DataFrame(image_paths)
        self.pre_transform = transforms.Compose([
            transforms.CenterCrop((160, 160))
        ])
        self.high_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4882, 0.4431, 0.3946],
                                 std=[0.2777, 0.2665, 0.2739])
        ])

        self.low_transform = transforms.Compose([
            transforms.Resize(int(160/scaling_factor), antialias=True),
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4882, 0.4431, 0.3946],
                                 std=[0.2777, 0.2665, 0.2739])
        ])

        if self.use_cache:
            self.cache_lr = []
            self.cache_hr = []

            print("Loading dataset into cache...")
            for item in self.image_paths[0]:
                img = Image.open(item).convert("RGB")
                
                img = self.pre_transform(img)
                lr_img = self.low_transform(img)
                hr_img = self.high_transform(img)

                self.cache_lr.append(lr_img)
                self.cache_hr.append(hr_img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.cache_lr[index], self.cache_hr[index]


if __name__ == "__main__":

    os.chdir('/faidra/project')
    dataset = ImageNet1000('./imagenet-mini/train', scaling_factor=2, use_cache=True)
    low_img, high_img = dataset[10]
    print("Number of images in dataset: ", len(dataset))
    to_pil = transforms.ToPILImage()
    # to_pil(low_img).show()
