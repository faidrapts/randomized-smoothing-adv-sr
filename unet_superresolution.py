import torch
import torch.nn as nn


class TwoConvs(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(TwoConvs, self).__init__()

        self.two_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.two_convs(x)


class UNetSR(nn.Module):

    def __init__(self, in_channels=3, out_channels=3,
                 features=[8, 16, 32, 64]) -> None:
        super(UNetSR, self).__init__()
        self.upsampling = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling part of UNet
        for feature in features:
            self.downsampling.append(TwoConvs(in_channels, feature))
            in_channels = feature

        self.bottom = TwoConvs(features[3], features[3]*2)

        # Upsampling part of Unet
        for feature in reversed(features):
            self.upsampling.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.upsampling.append(TwoConvs(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # each: two convs, max pool
        for down_step in self.downsampling:
            x = down_step(x)
            skip_connections.append(x)
            x = self.maxpool(x)
            # print(f'Downstep {down_step}: {x.shape}')

        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.upsampling)//2):
            x = self.upsampling[2*i](x)
            skip_conn = skip_connections[(2*i)//2]
            # print(f'Upsampling {i}: {x.shape}')
            
            concat_connect = torch.cat((skip_conn, x), dim=1)
            # print(f'Concat {i}: {concat_connect.shape}')
            
            x = self.upsampling[2*i+1](concat_connect)

        return self.final_conv(x)
