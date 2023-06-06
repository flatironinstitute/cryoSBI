import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from cryo_sbi.utils.image_utils import LowPassFilter, Mask


EMBEDDING_NETS = {}


def add_embedding(name):
    """
    Add embedding net to EMBEDDING_NETS dict

    Args:
        name (str): name of embedding net

    Returns:
        add (function): function to add embedding net to EMBEDDING_NETS dict
    """

    def add(class_):
        EMBEDDING_NETS[name] = class_
        return class_

    return add


@add_embedding("RESNET18")
class ResNet18_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_Encoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET50")
class ResNet50_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet50_Encoder, self).__init__()

        self.resnet = models.resnet50()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("RESNET101")
class ResNet101_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet101_Encoder, self).__init__()

        self.resnet = models.resnet101()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("CONVNET")
class ConvNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ConvNet_Encoder, self).__init__()

        self.convnet = models.convnext_tiny()
        self.convnet.features[0][0] = nn.Conv2d(
            1, 96, kernel_size=(4, 4), stride=(4, 4)
        )
        self.convnet.classifier[2] = nn.Linear(
            in_features=768, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convnet(x)
        return x


@add_embedding("CONVNET")
class RegNetX_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(RegNetX_Encoder, self).__init__()

        self.regnetx = models.regnet_x_3_2gf()
        self.regnetx.stem[0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.regnetx.fc = nn.Linear(
            in_features=1008, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.regnetx(x)
        return x


@add_embedding("EFFICIENT")
class EfficientNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(EfficientNet_Encoder, self).__init__()

        self.efficient_net = models.efficientnet_b0().features
        self.efficient_net[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(1280, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.efficient_net(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.leakyrelu(self.linear(x))
        return x


@add_embedding("SWINS_FFT_FILTER")
class SwinTransformerS_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(SwinTransformerS_Encoder, self).__init__()

        self.swin_transformer = models.swin_t()
        self.swin_transformer.features[0][0] = nn.Conv2d(
            1, 96, kernel_size=(4, 4), stride=(4, 4)
        )
        self.swin_transformer.head = nn.Linear(
            in_features=768, out_features=output_dimension, bias=True
        )
        self._fft_filter = LowPassFilter(128, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.swin_transformer(x)
        return x


@add_embedding("WIDERES50")
class WideResnet50_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(WideResnet50_Encoder, self).__init__()

        self.wideresnet = models.wide_resnet50_2()
        self.wideresnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.wideresnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("WIDERES101")
class WideResnet101_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(WideResnet101_Encoder, self).__init__()

        self.wideresnet = models.wide_resnet101_2()
        self.wideresnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.wideresnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("REGNET")
class RegNetY_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(RegNetY_Encoder, self).__init__()

        self.regnety = models.regnet_y_800mf()
        self.regnety.stem[0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.regnety.fc = nn.Linear(
            in_features=784, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.regnety(x)
        return x


@add_embedding("SHUFFLENET")
class ShuffleNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ShuffleNet_Encoder, self).__init__()

        self.shuffle_net = models.shufflenet_v2_x0_5()
        self.shuffle_net.conv1[0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.shuffle_net.fc = nn.Linear(
            in_features=1024, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.shuffle_net(x)
        return x


@add_embedding("RESNET18_FFT_FILTER")
class ResNet18_FFT_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(128, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    pass
