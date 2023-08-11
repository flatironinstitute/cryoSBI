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


@add_embedding("RESNET18_TEST2")
class ResNet18_Encoder_Test2(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_Encoder_Test2, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(15, 15), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


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
    
    
@add_embedding("RESNET18_TEST")
class ResNet18_Encoder_Test(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_Encoder_Test, self).__init__()
        print("Training with avg pooling")
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)

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

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.efficient_net(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        return x


@add_embedding("SWINS")
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

    def forward(self, x):
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


@add_embedding("REGNETY")
class RegNetY_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(RegNetY_Encoder, self).__init__()

        self.regnety = models.regnet_y_1_6gf()
        self.regnety.stem[0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.regnety.fc = nn.Linear(
            in_features=888, out_features=output_dimension, bias=True
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
    
    
@add_embedding("RESNET18_FFT_FILTER_224")
class ResNet18_FFT_Encoder_224(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_224, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(224, 30)
        print("embedding with 224 lp 30")

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x
    

@add_embedding("RESNET18_FFT_FILTER_224_LP25")
class ResNet18_FFT_Encoder_224_LP25(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_224_LP25, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(224, 25)
        print("embedding with 224 lp 25")

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET18_FFT_FILTER_132")
class ResNet18_FFT_Encoder_132(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_132, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(132, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET34")
class ResNet34_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet34_Encoder, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x
    
    
@add_embedding("RESNET34_256_LP")
class ResNet34_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet34_Encoder, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )
        self._fft_filter = LowPassFilter(256, 50)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("VGG19")
class VGG19_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(VGG19_Encoder, self).__init__()
        
        self.vgg19 = models.vgg19_bn().features
        self.vgg19[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        self.feedforward = nn.Sequential(
            *[
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=output_dimension, bias=True),
                nn.ReLU(inplace=True)
            ]
        )
        
        #self._fft_filter = LowPassFilter(256, 50)

    def forward(self, x):
        # Low pass filter images
        #x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.vgg19(x)
        x = self.avgpool(x).flatten(start_dim=1)
        x = self.feedforward(x)
        return x


if __name__ == "__main__":
    pass
