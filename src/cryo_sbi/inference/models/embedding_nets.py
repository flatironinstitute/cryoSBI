import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from escnn import group
from escnn import gspaces
import escnn.nn as enn

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


@add_embedding("ESCNN")
class SO2SteerableCNN(torch.nn.Module):

    def __init__(self, output_dimension: int):

        super(SO2SteerableCNN, self).__init__()

        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.rot2dOnR2(N=-1)

        # the group SO(2)
        G = self.r2_act.fibergroup # self.G: SO2 = self.r2_act.fibergroup

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # We need to mask the input image since the corners are moved outside the grid under rotations
        self.mask = enn.MaskModule(in_type, 128, margin=1)

        # convolution 1
        # first we build the non-linear layer, which also constructs the right feature type
        # we choose 8 feature fields, each transforming under the regular representation of SO(2) up to frequency 3
        # When taking the ELU non-linearity, we sample the feature fields on N=16 points
        activation1 = enn.FourierELU(self.r2_act, 8, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation1,
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields
        activation2 = enn.FourierELU(self.r2_act, 16, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation2
        )
        # to reduce the downsampling artifacts, we use a Gaussian smoothing filter
        self.pool1 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields
        activation3 = enn.FourierELU(self.r2_act, 32, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation3.in_type
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation3
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 64 regular feature fields
        activation4 = enn.FourierELU(self.r2_act, 32, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation4.in_type
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation4
        )
        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        activation5 = enn.FourierELU(self.r2_act, 64, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation5.in_type
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation5
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation6 = enn.FourierELU(self.r2_act, 64, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation6.in_type
        self.block6 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool3 = enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)


        in_type = self.block6.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation7 = enn.FourierELU(self.r2_act, 128, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation7.in_type
        self.block7 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation7
        )

        in_type = self.block7.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation8 = enn.FourierELU(self.r2_act, 128, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation8.in_type
        self.block8 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation8
        )

        self.pool4 = enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=0)


        in_type = self.block8.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation9 = enn.FourierELU(self.r2_act, 128, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation9.in_type
        self.block9 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation9
        )

        in_type = self.block9.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation10 = enn.FourierELU(self.r2_act, 256, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation10.in_type
        self.block10 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation10
        )

        self.pool5 = enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=0)


        """in_type = self.block10.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation11 = enn.FourierELU(self.r2_act, 128, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation11.in_type
        self.block11 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation11
        )

        in_type = self.block11.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation12 = enn.FourierELU(self.r2_act, 256, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation12.in_type
        self.block12 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type),
            activation12
        )

        self.pool6 = enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)"""


        
        # number of output invariant channels
        c = 256

        # last 1x1 convolution layer, which maps the regular fields to c=64 invariant scalar fields
        # this is essential to provide *invariant* features in the final classification layer
        output_invariant_type = enn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = enn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = x.unsqueeze(1)
        x = self.input_type(x)

        # mask out the corners of the input image
        x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
    
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)

        x = self.block7(x)
        x = self.block8(x)
        x = self.pool4(x)

        x = self.block9(x)
        x = self.block10(x)
        x = self.pool5(x)

        #x = self.block11(x)
        #x = self.block12(x)
        #x = self.pool6(x)

        # extract invariant features
        x = self.invariant_map(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layer
        #x = self.fully_net(x.reshape(x.shape[0], -1))

        return x.flatten(start_dim=1)


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

        self.efficient_net = models.efficientnet_b3().features
        self.efficient_net[0][0] = nn.Conv2d(
            1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(1536, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.efficient_net(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.leakyrelu(self.linear(x))
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
        self.vgg19[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.feedforward = nn.Sequential(
            *[
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=output_dimension, bias=True),
                nn.ReLU(inplace=True),
            ]
        )

        # self._fft_filter = LowPassFilter(256, 50)

    def forward(self, x):
        # Low pass filter images
        # x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.vgg19(x)
        x = self.avgpool(x).flatten(start_dim=1)
        x = self.feedforward(x)
        return x


if __name__ == "__main__":
    pass
