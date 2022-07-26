from torch import nn


class ConvNeuralNet(nn.Module):
    def __init__(self, input_pixels):
        super().__init__()

        self.input_pixels = input_pixels

        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)

        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6 * 4 * 4, out_features=8)

    def forward(self, x):

        x = x.view(-1, 1, self.input_pixels, self.input_pixels)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = nn.functional.relu(self.fc(x))
        return x


embedding_net = ConvNeuralNet()
