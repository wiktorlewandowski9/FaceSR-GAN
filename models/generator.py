import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Conv
        self.norm1 = nn.InstanceNorm2d(64)  # InstanceNorm zamiast BatchNorm
        self.relu1 = nn.ReLU()  # Activation function

        # Second layer
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)  # Conv z większą liczbą kanałów
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)  # Upsample 64x64 -> 128x128
        self.norm2 = nn.InstanceNorm2d(64)  # InstanceNorm
        self.relu2 = nn.ReLU()

        # Third layer
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)  # Upsample 128x128 -> 256x256
        self.norm3 = nn.InstanceNorm2d(64)
        self.relu3 = nn.ReLU()

        # Fourth layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.InstanceNorm2d(128)
        self.relu4 = nn.ReLU()

        # Fifth layer
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(64)
        self.relu5 = nn.ReLU()

        # Output layer
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pixel_shuffle1(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pixel_shuffle2(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.tanh(x)

        return x