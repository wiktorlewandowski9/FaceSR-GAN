import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # First layer
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1)  # Conv
        self.batch1 = nn.BatchNorm2d(64) # Normalization
        self.relu1 = nn.ReLU() # Activation function

        # Second layer
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # Third layer
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        # Fourth layer
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv5 = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = self.up3(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)

        x = self.up4(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.tanh(x)

        return x