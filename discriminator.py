import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)

        # Second conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2)

        # Third conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2)

        # Fourth conv layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2)

        # Flatten the output of the convolutional layers to vector
        self.flatten = nn.Flatten()

        # Fully connected layer to classify the image
        self.dense1 = nn.Linear(512 * 16 * 16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x) 

        return x