import torch
import torch.nn.functional as F
from torch import nn


class CifarMlp(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.classes = classes
        # CIFAR images are 32x32 pixels with 3 color channels
        input_dim = 32 * 32 * 3  # Flatten the 32x32x3 image into a single vector

        # Define the MLP architecture
        self.fc1 = nn.Linear(input_dim, 512)  # First dense layer
        self.fc2 = nn.Linear(512, 256)  # Second dense layer
        self.fc3 = nn.Linear(256, 128)  # Third dense layer
        self.fc4 = nn.Linear(128, 64)  # Fourth dense layer
        self.fc5 = nn.Linear(64, self.classes)  # Output layer

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
