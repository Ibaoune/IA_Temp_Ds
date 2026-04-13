import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=1, output_shape=(160, 170)):
        super(CNN, self).__init__()

        self.output_shape = output_shape
        self.out_channels = out_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 1, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        # Compute FC input size
        fc_input_size = 1 * input_shape[1] * input_shape[2]
        
        # Linear layers for parameters
        # We need as many FC layers as out_channels
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_size, np.prod(output_shape)) 
            for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = self.flatten(x)

        # FC layers
        params = []
        for i, fc in enumerate(self.fc_layers):
            p = fc(x)
            params.append(p.view(batch_size, 1, self.output_shape[0], self.output_shape[1]))

        # Concatenate outputs
        outputs = torch.cat(params, dim=1)
        return outputs
