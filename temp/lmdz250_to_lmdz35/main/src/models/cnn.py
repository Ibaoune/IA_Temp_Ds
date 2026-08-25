import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    """
    Baño-like CNN for statistical downscaling.

    Logic:
    - convolutional feature extractor on coarse predictors
    - flatten
    - dense projection to fine target grid

    Parameters
    ----------
    input_shape : tuple
        (C_in, H_in, W_in)
    out_channels : int
        1 for deterministic output
        2 for Gaussian output [mean, log_var]
        3 for Bernoulli-Gamma output
    output_shape : tuple
        (H_out, W_out)
    mode : str
        "cnn1"  -> last conv feature maps = 1
        "cnn10" -> last conv feature maps = 10
    """

    def __init__(self, input_shape, out_channels=1, output_shape=(150, 180), mode="cnn10"):
        super().__init__()

        in_channels = input_shape[0]
        h_in, w_in = input_shape[1], input_shape[2]

        self.output_shape = output_shape
        self.out_channels = out_channels
        self.mode = mode.lower()

        if self.mode == "cnn1":
            last_hidden = 1
        elif self.mode == "cnn10":
            last_hidden = 10
        else:
            raise ValueError("mode must be 'cnn1' or 'cnn10'")

        # Baño-like convolutional stack
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, last_hidden, kernel_size=3, padding=1)

        self.act = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        fc_input_size = last_hidden * h_in * w_in
        fc_output_size = int(np.prod(output_shape))

        # one dense head per output parameter
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_size, fc_output_size)
            for _ in range(out_channels)
        ])

    def forward(self, x):
        """
        Input  : (B, C_in, H_in, W_in)
        Output : (B, out_channels, H_out, W_out)
        """
        batch_size = x.size(0)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        x = self.flatten(x)

        outputs = []
        for fc in self.fc_layers:
            y = fc(x)
            y = y.view(batch_size, 1, self.output_shape[0], self.output_shape[1])
            outputs.append(y)

        outputs = torch.cat(outputs, dim=1)

        if self.out_channels == 2:
            mean = outputs[:, 0:1, :, :]
            log_var = outputs[:, 1:2, :, :].clamp(min=-10.0, max=10.0)
            outputs = torch.cat([mean, log_var], dim=1)

        return outputs