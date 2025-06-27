# model.py
import torch.nn as nn
import torchcde
from config import CONFIG

class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, CONFIG["cde_func_channels"]),
            nn.Tanh(),
            nn.Linear(CONFIG["cde_func_channels"], CONFIG["cde_func_channels"]),
            nn.Tanh(),
            nn.Linear(CONFIG["cde_func_channels"], input_channels * hidden_channels)
        )
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, t, z):
        return self.net(z).view(-1, self.hidden_channels, self.input_channels)

class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, x_padded):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_padded)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.initial(X.evaluate(X.interval[0]))
        z_final = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval, method='rk4', options={'step_size': 0.1})[:, 1]
        pred = self.readout(z_final)
        return pred
