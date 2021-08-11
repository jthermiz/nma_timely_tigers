import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayer(nn.Module):
    """Simple two layer MLP net"""

    def __init__(self, D_in, H, D_out):
        """Initialize class

        Parameters
        ----------
        D_in : int
            Number of input neurons
        H : int
            Number of hidden neurons
        D_out : int
            Number of output neurons
        """
        super(TwoLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.lil = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, D_out),
        )

    def forward(self, x):
        """Compute forward pass of net

        Parameters
        ----------
        x : Tensor
            Batch of examples

        Returns
        -------
        Tensor
            Output of net
        """
        xf = self.lil(x)
        xf = self.softmax(xf)
        return xf
