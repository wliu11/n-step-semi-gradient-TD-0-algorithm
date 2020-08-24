import numpy as np
from algo import ValueFunctionWithApproximation
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
Value function approximation using a linear neural network with three hidden layers
"""
class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        super(ValueFunctionWithNN, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        
        # Initialize the adam optimizer and mean squared error loss function 
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, betas=(0.9,0.999))
        self.loss_fn = torch.nn.MSELoss()
        
    def __call__(self,s):
        self.network.eval()
        x = self.network(torch.tensor(s, dtype=torch.float32))
        return x.detach().numpy()[0]

    
    # Train the neural network once, obtain loss, then take one step backward
    def update(self,alpha,G,s_tau):
        self.network.train()
        self.optimizer.zero_grad()
        x = torch.tensor(s_tau, dtype=torch.float32)
        pred = self.network(x)
        loss = self.loss_fn(pred, torch.tensor(G, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
    

