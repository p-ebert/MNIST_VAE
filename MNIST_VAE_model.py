import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

#Setting up the models

class MLP(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        super(MLP, self).__init__()

        #Encoder layers
        self.fc1=nn.Linear(input_size, 4*latent_size)
        self.fc2=nn.Linear(4*latent_size, 2*latent_size)
        self.fc3_mean=nn.Linear(2*latent_size, latent_size)
        self.fc3_var=nn.Linear(2*latent_size, latent_size)

        #Decoder layers
        self.fc4=nn.Linear(latent_size, 2*latent_size)
        self.fc5=nn.Linear(2*latent_size, 4*latent_size)
        self.fc6=nn.Linear(4*latent_size, output_size)

    def encode(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mean=F.relu(self.fc3_mean(x))
        log_variance=F.relu(self.fc3_var(x))
        return mean, log_variance

    def decode(self, x):
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=torch.sigmoid(self.fc6(x))
        return x

    def sample(self, mean, log_variance):
        var=torch.exp(0.5*log_variance)
        eta=torch.randn(var.size(), device=var.device)
        return mean+var*eta

    def forward(self, x):
        mean, log_variance=self.encode(x)
        z=self.sample(mean, log_variance)
        output=self.decode(z)
        return output, mean, log_variance
