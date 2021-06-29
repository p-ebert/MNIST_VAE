import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets
import torchvision.transforms

from MNIST_VAE_model import MLP

#Hyperparameters
batch_size=30
latent_size=20
epochs=80
model_checkpoints=[10,79,80]
project_dir_location="."

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
else:
    print("Using CPU")

#Importing data
train_data=torchvision.datasets.MNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loaded=torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

#Instanciating the model and optimiser
img_shape=iter(train_loaded).next()[0].shape[2:4]
vae=MLP(img_shape[0]*img_shape[1], latent_size, img_shape[0]*img_shape[1])
torch.save(vae.state_dict(), "{}/Trainedmodels/Model_0.pth".format(project_dir_location))
if use_cuda:
    vae.cuda()

optimiser=torch.optim.Adam(vae.parameters(), lr=0.0001)

#Definition of the loss of the VAE
def VAE_loss(x, x_gen, mean, log_variance):
    #recon_loss=-torch.sum(x*torch.log(x_gen))
    recon_loss=F.binary_cross_entropy(x_gen, x, reduction="sum")
    #recon_loss=F.mse_loss(x_gen, x)
    KLD_loss=-0.5*torch.sum(1+2*log_variance-mean.pow(2)-(2*log_variance).exp())
    return recon_loss+KLD_loss

for epoch in range(1,epochs+1):
    vae.train()
    training_loss=0
    for batch_idx, (data, label) in enumerate(train_loaded):
        if use_cuda:
            data=data.cuda()

        data=data.view(batch_size, 1, img_shape[0]*img_shape[1])

        optimiser.zero_grad()
        x_gen, mean, log_variance=vae(data)
        loss=VAE_loss(data, x_gen, mean, log_variance)
        training_loss+=loss
        loss.backward()
        optimiser.step()

        if batch_idx%500 ==0:
            print("Training epoch:{} Batch number: {} Loss: {}".format(epoch, batch_idx, loss))

    print("EPOCH {}, Loss per batch: {}".format(epoch, training_loss/len(train_loaded.dataset)))

    if epoch in model_checkpoints:
        torch.save(vae.state_dict(), "{}/Trainedmodels/Model_{}.pth".format(project_dir_location, epoch))
