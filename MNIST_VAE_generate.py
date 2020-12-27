import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets
import torchvision.transforms

from MNIST_VAE_model import MLP

import matplotlib.pyplot as plt

batch_size=30
latent_size=20
epochs=80
model_checkpoints=[0,10,80]
project_dir_location="."


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
    device="gpu"
else:
    print("Using CPU")
    device="cpu"

#Importing data
test_data=torchvision.datasets.MNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loaded=torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)

img_shape=iter(test_loaded).next()[0].shape[2:4]

vae=MLP(img_shape[0]*img_shape[1], latent_size, img_shape[0]*img_shape[1])
if use_cuda:
    vae.cuda()

vae.eval()


for i in model_checkpoints:
    vae.load_state_dict(torch.load("{}/Trainedmodels/Model_{}.pth".format(project_dir_location, i), map_location=torch.device(device)))
    with torch.no_grad():
        z=torch.randn(64,20)
        if use_cuda:
            z=z.cuda()
        img_gen=vae.decode(z)
        img_gen=img_gen.view(8, 8, img_shape[0], img_shape[1]).numpy()

    fig, axes=plt.subplots(8,8, figsize=[30,30])


    for j in range(len(axes)):
        for k in range(len(axes[0])):
            axes[j,k].imshow(img_gen[j,k], cmap="gray")
    plt.savefig("{}/Img_{}.png".format(project_dir_location, i))
    plt.show()
