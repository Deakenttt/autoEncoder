# load packages
# make sure to install the pacakge "tqdm" for the progress bar when training.
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.image as mpimg
from scipy import ndimage

path_prefix = "./"

# from google.colab import drive
# drive.mount('/content/gdrive')
#path_prefix = "/content/gdrive/MyDrive/CMPT 726-419 Spring 2021 A2"
# import sys
# sys.path.insert(1, path_prefix)


class Autoencoder(nn.Module):

    def __init__(self, dim_latent_representation=2):
        super(Autoencoder, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, output_size=2):
                super(Encoder, self).__init__()
                # needs your implementation
                self.encoder = torch.nn.Linear(28 * 28, output_size)

            def forward(self, x):
                # needs your implementation
                #print("x shape before reshape", np.shape(x)) # [batchSize, 1, 28, 28]
                x = x.view(-1, 28 * 28)  # reshape the shape to [batchSize, 28*28]
                #print("x shape after reshape", np.shape(x))
                x = self.encoder(x)   # encode the shape to [batchSize, dim=2]
                #print("x shape after encode", np.shape(x))
                return x

        class Decoder(nn.Module):
            def __init__(self, input_size=2):
                super(Decoder, self).__init__()
                # needs your implementation
                self.decoder = torch.nn.Linear(input_size, 28 * 28)
                self.Sigmoid = nn.Sigmoid()
            def forward(self, z):
                # needs your implementation
                z = self.decoder(z)  # decode shape back to [batchSize, dim=28*28]
                #print("shape after decode", np.shape(z))
                z = self.Sigmoid(z)
                z = z.view(-1, 1, 28, 28)   # reshape the shape to [32, 1, 28, 28]
                #print("shape after reshape", np.shape(z))
                #print("new y shape is", np.shape(z))
                return z

        self.encoder = Encoder(output_size=dim_latent_representation)
        self.decoder = Decoder(input_size=dim_latent_representation)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


from autoencoder_starter import Autoencoder_Trainer

LEARNING_RATE = 1e-3
EPOCH_NUMBER = 10  # the number of epochs and learning rate can be tuned.

autoencoder = Autoencoder(dim_latent_representation=2)
trainer = Autoencoder_Trainer(autoencoder_model=autoencoder, learning_rate=LEARNING_RATE, path_prefix=path_prefix)

try:
    for epoch in range(1, EPOCH_NUMBER + 1):
        trainer.train(epoch)
        trainer.validate(epoch)
except (KeyboardInterrupt, SystemExit):
    print("Manual Interruption")

with torch.no_grad():
    model = trainer.model
    model.eval()
    z = []
    label = []
    for x, y in trainer.val_loader:
        z_ = model.encoder(x.to(trainer.device))
        z += z_.cpu().tolist()
        label += y.cpu().tolist()
    z = np.asarray(z)
    label = np.asarray(label)

from autoencoder_starter import scatter_plot

scatter_plot(latent_representations=z, labels=label)

with torch.no_grad():
    samples = torch.randn(7, 2).to(trainer.device)
    samples = trainer.model.decoder(samples).cpu()

images = samples

from autoencoder_starter import display_images_in_a_row

display_images_in_a_row(images)

images = trainer.get_val_set()  # get the entire validation set
total_number = 64
images = images[:total_number]

from autoencoder_starter import display_images_in_a_row

print("Original images")
display_images_in_a_row(images.cpu())

with torch.no_grad():
    images = images.to(trainer.device)
    reconstructed = trainer.model(images).cpu()
print("Reconstructed images")
display_images_in_a_row(reconstructed)
