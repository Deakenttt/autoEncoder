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

    def __init__(self, dim_latent_representation=64):
        super(Autoencoder, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, output_size=64):
                super(Encoder, self).__init__()
                # needs your implementation
                self.hidden = torch.nn.Linear(28 * 28, 1024)
                self.Relu = nn.ReLU()
                self.encoder = torch.nn.Linear(1024, output_size)

            def forward(self, x):
                # needs your implementation
                #print("x shape before reshape", np.shape(x))
                x = x.view(-1, 28 * 28)  # reshape the shape to [batchSize, 28*28]
                #print("x shape after reshape", np.shape(x))
                x = self.hidden(x)  # extract the shape to [batchSize, 1024]
                #print("x shape after hidden layer", np.shape(x))
                x = self.Relu(x)
                #print("x shape after Relu", np.shape(x))
                x = self.encoder(x)   # encode the shape to [1024, dim=2]
                #print("x shape after encode", np.shape(x))
                return x

        class Decoder(nn.Module):
            def __init__(self, input_size=64):
                super(Decoder, self).__init__()
                # needs your implementation
                self.hidden = torch.nn.Linear(input_size, 1024)
                self.Relu = nn.ReLU()
                self.decoder = torch.nn.Linear(1024, 28 * 28)
                self.Sigmod = nn.Sigmoid()

            def forward(self, z):
                # needs your implementation
                z = self.hidden(z)  # extract the shape to [dim=2, 1024]
                z = self.Relu(z)
                z = self.decoder(z)  # decode shape back to [1024, dim=28*28]
                #print("shape after decode", np.shape(z))
                z = self.Sigmod(z)
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

autoencoder = Autoencoder(dim_latent_representation=64)
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

#from autoencoder_starter import scatter_plot
# scatter_plot(latent_representations=z, labels=label)


with torch.no_grad():
    samples = torch.randn(11, 64).to(trainer.device)
    samples = trainer.model.decoder(samples).cpu()

print("the size of sample we choose", samples.shape)
images = samples

from autoencoder_starter import display_images_in_a_row

display_images_in_a_row(images)

images = trainer.get_val_set()  # get the entire validation set
total_number = 64
images = images[:total_number]
print("tensor size of first 64 image", images.shape)


from autoencoder_starter import display_images_in_a_row
print("Original images")
display_images_in_a_row(images.cpu(), file_path='./ori.png')

print("==========print validation set as tensor ===========")
#print("pick one image", images[12].shape)
display_images_in_a_row(images[12], file_path='./tmp0.png', display=True)
#print("pick another one image", images[11].shape)
display_images_in_a_row(images[11], file_path='./tmp1.png', display=True)
print("=========part 1.4 ")
T = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
itpImages = torch.zeros((11, 1, 28, 28))
i = 0
for t in T:
    itpImages[i] = t * images[12] + (1-t) * images[11]
    i = i + 1
display_images_in_a_row(itpImages, file_path='./itp.png')


#
# #print("pick one image", images[12].shape)
# display_images_in_a_row(images[12], file_path='./tmp0.png', display=True)
# #print("pick another one image", images[11].shape)
# display_images_in_a_row(images[11], file_path='./tmp1.png', display=True)
# with torch.no_grad():
#     print("===========part 1.5 ")
#     print("===== interpolate image Using part 1.2 2-dim model")
#     # print("pick one image", images[12].shape)
#     T = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     itpImages = torch.zeros((11, 1, 28, 28))
#     index = 0
#     for t in T:
#         model = trainer.model
#         #print("the shape of image before encode: ", images[12].shape)
#         tmp = model.encoder(t * images[12].view(1,1,28,28)) + model.encoder((1-t) * images[11].view(1,1,28,28))
#         #print("the shape of image after encode: ", tmp.shape)
#         sample = tmp.to(trainer.device)
#         #print("the shape of image before decode: ", sample.shape)
#         itpImages[index] = model.decoder(sample).cpu()
#         #print("the shape of image after decode: ", itpImages[0].shape)
#         index = index + 1
#     display_images_in_a_row(itpImages, file_path='./itpG-dim.png')
#

with torch.no_grad():
    images = images.to(trainer.device)
    reconstructed = trainer.model(images).cpu()
print("Reconstructed images")
display_images_in_a_row(reconstructed, file_path='./rec.png')
