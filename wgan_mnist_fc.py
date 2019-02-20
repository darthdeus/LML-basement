import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np


###########################
## Gen & Critic Networks ##
###########################

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(noise_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x.view(-1,28,28)

class Critic(nn.Module):
    def __init__(self, clipping):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        torch.nn.init.uniform_(self.fc1.weight.data, -clipping, clipping)
        torch.nn.init.uniform_(self.fc2.weight.data, -clipping, clipping)
        torch.nn.init.uniform_(self.fc3.weight.data, -clipping, clipping)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def clip_weights(self, c):
        for w in self.parameters():
            w.data = torch.clamp(w.data, -c, c)


##################
### Load MNIST ###
##################

train = datasets.MNIST(root='./mnist', train=True, download=True)
# Move whole dataset to GPU. Transformations to mean=0 and sd=1 are applied manually.
images = ((train.train_data.float()/255-0.1307)/0.3081).cuda()


#################
# Training Loop #
#################

batch_size = 256
noise_dim = 10
epochs = 50000
n_critic = 1
lr = 0.0005
clipping = 0.5

G = Generator(noise_dim).cuda()
C = Critic(clipping).cuda()

G_optim = torch.optim.RMSprop(G.parameters(), lr)
C_optim = torch.optim.RMSprop(C.parameters(), lr)

# Variables for Generator input (noise) and real images.
noise = torch.cuda.FloatTensor(batch_size, noise_dim)
indices = torch.cuda.LongTensor(batch_size)

index = 0
for m in range(epochs):
    for n in range(n_critic):
        # Sample from normal distribution with mean=0, sd=1.
        noise.normal_()
        # Sample indices to create batch of real images. -> images[indices]
        indices.random_(0,60000)
        # Loss and RMSprop for Critic.
        C_optim.zero_grad()
        loss = - 1/batch_size * (torch.sum(C(images[indices])) - torch.sum(C(G(noise))))
        loss.backward()
        C_optim.step()
        # Clip weights of Critic.
        C.clip_weights(clipping)

    # Create new input for G.
    noise.normal_()
    # Loss and RMSprop for Generator.
    G_optim.zero_grad()
    loss = - 1/batch_size * torch.sum(C(G(noise)))
    loss.backward()
    G_optim.step()

    if index%500 == 0:
        samples = torch.FloatTensor(10, noise_dim).cuda()
        samples.normal_()
        filename = 'img'
        img = ((G(samples).detach().cpu().numpy()*0.3081)+0.1307)*255
        tmp = img[0]
        for i in range(1,10):
            tmp = np.append(tmp, img[i], -1)
        print(tmp)
        plt.imsave(filename + str(index) + '.png', tmp)
    index += 1
