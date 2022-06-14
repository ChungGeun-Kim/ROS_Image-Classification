

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 20
batch_size = 1
learning_rate = 0.001

# Image preprocessing modules

transform = transforms.Compose([
    transforms.CenterCrop((350,350)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


train_dataset = torchvision.datasets.ImageFolder(root='/home/iracpc/data/hand_gesture4/dataset/train', 
                                                 transform=transform)

test_dataset = torchvision.datasets.ImageFolder(root='/home/iracpc/data/hand_gesture4/dataset/test', 
                                                transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes = train_dataset.classes

dataiter = iter(train_loader)
images, labels = dataiter.next()

def imshow(img):
    img = img / 2 + 0.5 
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))

imshow(torchvision.utils.make_grid(images,nrow=1))

