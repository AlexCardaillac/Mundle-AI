#%%
# Import libraries
import numpy as np
from torch import from_numpy, linspace
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import bnn_image as b_img
#%%
num_workers = 0
batch_size = 128
valid_size = 0.2
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers,
        pin_memory=True)
test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)

d_m = b_img.DataManager(train_loader, test_loader, valid_loader)
print('Data loaded')

#%%
has_cuda = True
model_file = None
model_file = 'ex_mnist1.pt'

# initialize the NN
model = b_img.create_model(model_file, has_cuda, nn_layout=(28*28, 800, 10), prior_var=10.0)
print('Model created')
#%%
print('Initiate training')
nb_epochs = 25
model.train(d_m, nb_epochs, 0.001, True, 'ex_mnist1.pt', e_print=1)
print('Finished training')
#%%
print('Testing model')
model.test(d_m)
#%%
model.uncertainty_level(0, 0)

# %%
model.display_weight_dist(0, 0)

# %%
u_map = model.plot_feature_uncertainty_map((28,28))

# %%
outputs, _ = model.predict(next(iter(test_loader))[0][:5])

# %%
_ = model.predict_with_uncertainty(next(iter(test_loader))[0][:5])


# %%
from lime.wrappers.scikit_image import SegmentationAlgorithm
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

_ = model.model_explanation(next(iter(test_loader))[0][0][0], segmenter=segmenter, u_map=u_map)

