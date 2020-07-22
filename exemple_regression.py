#%%
# Import libraries
import numpy as np
from torch import from_numpy, linspace
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch

import bnn_regression as b_reg
#%%
class RegDataset(Dataset):
    def __init__(self, x_values, labels):
        self.x = from_numpy(x_values).float()
        self.y = from_numpy(labels).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        data = self.x[i]

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

#%%
train_data = RegDataset(np.linspace(-10, 10, 10).reshape((10, 1)), np.linspace(-10, 10, 10)**2)
test_data = RegDataset(np.linspace(-10, 10, 100).reshape((100, 1)), np.linspace(-10, 10, 100)**2)
test_data2 = RegDataset(np.linspace(-100, 100, 1000).reshape((1000, 1)), np.linspace(-100, 100, 1000)**2)

# data loaders
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
test_loader2 = DataLoader(test_data2, batch_size=100, shuffle=True)

d_m = b_reg.DataManager(train_loader, test_loader, None)

print('Data loaded')
#%%
has_cuda = True
model_file = None
model_file = 'soft_test_reg1.pt'

model = b_reg.create_model(model_file, has_cuda, nn_layout=(1, 32, 1), prior_var=10.0)
print('Model created')
#%%
print('Initiate training')
nb_epochs = 600
model.train(d_m, nb_epochs, 0.1, True, 'soft_test_reg1.pt', e_print=100)
print('Finished training')
#%%
print('Testing model')
_, samps = model.test(d_m)
#%%
model.uncertainty_along_axis(train_loader)

#%%
model.uncertainty_along_axis(test_loader2)
#%%
model.uncertainty_along_axis(test_loader2, x_range=(-75, -25))
model.uncertainty_along_axis(test_loader2, x_range=(None, -25))
model.uncertainty_along_axis(test_loader2, x_range=(75, None))

# %%
model.uncertainty_level(0, 0)
