#%%
# Import libraries
import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import mundle_ai.bnn_regression as b_reg
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
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = np.expand_dims(r * np.sin(theta), 1)
y = np.expand_dims(r * np.cos(theta), 1)

X = np.append(x, y, axis=1)

train_size = 0.5
num_train = len(X)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(train_size * num_train))
train_index, test_index = indices[split:], indices[:split]

data_set = RegDataset(X, z)

# data loaders
train_sampler = SubsetRandomSampler(train_index)
test_sampler = SubsetRandomSampler(test_index)

train_loader = DataLoader(data_set, batch_size=100, sampler=train_sampler)
test_loader = DataLoader(data_set, batch_size=100, sampler=test_sampler)

d_m = b_reg.DataManager(train_loader, test_loader, None)

print('Data loaded')
#%%
has_cuda = True
model_file = None
# model_file = 'soft_test_reg2.pt'

model = b_reg.create_model(model_file, has_cuda, nn_layout=(2, 32, 1), prior_var=10.0)
print('Model created')
#%%
print('Initiate training')
nb_epochs = 1000
model.train(d_m, nb_epochs, 0.01, True, 'soft_test_reg2.pt', e_print=50)
print('Finished training')
#%%
print('Testing model')
_, samps = model.test(d_m)
#%%
model.uncertainty_along_axis(test_loader, 0)
model.uncertainty_along_axis(test_loader, 1)

#%%
model.uncertainty_level(0, 0)

# %%
model.model_explanation(X, X[0], feature_names=['x', 'y'], class_names=['z'])
