# Import libraries
import torch
import torch.nn as nn
from torch.nn.functional.F import linear
import numpy as np
import matplotlib.pyplot as plt

class DataManager(object):
    """
        Data manager class, containing all the data sets used to train, validate, test the model.
    """
    def __init__(self, train_set, test_set, valid_set=None):
        """Create the data manager

        Args:
            train_set (DataLoader): Train data set as a pytorch DataLoader or any class containing similar methods
            test_set (DataLoader): Test data set as a pytorch DataLoader or any class containing similar methods
            valid_set (DataLoader, optional): Valid data set as a pytorch DataLoader or any class containing similar methods. Defaults to None.
        """
        self.train_set = train_set
        self.test_set = test_set
        self.valid_set = valid_set

class ScaleMixtureGaussian(object):
    """
        Scale mixture gaussian that can be used to replace to standard normal distribution
    """
    def __init__(self, pi=0.25, sigma1=0.75, sigma2=0.1):
        """Init the distribution

        Args:
            pi (float, optional): [description]. Defaults to 0.25.
            sigma1 (float, optional): Sigma of the first normal distribtion. Defaults to 0.75.
            sigma2 (float, optional): Sigma of the second normal distribtion. Defaults to 0.1.
        """
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, inputs):
        prob_g1 = torch.exp(self.gaussian1.log_prob(inputs))
        prob_g2 = torch.exp(self.gaussian2.log_prob(inputs))
        return torch.log(self.pi * prob_g1 + (1-self.pi) * prob_g2)

class BnnLayer(nn.Module):
    """
        BNN custom Layer implementing BBB.
    """
    def __init__(self, input_features, output_features, device, prior_var=10.):
        """Init the layer and set all the parameters

        Args:
            input_features (int): Number of input features
            output_features (int): Number of output features
            device (torch.device): The device on which the tensors are stored
            prior_var (float, optional): Inital variance of the prior distribution. Defaults to 10..
        """
        super().__init__()

        self.device = device
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        self.normal_d = torch.distributions.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(
            torch.empty(output_features, input_features, device=device)
            .uniform_(-0.5, 0.5))
        self.w_rho = nn.Parameter(
            torch.empty(output_features, input_features, device=device)\
            .uniform_(-0.5, 0.5))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(
            torch.empty(output_features, device=device)
            .uniform_(-0.5, 0.5))
        self.b_rho = nn.Parameter(
            torch.empty(output_features, device=device)
            .uniform_(-0.5, 0.5))

        # initialize weight samples
        self.w = self.sample_softmax(self.w_mu, self.w_rho)

        # initialize bias samples
        self.b = self.sample_softmax(self.b_mu, self.b_rho)

        # initialize prior distribution for all of the weights and biases
        if type(prior_var) == float or type(prior_var) == int:
            self.prior = torch.distributions.Normal(torch.tensor(0.0, device=self.device), torch.tensor(prior_var, device=self.device))
        else:
            self.prior = ScaleMixtureGaussian(*torch.tensor(prior_var, device=self.device))

        self.log_prior = 0
        self.log_post = 0


    def forward(self, inputs):
        """Forward pass of the layer

        Args:
            inputs (torch.tensor): inputs

        Returns:
            torch.tensor: outputs
        """
        # sample weights
        self.w = self.sample_softmax(self.w_mu, self.w_rho)

        # sample bias
        self.b = self.sample_softmax(self.b_mu, self.b_rho)

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        w_post = torch.distributions.Normal(self.w_mu.data, self.softmax(self.w_rho))
        b_post = torch.distributions.Normal(self.b_mu.data, self.softmax(self.b_rho))
        self.log_post = w_post.log_prob(self.w).sum() + b_post.log_prob(self.b).sum()

        return linear(inputs, self.w, self.b)

    def softmax(self, rho):
        """Softmax function

        Args:
            rho (float): rho

        Returns:
            float: output
        """
        return torch.log(1+torch.exp(rho))

    def sample_softmax(self, mu, rho):
        """Generate sample of the distribution and calculate softmax

        Args:
            mu (tensor.float): mu
            rho (tensor.float): rho

        Returns:
            tensor.float: sampled distribution
        """
        epsilon = self.normal_d.sample(mu.shape)
        return mu + torch.log(1+torch.exp(rho)) * epsilon

class Bnn(nn.Module):
    """
        Bayesian neural network implementing BNN.
    """
    def __init__(self, nn_layout, device, noise_tol=.1,  prior_var=10.):
        # BNN init with custom structure
        super().__init__()
        self.input_units = nn_layout[0]
        self.output_units = nn_layout[-1]

        noise_tol = float(noise_tol)
        if type(prior_var) == int:
            prior_var = float(prior_var)

        fcs = []
        fcs.extend([
            BnnLayer(nn_layout[i], nn_layout[i+1], device, prior_var=prior_var)
            for i in range(len(nn_layout)-1)
        ])
        self.funcs = nn.ModuleList(fcs)
        self.device = device
        self.noise_tol = torch.tensor(noise_tol, device=self.device)

    def forward(self, x, a_func):
        # Forward pass of the neural network
        for layer in self.funcs[:-1]:
            x = a_func(layer(x))
        return self.funcs[-1](x)

    def log_prior(self):
        # calculate the sum of the log prior over all the layers
        return sum(l.log_prior for l in self.funcs)

    def log_post(self):
        # calculate the sum of the log posterior over all the layers
        return sum(l.log_post for l in self.funcs)

    def sample_elbo(self, inputs, target, samples, nb_batches):
        pass

    def loss_fn(self, log_post, log_prior, log_like, nb_batches):
        return (log_post - log_prior) / nb_batches - log_like

class BnnModelManager(object):
    def __init__(self, bnn_class, nn_layout, prior_var, source=None, cuda=False, verbose=False):
        # initialize the NN
        self.cuda = cuda
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = bnn_class(nn_layout, self.device, prior_var=prior_var)
        if source:
            self.model.load_state_dict(torch.load(source))
        if cuda:
            self.model.to(self.device)
        if verbose:
            print(self.model)

    def train(self, data_manager, n_epochs, learning_rate=0.1, verbose=True, save_file='tmp.pt', e_print=100):
        # Train the network

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        num_batches_train = len(data_manager.train_set)
        valid_mode = False if not data_manager.valid_set else True
        num_batches_valid = len(data_manager.valid_set) if valid_mode else None

        if not valid_mode:
            self.model.train() # prep model for training
        else:
            valid_loss_min = np.Inf  # set initial "min" to infinity

        for epoch in range(n_epochs):  # loop over the dataset as many times as specified
            # monitor losses
            train_loss = 0
            valid_loss = 0

            ###################
            # train the model #
            ###################
            if valid_mode:
                self.model.train() # prep model for training

            # for data, label in tqdm(data_manager.train_loader, miniters=nb_batches/100):
            for data, label in data_manager.train_set:
                if self.cuda:
                    data, label = data.to(self.device), label.to(self.device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass and calculate loss
                loss = self.model.sample_elbo(data, label, 1, num_batches_train)
                # backward pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()

                # update train loss
                train_loss += loss.item()

            # Average loss
            train_loss /= len(data_manager.train_set.sampler)

            ######################
            # validate the model #
            ######################
            if valid_mode:
                self.model.eval()  # prep model for evaluation
                with torch.no_grad():
                    for data, label in data_manager.valid_set:
                        if self.cuda:
                            data, label = data.to(self.device), label.to(self.device)
                        # calculate the loss
                        loss = self.model.sample_elbo(data, label, 1, num_batches_valid)
                        # update running validation loss
                        valid_loss += loss.item()

                    valid_loss /= len(data_manager.valid_set.sampler)

                if verbose and (epoch+1) % e_print == 0:
                    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                        epoch+1, n_epochs, train_loss, valid_loss))
                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    if verbose:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min, valid_loss))
                    torch.save(self.model.state_dict(), save_file)
                    valid_loss_min = valid_loss
            elif verbose and (epoch+1) % e_print == 0:
                print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(
                    epoch+1, n_epochs, train_loss))

        if valid_mode:
            # Load the Model with Lowest Validation Loss
            self.model.load_state_dict(torch.load(save_file))
        else:
            torch.save(self.model.state_dict(), save_file)

    def test(self, data_manager, verbose=True, samples=100):
        pass

    def predict(self, inputs, samples=100, verbose=False):
        pass

    def predict_with_uncertainty(self, inputs, samples=100, extra=5.0, verbose=False):
        pass

    def softplus(self, x):
        return np.log(1. + np.exp(x))

    def display_weight_dist(self, layer_id, feature_id, node=None):
        import scipy.stats as stats

        layer = self.model.funcs[layer_id]
        if node == None:
            rho = layer.w_rho[:, feature_id].detach().cpu().mean()
        else:
            rho = layer.w_rho[node][feature_id].detach().cpu()
        std = abs(self.softplus(rho))

        if node == None:
            mean = layer.w_mu[:, feature_id].detach().cpu().mean()
        else:
            mean = layer.w_mu[node][feature_id].detach().cpu()
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        print(f'Std: {std}, rho: {rho}, mean: {mean}')
        plt.plot(x, stats.norm.pdf(x, mean, std), label='std = '+str(round(std.item(), 4)))
        plt.legend()
        plt.show()
        return (std.item(), rho.item(), mean.item())

    def model_uncertainty(self):
        tabs = []
        for j, func in enumerate(self.model.funcs):
            ff = []
            for i in range(func.input_features):
                ff.append(self.uncertainty_level(j, i))
            tabs.append(np.mean(ff)+np.sum(ff)*((j+1)**2))
        return np.sum(tabs)

    def uncertainty_level(self, layer_id, feature_id):
        layer = self.model.funcs[layer_id]
        stds = abs(self.softplus(layer.w_rho[:, feature_id].detach().cpu()))
        return (stds.mean()+(stds.max()-stds.min())).item()

    def inputs_uncertainty_table(self, layer_id, input_names=None, sort_list=True, verbose=True):
        layer = self.model.funcs[layer_id]

        tabs = []
        for i in range(layer.input_features):
            tabs.append(
                (input_names[i] if input_names else i,
                self.uncertainty_level(layer_id, i))
            )
        if sort_list:
            tabs.sort(key=lambda x:x[1])
        if verbose:
            print('rank | feature | uncertainty')
            for i, t in enumerate(tabs[::-1]):
                print(f'{i+1:4} | {t[0]:7} | {round(t[1], 4)}')
        return tabs

    def nodes_std_table(self, layer_id, feature_id, verbose=True):
        layer = self.model.funcs[layer_id]
        stds = abs(self.softplus(layer.w_rho[:, feature_id].detach().cpu())).numpy()
        if verbose:
            print(f'id  |    mu   | std')
            for i, (s, m) in enumerate(zip(stds, layer.w_mu[:, feature_id])):
                print(f'{i:3} | {round(m.item(), 4):7} | {round(s.item(), 4)}')
        return list(zip(stds, layer.w_mu[:, feature_id].detach().cpu().numpy()))
