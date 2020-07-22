#%%
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from bnn_model import Bnn, BnnLayer, DataManager, BnnModelManager

class BnnRegression(Bnn):
    """
        Bayesian neural network implementing BNN. Regression version.
    """
    def __init__(self, nn_layout, device, noise_tol=.1,  prior_var=1.):
        # BNN init with custom structure
        super().__init__(nn_layout, device, noise_tol,  prior_var)

    def forward(self, x):
        # Forward pass of the neural network
        return super().forward(x, torch.sigmoid)

    def sample_elbo(self, inputs, target, samples, nb_batches):
        # Calculate the loss function, the negative elbo

        #initialise all tensors according to the wanted number of sample and number of classes
        outputs = torch.zeros(samples, target.shape[0], device=self.device)
        log_priors = torch.zeros(samples, device=self.device)
        log_posts = torch.zeros(samples, device=self.device)
        log_likes = torch.zeros(samples, device=self.device)

        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(inputs).reshape(-1) # call the forward pass / make predictions
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()

        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        # return the negative elbo
        return self.loss_fn(log_post, log_prior, log_like, nb_batches)
        # return (log_post - log_prior) / nb_batches - log_like

class ModelManager(BnnModelManager):
    def __init__(self, source=None, cuda=False, verbose=False, nn_layout=(6, 32, 1), prior_var=1.0):
        # initialize the NN
        super().__init__(
            BnnRegression, nn_layout, prior_var,
            source, cuda, verbose)

    def test(self, data_manager, verbose=True, samples=100):
        # Test the trained Network
        self.model.eval() # prep model for evaluation
        with torch.no_grad():
            yy = np.array([])
            res = np.array([])
            y_samp_res = []

            for data, target in data_manager.test_set:
                yy = np.append(yy, target.numpy())
                if self.model.cuda:
                    data, target = data.to(self.device), target.to(self.device)
                y_samp = np.zeros((samples, len(data)))
                for samp in range(samples):
                    y_tmp = self.model(data).detach().cpu().numpy()
                    y_samp[samp] = y_tmp.reshape(-1)
                res = np.append(res, np.mean(y_samp, axis=0))
                y_samp_res.append(y_samp)

            rmse = mean_squared_error(yy, res)**0.5
            if verbose:
                print("RMSE: {:.3f}".format(rmse))
                print("R2: {:.3f}\n".format(r2_score(yy, res)))

            # return rmse result and samples
            return rmse, y_samp if len(y_samp_res) == 1 else y_samp_res

    def predict(self, inputs, samples=100, verbose=False):
        self.model.eval()
        with torch.no_grad():
            batch = torch.from_numpy(np.array(inputs)).type(torch.FloatTensor)

            if self.cuda:
                batch = batch.to(self.device)

            y_samp = np.zeros((samples, len(batch)))
            for samp in range(samples):
                y_tmp = self.model(batch).detach().cpu().numpy()
                y_samp[samp] = y_tmp.reshape(-1)

            preds = np.mean(y_samp, axis=0)

            if verbose:
                for idx, label in enumerate(preds):
                    print(f'Instance {idx}: {label}')
        return preds

    def predict_with_uncertainty(self, inputs, samples=100, ci=5.0, verbose=False):
        self.model.eval()
        with torch.no_grad():
            batch = torch.from_numpy(np.array(inputs)).type(torch.FloatTensor)

            if self.cuda:
                batch = batch.to(self.device)

            y_samp = np.zeros((samples, len(batch)))
            for samp in range(samples):
                y_tmp = self.model(batch).detach().cpu().numpy()
                y_samp[samp] = y_tmp.reshape(-1)

            preds = list(zip(
                    np.mean(y_samp, axis=0),
                    np.percentile(y_samp, ci/2, axis=0),
                    np.percentile(y_samp, 100-ci/2, axis=0))
                )

            if verbose:
                print(f'With a {round(100-ci, 2)}%CI:')
                print('Instance |   min   |   pred  |   max')
                for idx, inf in enumerate(preds):
                    print(f'{idx:8} | {round(inf[1], 4):7} | {round(inf[0], 4):7} | {round(inf[2], 4):7}')
        return preds

    def model_explanation(self, train_data, instance, num_samples=5000, num_features=10, feature_names=None, class_names=None, categorical_features=None):
        import lime
        import lime.lime_tabular
        from IPython.core.display import display, HTML

        explainer = lime.lime_tabular.LimeTabularExplainer(train_data, feature_names=feature_names, class_names=class_names, categorical_features=categorical_features, mode='regression')
        exp = explainer.explain_instance(instance, self.predict, num_features=num_features, num_samples=num_samples)

        exp_list = exp.as_list()
        exp_html = exp.as_html()

        tabs = self.inputs_uncertainty_table(0, sort_list=False, input_names=feature_names, verbose=False)
        min_v = min(tabs, key = lambda k: k[1])[1]
        max_v = max(tabs, key = lambda k: k[1])[1]
        g_scales = []
        for x in exp_list:
            for u in tabs:
                if u[0] in x[0]:
                    g_scales.append((u[1]-min_v)/(max_v-min_v))
                    break

        js_add= f'''
                tabs= {g_scales};
                d3.selectAll('.explanation rect')
                    .each(function(d, i) {'{'}
                        d3.select(this).style('fill', 'rgb(0, ' + (255*tabs[i]) + ', 0)');
                    {'}'});
        '''
        exp_html_split = exp_html.split('\n')
        exp_html_split[-3] = js_add
        exp_html = '\n'.join(exp_html_split)

        file_ = open(f'tmp_exp.html', 'w', encoding='utf8')
        file_.write(exp_html)
        file_.close()

        display(HTML(exp_html))

        return exp

    def uncertainty_along_axis(self, data_loader, feature_id=0, samples=100, x_range=None):
        samps = None
        x_tmp = None

        self.model.eval()
        for data, _ in data_loader:
            if x_tmp is None:
                x_tmp = data.cpu().detach().numpy()
            else:
                x_tmp = np.append(x_tmp, data.cpu().detach().numpy(), 0)
            if self.cuda:
                data = data.to(self.device)
            y_samp = np.zeros((samples, data.shape[0]))
            for s in range(samples):
                y_tmp = self.model(data).cpu().detach().numpy()
                y_samp[s] = y_tmp.reshape(-1)
            if samps is None:
                samps = y_samp
            else:
                samps = np.append(samps, y_samp, 1)

        x_tmp = x_tmp[:, feature_id]
        x_tmp, samps = np.sort(x_tmp), samps.T[np.argsort(x_tmp)].T

        if x_range is not None:
            x_min, x_max = x_range
            if x_min is not None:
                id_min = np.argmax(x_tmp>=x_min)
                x_tmp = x_tmp[id_min:]
                samps = samps.T[id_min:].T
            if x_max is not None:
                id_max = np.argmin(x_tmp<=x_max)
                x_tmp = x_tmp[:id_max]
                samps = samps.T[:id_max].T
            print(x_min, x_max)

        plt.plot(x_tmp, np.mean(samps, axis=0), label='Mean Posterior')
        plt.fill_between(x_tmp, np.percentile(samps, 2.5, axis=0), np.percentile(samps, 97.5, axis=0), alpha=0.25, label='95% Confidence Interval')
        plt.legend()
        plt.grid(linestyle='--')
        plt.title('Posterior Prediction')
        plt.show()

def create_model(source=None, cuda=False, nn_layout=(6, 32, 1), prior_var=1.0):
    # create the BNN and return a model manager
    if cuda and torch.cuda.is_available():
        cuda = True
        print('Using PyTorch version: {} with CUDA'.format(torch.__version__))
    elif cuda:
        print('CUDA not available')
        cuda = False

    return ModelManager(source, cuda, nn_layout=nn_layout, verbose=True, prior_var=prior_var)
