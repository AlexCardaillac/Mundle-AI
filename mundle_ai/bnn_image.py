#%%
# Import libraries
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from .bnn_model import Bnn, DataManager, BnnModelManager


class BnnImgClass(Bnn):
    """
        Bayesian neural network implementing BNN. Image classification version.
    """
    def __init__(self, nn_layout, device, noise_tol=.1,  prior_var=10.):
        # BNN init with custom structure
        super().__init__(nn_layout, device, noise_tol,  prior_var)

    def forward(self, x):
        # Forward pass of the neural network
        x = x.view(-1,self.input_units)
        return F.log_softmax(super().forward(x, F.relu), dim=1)

    # def sample_elbo(self, inputs, target, samples, num_batches=1, verbose=False):
    def sample_elbo(self, inputs, target, samples, nb_batches):
        # we calculate the negative elbo
        #initialize tensors

        outputs = torch.zeros(samples, target.shape[0], self.output_units, device=self.device)
        log_priors = torch.zeros(samples, device=self.device)
        log_posts = torch.zeros(samples, device=self.device)

        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(inputs) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior

        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        neg_log_like = F.nll_loss(outputs.mean(0), target, size_average=False)

        # return the negative elbo
        return self.loss_fn(log_post, log_prior, -neg_log_like, nb_batches)

class ModelManager(BnnModelManager):
    def __init__(self, source=None, cuda=False, verbose=False, nn_layout=(6, 32, 1), prior_var=1.0):
        # initialize the NN
        super().__init__(
            BnnImgClass, nn_layout, prior_var,
            source, cuda, verbose)

    def test(self, data_manager, verbose=True, samples=100):
        # Test the trained Network
        self.model.eval() # prep model for evaluation

        class_correct = list(0. for i in range(self.model.output_units))
        class_total = list(0. for i in range(self.model.output_units))

        with torch.no_grad():
            for data, target in data_manager.test_set:
                if self.model.cuda:
                    data, target = data.to(self.device), target.to(self.device)

                outputs = torch.zeros(samples, data.shape[0], self.model.output_units).to(self.device)
                for i in range(samples):
                    outputs[i] = self.model(data)

                output = outputs.mean(dim=0).argmax(dim=-1)
                correct = np.squeeze(output.eq(target.data.view_as(output)))
                for i in range(len(target)):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        for i in range(self.model.output_units):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

        return (class_correct, class_total)

    def predict(self, inputs, samples=100, verbose=False):
        self.model.eval()
        with torch.no_grad():
            batch = torch.from_numpy(np.array(inputs)).type(torch.FloatTensor)

            if self.cuda:
                batch = batch.to(self.device)

            outputs = torch.zeros((samples, len(batch), self.model.output_units))
            for i in range(samples):
                outputs[i] = self.model(batch)
            preds = outputs.mean(dim=0)

            if verbose:
                for idx, label in enumerate(preds.argmax(1)):
                    print('Instance {}: {}'.format(idx, label))
        return preds.argmax(1).numpy(), outputs.numpy()

    def predict_with_uncertainty(self, inputs, samples=100, class_names=None, verbose=True):
        nb_inputs = len(inputs)
        preds, outputs = self.predict(inputs, verbose=False)

        probs = outputs.argmax(axis=2).T

        if verbose:
            fig = plt.figure(figsize=(25, 4))
            for idx in range(nb_inputs):
                ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
                plt.imshow(inputs[idx][0])
                ax.set_title(f'{class_names[preds[idx]] if class_names else preds[idx]}')
            plt.show()

            plt.subplots(5,1,figsize=(10,4))
            mid = int(nb_inputs/2)
            for i in range(nb_inputs):
                plt.subplot(nb_inputs,1,i+1)
                plt.ylim(0,100)
                plt.xlabel('Classes')
                plt.xticks(range(self.model.output_units), class_names if class_names else range(self.model.output_units))
                if i == mid:
                    plt.ylabel('Probabilities')
                plt.yticks(range(50,101,50))
                plt.hist(probs[i], np.arange(-0.5, 10, 1))

        return preds, probs

    def plot_feature_uncertainty_map(self, img_shape, verbose=True):
        grid = np.array(self.inputs_uncertainty_table(0, sort_list=False, verbose=False))[:,1].reshape(img_shape)

        if img_shape[0] == 3:
            if verbose:
                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 6),
                                        subplot_kw={'xticks': [], 'yticks': []})

                for ax, zone_g, title in zip(axs.flat, [grid[0], grid[1], grid[2]], ['Red Channel', 'Green Channel', 'Blue Channel']):
                    ax.imshow(zone_g, cmap='viridis')
                    ax.set_title(title)

                plt.tight_layout()
                plt.show()
            u_map = np.mean([grid[0], grid[1], grid[2]], 0)
        else:
            u_map = grid

        if verbose:
            plt.imshow(u_map, cmap='viridis')
            plt.tight_layout()
            plt.show()
        return u_map

    def model_explanation(self, img, top_labels=5, hide_color=0, num_samples=1000, segmenter=None, colored_image=False, u_map=None):
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        from skimage.color import rgb2gray

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
                        img,
                        lambda x: np.mean(self.predict(np.array([rgb2gray(t) for t in x]))[1], axis=0),
                        top_labels=top_labels, hide_color=hide_color,
                        num_samples=num_samples, segmentation_fn=segmenter
                    )
        fig = plt.figure(figsize=(40, 10))

        ax = fig.add_subplot(2, 20/2, 1, xticks=[], yticks=[])
        plt.imshow(img)
        ax.set_title('Original image')

        ax = fig.add_subplot(2, 20/2, 2, xticks=[], yticks=[])
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=50, hide_rest=False)
        plt.imshow(mark_boundaries(temp, mask, mode='subpixel'))
        ax.set_title('All positive & negative features')

        if u_map is None:
            if not colored_image:
                img_side = int(self.model.funcs[0].input_features**0.5)
                u_map = self.plot_feature_uncertainty_map((img_side, img_side), verbose=False)
            else:
                img_side = int(self.model.funcs[0].input_features / 3 **0.5)
                u_map = self.plot_feature_uncertainty_map((3, img_side, img_side), verbose=False)

        ax = fig.add_subplot(2, 20/2, 3, xticks=[], yticks=[])
        u_mask = np.zeros(explanation.segments.shape)
        for f in np.unique(explanation.segments):
            u_mask[explanation.segments == f] = u_map[explanation.segments == f].mean()
        plt.imshow(u_mask, cmap='Blues')
        ax.set_title('Uncertainty distribution')

        return explanation



def create_model(source=None, cuda=False, nn_layout=(28*28, 512, 10), prior_var=1.0):
    # create the BNN and return a model manager
    if cuda and torch.cuda.is_available():
        cuda = True
        print('Using PyTorch version: {} with CUDA'.format(torch.__version__))
    elif cuda:
        print('CUDA not available')
        cuda = False

    return ModelManager(source, cuda, nn_layout=nn_layout, verbose=True, prior_var=prior_var)
