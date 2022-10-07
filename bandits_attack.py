import os

import numpy as np
from tqdm import tqdm


ENV_MODEL = os.environ.get('ENV_MODEL')

if ENV_MODEL is None:
    ENV_MODEL = 'deepapi'

SCALE = 1.0
PREPROCESS = lambda x: x

if ENV_MODEL == 'keras':
    from tensorflow.keras.applications.vgg16 import preprocess_input
    SCALE = 255.0
    PREPROCESS = preprocess_input

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1) / 2 # from [-1, 1] to [0, 1]
    pos = real_x * np.exp(lr * g)
    neg = (1 - real_x) * np.exp(-lr * g)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1

def linf_step(x, g, lr):
    return x + lr * np.sign(g)

# def l2_prior_step(x, g, lr):
#     new_x = x + lr * g / np.linalg.norm(g)
#     norm_new_x = np.linalg.norm(new_x)
#     norm_mask = (norm_new_x < 1.0).float()
#     return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x

# def gd_prior_step(x, g, lr):
#     return x + lr * g

# def l2_image_step(x, g, lr):
#     return x + lr * g / np.linalg.norm(g)

def cross_entropy(y_pred, y_true):
    """
    y_pred is the output from fully connected layer (num_examples x num_classes)
    y_true is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    def softmax(X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    y_true = np.array(y_true)
    m = y_true.shape[0]
    p = softmax(y_pred)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y_true])
    # loss = np.sum(log_likelihood) / m
    return log_likelihood / m

class BanditsAttack():
    def __init__(self,  classifier):
        """
        Create a class: `BanditAttack` instance.
        - classifier: model to attack
        """
        self.classifier = classifier

    def init(self, x):
        """
        Initialize the attack.
        """
        y_pred = self.classifier.predict(PREPROCESS(x.copy()))

        x_adv = x.copy()

        priors = []
        for i in range(len(x)):
            h, w, c = x[i].shape
            priors.append(np.zeros((h, w, c)))
    
        return x_adv, y_pred, priors

    def step(self, x, x_adv, y, priors, epsilon, fd_eta, image_lr, online_lr, exploration):

        x_query_1 = []
        x_query_2 = []

        exp_noises = []

        for i, img in enumerate(x_adv):
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            h, w, c = img.shape
            dim = h * w * c
            exp_noise = exploration * np.random.normal(0.0, 1.0, size = priors[i].shape) / (dim ** 0.5) * SCALE

            # Query deltas for finite difference estimator
            q1 = priors[i] + exp_noise
            q2 = priors[i] - exp_noise

            x_query_1.append(img + fd_eta * q1 / np.linalg.norm(q1))
            x_query_2.append(img + fd_eta * q2 / np.linalg.norm(q2))

            exp_noises.append(exp_noise)

        if ENV_MODEL == 'keras':
            x_query_1 = np.array(x_query_1)
            x_query_2 = np.array(x_query_2)
            exp_noises = np.array(exp_noises)

        # Loss points for finite difference estimator
        l1 = cross_entropy(self.classifier.predict(PREPROCESS(x_query_1.copy())), y) # L(prior + c*noise)
        l2 = cross_entropy(self.classifier.predict(PREPROCESS(x_query_2.copy())), y) # L(prior - c*noise)

        for i, img in enumerate(x_adv):
            # Finite differences estimate of directional derivative
            est_deriv = (l1[i] - l2[i]) / (fd_eta * exploration)
            # 2-query gradient estimate
            est_grad = est_deriv * exp_noises[i]
            # Update the prior with the estimated gradient
            priors[i] = eg_step(priors[i], est_grad, online_lr)

            ## Update the image:
            # take a pgd step using the prior
            img = linf_step(img, priors[i], image_lr)
            img = x[i] + np.clip(img - x[i], -epsilon, epsilon)
            img = np.clip(img, 0, 1 * SCALE)

            x_adv[i] = img

        # y_pred = self.classifier.predict(x_adv)

        return x_adv, priors


    def attack(self, x, y, epsilon=0.05 * SCALE, fd_eta=0.01 * SCALE, image_lr=0.01*SCALE, online_lr=100, exploration=1.0, max_it=10000, distributed=False, batch=50, max_workers=10):
        """
        Initiate the attack.

        - x: input data
        - epsilon: perturbation on each pixel
        - max_it: number of iterations
        - distributed: if True, use distributed attack
        - batch: number of queries per worker
        - max_workers: number of workers
        """

        x_adv, y_pred, priors = self.init(x)

        # Continue query count
        y_pred_classes = np.argmax(y_pred, axis=1)
        correct_classified_mask = (y_pred_classes == y)
        not_dones_mask = correct_classified_mask.copy()

        print('Clean accuracy: {:.2%}'.format(np.mean(correct_classified_mask)))

        if distributed:
            pbar = tqdm(range(0, max_it, batch), desc="Distributed Bandits Attack")
        else:
            pbar = tqdm(range(0, max_it), desc="Bandits Attack")


        total_queries = np.zeros(len(x))

        for p in pbar:

            not_dones = [i for i, v in enumerate(not_dones_mask) if v]

            x_curr = [x[idx] for idx in not_dones]
            x_adv_curr = [x_adv[idx] for idx in not_dones]
            y_curr = [y[idx] for idx in not_dones]
            prior_curr = [priors[idx] for idx in not_dones]

            if ENV_MODEL == 'keras':
                x_curr = np.array(x_curr)
                x_adv_curr = np.array(x_adv_curr)
                y_curr = np.array(y_curr)
                prior_curr = np.array(prior_curr)

            if distributed:
                x_adv_curr, prior_curr = self.batch(x_curr, x_adv_curr, y_curr, prior_curr, epsilon, fd_eta, image_lr, online_lr, exploration, max_workers, batch)
            else:
                x_adv_curr, prior_curr = self.step(x_curr, x_adv_curr, y_curr, prior_curr, epsilon, fd_eta, image_lr, online_lr, exploration)

            y_curr = np.argmax(self.classifier.predict(PREPROCESS(x_adv_curr.copy())), axis=1)

            for i in range(len(not_dones)):
                x_adv[not_dones[i]] = x_adv_curr[i]
                priors[not_dones[i]] = prior_curr[i]
                y_pred_classes[not_dones[i]] = y_curr[i]

            # Logging stuff
            total_queries += 3 * not_dones_mask

            not_dones_mask = not_dones_mask * (y_pred_classes == y)

            # max_curr_queries = total_queries.max()

            success_mask = correct_classified_mask * (1 - not_dones_mask)
            num_success = success_mask.sum()
            current_success_rate = (num_success / correct_classified_mask.sum())

            if num_success == 0:
                success_queries = -1
            else:
                success_queries = ((success_mask * total_queries).sum() / num_success)
            
            pbar.set_postfix({'Total Queries': total_queries.sum(), 'Success Rate': current_success_rate, 'Avg Queries': success_queries})

            # Early break
            if current_success_rate == 1.0:
                break

        return x_adv
