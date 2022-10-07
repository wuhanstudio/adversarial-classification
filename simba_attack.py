"""
This module implements the black-box attack `SimBA`.
"""

import os
import numpy as np
import concurrent.futures
from tqdm import tqdm

ENV_MODEL = os.environ.get('ENV_MODEL')

if ENV_MODEL is None:
    ENV_MODEL = 'deepapi'

SCALE = 1.0
PREPROCESS = lambda x: x

if ENV_MODEL == 'keras':
    from tensorflow.keras.applications.vgg16 import preprocess_input
    SCALE = 255
    PREPROCESS = preprocess_input

def proj_lp(v, xi=0.1, p=2):
    """
    SUPPORTS only p = 2 and p = Inf for now
    """
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten('C')))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

class SimBA():
    """
    Implementation of the `SimBA` attack. Paper link: https://arxiv.org/abs/1905.07121
    """

    def __init__(self,  classifier):
        """
        Create a class: `SimBA` instance.
        - classifier: model to attack
        """
        self.classifier = classifier
    
    def init(self, x):
        """
        Initialize the attack.
        """

        x_adv = x.copy()
        y_pred = self.classifier.predict(PREPROCESS(x.copy()))

        perm = []
        for xi in x:
            perm.append(np.random.permutation(xi.reshape(-1).shape[0]))

        return x_adv, y_pred, perm

    def step(self, x_adv, y_pred, perm, index, epsilon):
        """
        Single step for non-distributed attack.
        """

        x_adv_plus = []
        x_adv_minus = []
        for i in range(0, len(x_adv)):
            diff = np.zeros(x_adv[i].reshape(-1).shape[0])
            diff[perm[i][index]] = epsilon
            diff = diff.reshape(x_adv[i].shape)
            x_adv_plus.append(np.clip(x_adv[i] + diff, 0, 1 * SCALE))
            x_adv_minus.append(np.clip(x_adv[i] - diff, 0, 1 * SCALE))

        if ENV_MODEL == 'keras':
            x_adv_plus = np.array(x_adv_plus)
            x_adv_minus = np.array(x_adv_minus)

        plus = self.classifier.predict(PREPROCESS(x_adv_plus.copy()))
        minus = self.classifier.predict(PREPROCESS(x_adv_minus.copy()))
        
        for i in range(0, len(x_adv)):
            if plus[i][np.argmax(y_pred[i])] < y_pred[i][np.argmax(y_pred[i])]:
                x_adv[i] = x_adv_plus[i]
                y_pred[i] = plus[i]
            elif minus[i][np.argmax(y_pred[i])] < y_pred[i][np.argmax(y_pred[i])]:
                x_adv[i] = x_adv_minus[i]
                y_pred[i] = minus[i]
            else:
                raise ValueError('Something went wrong...')
                pass

        return x_adv, y_pred

    def batch(self, x_adv, y_pred, perm, index, epsilon, max_workers=10, batch=50):
        """
        Single step for distributed attack.
        """
        noises = []
        for i in range(0, len(x_adv)):
            noises.append(np.zeros(x_adv[i].shape))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.step, x_adv, y_pred, perm, index+j, epsilon): j for j in range(0, batch)}
            for future in concurrent.futures.as_completed(future_to_url):
                j = future_to_url[future]
                try:
                    x_adv_new, _ = future.result()
                    for i in range(0, len(x_adv)):
                        noises[i] = noises[i] + x_adv_new[i] - x_adv[i]
                except Exception as exc:
                    print('Task %r generated an exception: %s' % (j, exc))
                else:
                    pass

        for i in range(0, len(x_adv)):

            if(np.sum(noises[i]) != 0):
                noises = proj_lp(noises[i], xi = 1)

            x_adv[i] = np.clip(x_adv[i] + noises[i], 0, 1 * SCALE)

        y_adv = self.classifier.predict(PREPROCESS(x_adv.copy())) 

        return x_adv, y_adv

    def attack(self, x, y, epsilon=0.05*SCALE, max_it=1000, distributed=False, batch=50, max_workers=10):
        """
        Initiate the attack.

        - x: input data
        - epsilon: perturbation on each pixel
        - max_it: number of iterations
        - distributed: if True, use distributed attack
        - batch: number of queries per worker
        - max_workers: number of workers
        """

        x_adv, y_pred, perm = self.init(x)

        # Continue query count
        y_pred_classes = np.argmax(y_pred, axis=1)
        correct_classified_mask = (y_pred_classes == y)
        not_dones_mask = correct_classified_mask.copy()

        print('Clean accuracy: {:.2%}'.format(np.mean(correct_classified_mask)))

        if distributed:
            pbar = tqdm(range(0, max_it, batch), desc="Distributed SimBA Attack")
        else:
            pbar = tqdm(range(0, max_it), desc="SimBA Attack")

        total_queries = np.zeros(len(x))

        for p in pbar:

            not_dones = [i for i, v in enumerate(not_dones_mask) if v]

            # x_curr = [x[idx] for idx in not_dones]
            x_adv_curr = [x_adv[idx] for idx in not_dones]
            y_curr = [y_pred[idx] for idx in not_dones]
            perm_curr = [perm[idx] for idx in not_dones]

            if ENV_MODEL == 'keras':
                # x_curr = np.array(x_curr)
                x_adv_curr = np.array(x_adv_curr)
                y_curr = np.array(y_curr)
                perm_curr = np.array(perm_curr)

            if distributed:
                x_adv_curr, y_curr = self.batch(x_adv_curr, y_curr, perm_curr, p, epsilon*SCALE, max_workers, batch)
            else:
                x_adv_curr, y_curr = self.step(x_adv_curr, y_curr, perm_curr, p, epsilon*SCALE)

            for i in range(len(not_dones)):
                x_adv[not_dones[i]] = x_adv_curr[i]
                y_pred[not_dones[i]] = y_curr[i]

            # Logging stuff
            total_queries += 2 * not_dones_mask

            y_pred_classes = np.argmax(y_pred, axis=1)
            not_dones_mask = not_dones_mask * (y_pred_classes == y)

            # max_curr_queries = total_queries.max()

            success_mask = correct_classified_mask * (1 - not_dones_mask)
            num_success = success_mask.sum()
            current_success_rate = (num_success / correct_classified_mask.sum())

            if num_success == 0:
                success_queries = -1
            else:
                success_queries = ((success_mask * total_queries).sum() / num_success)

            # pbar.set_postfix({'origin prob': y_list[-1], 'l2 norm': np.sqrt(np.power(x_adv - x, 2).sum())})
            pbar.set_postfix({'Total Queries': total_queries.sum(), 'Mean Higest Prediction': y_curr.max(axis=1).mean(), 'Success Rate': current_success_rate, 'Avg Queries': success_queries})

            # Early break
            if current_success_rate == 1.0:
                break

        return x_adv
