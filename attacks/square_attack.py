import os
import gc
import numpy as np
from tqdm import tqdm
import concurrent.futures

ENV_MODEL = os.environ.get('ENV_MODEL')
ENV_MODEL_TYPE = os.environ.get('ENV_MODEL_TYPE')

if ENV_MODEL is None:
    ENV_MODEL = 'deepapi'

if ENV_MODEL_TYPE is None:
    ENV_MODEL_TYPE = 'inceptionv3'

SCALE = 255
PREPROCESS = lambda x: x

if ENV_MODEL == 'keras':
    if ENV_MODEL_TYPE == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif ENV_MODEL_TYPE == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif ENV_MODEL_TYPE == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input

    PREPROCESS = lambda x: preprocess_input(x.copy())

class SquareAttack():

    def __init__(self,  classifier):
        """
        Create a class: `SquareAttack` instance.
        - classifier: model to attack
        """
        self.min_val, self.max_val = 0.0, 1.0 * SCALE
        self.classifier = classifier

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p


    def model_loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[np.array(y)] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


    def init(self, x, y, epsilon, targeted, loss_type):
        # Initialize the attack
        x_adv = []
        for i, xi in enumerate(x):
            h, w, c = xi.shape[:]
            # [1, w, c], i.e. vertical stripes work best for untargeted attacks
            init_delta = np.random.choice([-epsilon, epsilon], size=[len(x), 1, w, c])
            x_adv.append(np.clip(xi + init_delta[i], self.min_val, self.max_val))

        if ENV_MODEL == 'keras':
            x_adv = np.array(x_adv)

        logits = self.classifier.predict(PREPROCESS(x_adv))

        n_queries = np.ones(len(x))  # ones because we have already used 1 query

        loss_min = self.model_loss(y, logits, targeted, loss_type=loss_type)
        margin_min = self.model_loss(y, logits, targeted, loss_type='margin_loss')

        return x_adv, margin_min, loss_min, n_queries


    def step(self, x, y, x_adv, margin_min, loss_min, n_queries, i_iter, n_iters, p_init, eps, targeted, loss_type):
        """ Horicontal: One step of the attack. """

        idx_to_fool = margin_min > 0
        idx_to_fool = [i for i, v in enumerate(idx_to_fool) if v]

        if len(idx_to_fool) == 0:
            return x_adv, margin_min, loss_min, n_queries

        x_curr = []
        x_adv_curr = []
        y_curr = []
        for idx in idx_to_fool:
            x_curr.append(x[idx])
            x_adv_curr.append(x_adv[idx])
            y_curr.append(y[idx])

        if ENV_MODEL == 'keras':
            x_curr = np.array(x_curr)
            x_adv_curr = np.array(x_adv_curr)
            y_curr = np.array(y_curr)

        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = [ (xa - xc) for xa, xc in zip(x_adv_curr, x_curr) ]

        p = self.p_selection(p_init, i_iter, n_iters)

        for i_img in range(len(x_adv_curr)):
            h, w, c = x[i_img].shape[:]
            n_features = c*h*w

            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img][center_h:center_h+s, center_w:center_w+s, :]
            x_best_curr_window = x_adv_curr[i_img][center_h:center_h+s, center_w:center_w+s, :]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img][center_h:center_h+s, center_w:center_w+s, :], self.min_val, self.max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img][center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[1, 1, c])

        x_new = [  np.clip(xc + d, self.min_val, self.max_val) for xc, d in zip(x_curr, deltas) ]

        if ENV_MODEL == 'keras':
            x_new = np.array(x_new)

        logits = self.classifier.predict(PREPROCESS(x_new))

        loss = self.model_loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = self.model_loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1]*3])

        for i in range(len(idx_improved)):
            x_adv[idx_to_fool[i]] = idx_improved[i] * x_new[i] + ~idx_improved[i] * x_adv_curr[i]

        n_queries[idx_to_fool] += 1

        return x_adv, margin_min, loss_min, n_queries

    def batch(self, x, y, x_adv, margin_min, loss_min, n_queries, i_iter, n_iters, p_init, eps, targeted, loss_type, concurrency):
        """ Verticle: Multiple steps of the attack. """

        idx_to_fool = margin_min > 0
        idx_to_fool = [i for i, v in enumerate(idx_to_fool) if v]

        x_curr = [x[idx] for idx in idx_to_fool]
        x_adv_curr = [x_adv[idx] for idx in idx_to_fool]
        y_curr = [y[idx] for idx in idx_to_fool]

        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]

        deltas = [ (xa - xc) for xa, xc in zip(x_adv_curr, x_curr) ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.step, x, y, x_adv, margin_min, loss_min, n_queries, i_iter, n_iters, p_init, eps, targeted, loss_type): j for j in range(0, concurrency)}
            for future in concurrent.futures.as_completed(future_to_url):
                j = future_to_url[future]
                try:
                    x_adv, _, loss_min_temp, _ = future.result()
                    idx_improved = loss_min_temp[idx_to_fool] < loss_min_curr
                    idx_improved = [i for i, v in enumerate(idx_improved) if v]

                    for idx in idx_improved:
                        deltas[idx] = deltas[idx] + x_adv[idx_to_fool[idx]] - x_adv_curr[idx]

                except Exception as e:
                    print('Task %r generated an exception: %s' % (j, e))
                else:
                    pass

        x_new = [  np.clip(xc + d, self.min_val, self.max_val) for xc, d in zip(x_curr, deltas) ]

        logits = self.classifier.predict(PREPROCESS(x_new))

        loss = self.model_loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = self.model_loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1]*3])

        for i in range(len(idx_improved)):
            x_adv[idx_to_fool[i]] = idx_improved[i] * x_new[i] + ~idx_improved[i] * x_adv_curr[i]

        n_queries[idx_to_fool] +=  (concurrency + 1)

        return x_adv, margin_min, loss_min, n_queries

    def attack(self, x, y, targeted, epsilon = 0.05, max_it = 1000, p_init = 0.05, loss_type = 'margin_loss', log_dir = None, concurrency=1): 
        """ The Linf square attack """
        n_targets = 0
        if type(x) == list:
            n_targets = len(x)
        elif type(x) == np.ndarray:
            n_targets = x.shape[0]
        else:
            raise ValueError('Input type not supported...')

        assert n_targets > 0

        assert(len(x) > 0)
        assert(len(x) == len(y))

        y_label = np.argmax(y, axis=1)

        # Tensorboard
        self.tb = None
        if log_dir is not None:
            from utils.logger import TensorBoardLogger
            self.tb = TensorBoardLogger(log_dir)

        logits_clean = self.classifier.predict(PREPROCESS(x))

        corr_classified = [(logits_clean[i].argmax() == y_label[i]) for i in range(len(x))]

        # important to check that the model was restored correctly and the clean accuracy is high
        print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)))

        if np.mean(corr_classified) == 0:
            print('No clean examples classified correctly. Aborting...')
            n_queries = np.ones(len(x))  # ones because we have already used 1 query

            mean_nq, mean_nq_ae = np.mean(n_queries), np.mean(n_queries)

            self.tb.log_scalar('Accuracy', 0, 0)
            self.tb.log_scalar('Current Accuracy', 0, 0)
            self.tb.log_scalar('Total Mean Number of Queries', mean_nq, 0)
            self.tb.log_scalar('Success Mean Number of Queries', mean_nq_ae, 0)
            self.tb.log_scalar('Average Margin', 0, 0)

            return x, n_queries

        if ENV_MODEL == 'keras':
            pbar = tqdm(range(0, max_it), desc="Non-Distributed Square Attack")
        elif ENV_MODEL == 'deepapi':
            if n_targets > 1:
                # Horizontally Distributed Attack
                pbar = tqdm(range(0, max_it - 1), desc="Distributed Square Attack (Horizontal)")
            else:
                # Vertically Distributed Attack
                pbar = tqdm(range(0, max_it - 1, concurrency), desc="Distributed Square Attack (Vertical)")
        else:
            raise ValueError('Environment not supported...')

        np.random.seed(0)  # important to leave it here as well

        # Only attack the correctly classified examples
        y = y[corr_classified]
        if type(x) == list:
            idx_corr_classified = [i for i, v in enumerate(corr_classified) if v]
            x = [xi for i, xi in enumerate(x) if i in idx_corr_classified]
        elif type(x) == np.ndarray:
            x = x[corr_classified]

        x_adv, margin_min, loss_min, n_queries = self.init(x, y, epsilon * SCALE, targeted, loss_type)

        # Main loop

        for i_iter in pbar:

            if ENV_MODEL == 'keras':
                x_adv, margin_min, loss_min, n_queries = self.step(x, y, x_adv, margin_min, loss_min, n_queries, i_iter, max_it, p_init, epsilon * SCALE, targeted, loss_type)

            elif ENV_MODEL == 'deepapi':
                if n_targets > 1:
                    # Horizontally Distributed Attack
                    x_adv, margin_min, loss_min, n_queries = self.step(x, y, x_adv, margin_min, loss_min, n_queries, i_iter, max_it, p_init, epsilon * SCALE, targeted, loss_type)
                else:
                    # Vertically Distributed Attack
                    x_adv, margin_min, loss_min, n_queries = self.batch(x, y, x_adv, margin_min, loss_min, n_queries, i_iter, max_it, p_init, epsilon * SCALE, targeted, loss_type, concurrency=concurrency)
            else:
                raise ValueError('Model type not supported...')

            acc, acc_curr, mean_nq, mean_nq_ae, median_nq_ae, avg_margin_min = self.evaluate(margin_min, n_queries, i_iter, np.sum(corr_classified))

            pbar.set_postfix({'Total Queries': n_queries.sum(), 'Average Margin': avg_margin_min, 'Attack Success Rate': 1-acc, 'Avg Queries': mean_nq_ae})

            # print('{}: acc={:.2%} acc_curr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f})'.
                # format(i_iter+1, acc, acc_curr, mean_nq_ae, median_nq_ae, avg_margin_min, len(x), eps))

            if acc == 0:
                print('\nSuceessfully found adversarial examples for all examples')
                break

            gc.collect()

        return x_adv, n_queries


    def evaluate(self, margin_min, n_queries, i_iter, n_ex_total):
        if len(margin_min) > 0 and len(n_queries) > 0:
            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_curr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
            avg_margin_min = np.mean(margin_min)

            if self.tb is not None:
                self.tb.log_scalar('Accuracy', acc, i_iter+1)
                self.tb.log_scalar('Current Accuracy', acc_curr, i_iter+1)
                self.tb.log_scalar('Total Number of Queries', n_queries.sum(), i_iter+1)
                self.tb.log_scalar('Total Mean Number of Queries', mean_nq, i_iter+1)
                self.tb.log_scalar('Success Mean Number of Queries', mean_nq_ae, i_iter+1)
                self.tb.log_scalar('Average Margin', avg_margin_min, i_iter+1)

            return acc, acc_curr, mean_nq, mean_nq_ae, median_nq_ae, avg_margin_min

        else:
            return -1, -1, -1, -1, -1, -1
