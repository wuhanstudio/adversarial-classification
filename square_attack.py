from deepapi import VGG16ImageNet
import fiftyone.zoo as foz

import numpy as np
from PIL import Image

from datetime import datetime
from logger import TensorBoardLogger

N_SAMPLES = 10

DISTRIBUTED = True

# Tensorboard
log_dir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoardLogger(log_dir)

imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")
imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

def dense_to_onehot(y_test, n_classes):
    y_test_onehot = np.zeros([len(y_test), n_classes], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot

def p_selection(p_init, it, n_iters):
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

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def model_loss(y, logits, targeted=False, loss_type='margin_loss'):
    """ Implements the margin loss (difference between the correct and 2nd best class). """
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

def square_attack_linf(model, x, y, corr_classified, eps, n_iters, p_init, targeted, loss_type):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 # if x.max() <= 1 else 255

    n_ex_total = len(x) # x.shape[0]

    for i, cor in enumerate(corr_classified):
        if cor is False:
            x.pop(i)
            y.pop(i)
    # x, y = np.array(x)[corr_classified], np.array(y)[corr_classified]

    # [1, w, c], i.e. vertical stripes work best for untargeted attacks

    x_best = []
    for i, xi in enumerate(x):
        h, w, c = xi.shape[:]
        init_delta = np.random.choice([-eps, eps], size=[len(x), 1, w, c])
        x_best.append(np.clip(xi + init_delta[i], min_val, max_val))

    logits = []
    if DISTRIBUTED:
        logits = model.predictX(x_best)
    else:
        for xb in x_best:
            logits.append(model.predict(np.array([xb]))[0])
        logits = np.array(logits)

    loss_min = model_loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model_loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(len(x))  # ones because we have already used 1 query

    # time_start = time.time()

    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        idx_to_fool = [i for i, x in enumerate(idx_to_fool) if x]

        x_curr = []
        x_best_curr = []
        y_curr = []
        for idx in idx_to_fool:
            x_curr.append(x[idx])
            x_best_curr.append(x_best[idx])
            y_curr.append(y[idx])

        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = [ (xb - xc) for xb, xc in zip(x_best_curr, x_curr) ]

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(len(x_best_curr)):
            h, w, c = x[i_img].shape[:]
            n_features = c*h*w

            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img][center_h:center_h+s, center_w:center_w+s, :]
            x_best_curr_window = x_best_curr[i_img][center_h:center_h+s, center_w:center_w+s, :]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img][center_h:center_h+s, center_w:center_w+s, :], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img][center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[1, 1, c])

        x_new = [  np.clip(xc + d, min_val, max_val) for xc, d in zip(x_curr, deltas) ]

        logits = []
        if DISTRIBUTED:
            logits = model.predictX(x_new)
        else:
            for xn in x_new:
                logits.append(model.predict(np.array([xn]))[0])
            logits = np.array(logits)

        loss = model_loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model_loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1]*3])

        for i in range(len(idx_improved)):
            x_best[idx_to_fool[i]] = idx_improved[i] * x_new[i] + ~idx_improved[i] * x_best_curr[i]

        n_queries[idx_to_fool] += 1

        if len(margin_min) > 0 and len(n_queries) > 0:
            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
            avg_margin_min = np.mean(margin_min)

            # time_total = time.time() - time_start
            print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f})'.
                format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, len(x), eps))

            tb.log_scalar('acc', acc, i_iter+1)
            tb.log_scalar('acc_corr', acc_corr, i_iter+1)
            tb.log_scalar('mean_nq', mean_nq, i_iter+1)
            tb.log_scalar('avg_nq_ae', mean_nq_ae, i_iter+1)
            tb.log_scalar('median_nq_ae', median_nq_ae, i_iter+1)
            tb.log_scalar('avg_margin', margin_min.mean(), i_iter+1)

            # metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean()]

        if acc == 0:
            break

    return n_queries, x_best


if __name__ == '__main__':

    x_test = []
    y_test = []

    for sample in imagenet_dataset:
        x = Image.open(str(sample['filepath']))
        y = imagenet_labels.index(sample['ground_truth']['label'])
        
        x_test.append(np.array(x) / 255.0)
        y_test.append(y)

    x_test = x_test[:N_SAMPLES]
    y_test = y_test[:N_SAMPLES]

    model = VGG16ImageNet('http://localhost:8080/vgg16')

    logits_clean = []
    corr_classified = []

    if DISTRIBUTED:
        logits_clean = model.predictX(x_test)
        for i, x in enumerate(x_test):
            corr_classified.append((logits_clean[i].argmax() == y_test[i]))
            if logits_clean[i].argmax() != y_test[i]:
                print('Incorrectly classified: {} {} as {}'.format(i, model.imagenet_labels[y_test[i]], model.imagenet_labels[logits_clean[i].argmax()]))
    else:
        for i, x in enumerate(x_test):
            logits = model.predict(np.array([x]))[0]

            logits_clean.append(logits)
            corr_classified.append((logits.argmax() == y_test[i]))
            if logits.argmax() != y_test[i]:
                print('Incorrectly classified: {} {} as {}'.format(i, model.imagenet_labels[y_test[i]], model.imagenet_labels[logits.argmax()]))
        logits_clean = np.array(logits_clean)

    # important to check that the model was restored correctly and the clean accuracy is high
    print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)))

    y_target = y_test
    y_target_onehot = dense_to_onehot(y_target, n_classes=len(imagenet_labels))

    # Note: we count the queries only across correctly classified images
    n_queries, x_adv = square_attack_linf(model, x_test, y_target_onehot, corr_classified, 8 / 255.0, 1000,
                                        0.05, False, 'margin_loss')

    for i, x in enumerate(x_adv):
        im = Image.fromarray(np.array(np.uint8(x*255.0)))
        im_adv = Image.fromarray(np.array(np.uint8(x*255.0)))
        im.save(f"images/x_{i}.jpg")
        im_adv.save(f"images/x_{i}_adv.jpg")
