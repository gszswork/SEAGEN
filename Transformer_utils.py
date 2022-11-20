import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer_utils import *
from transformer.Models import get_non_pad_mask
import matplotlib.pyplot as plt



def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    #event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1])
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid, dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time):
    """ Log-likelihood of sequence. """
    # input requirements:
    # model: hawkes_module;  data: [batch_size, length, d_model];  time: [batch, length]
    all_hid = model.linear(data)

    all_lambda = softplus(all_hid, model.beta)
    #print('all_lambda shape: ', all_lambda.shape)
    type_lambda = torch.sum(all_lambda, dim=2)
    #print('type_lambda_shape: ', type_lambda.shape)
    # event log-likelihood
    #print(time.shape)
    '''
    if type_lambda.shape[1] > 50:
        x_axis = time.squeeze(dim=0).cpu().detach()
        y_axis = all_lambda.squeeze(dim=0).squeeze(dim=-1).cpu().detach()
        plt.plot(y_axis)
        plt.show()
    '''
    event_ll = compute_event(type_lambda)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll#, all_lambda.squeeze(dim=0).squeeze(dim=-1)



class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
