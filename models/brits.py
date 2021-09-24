import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable # torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. Variable is deprecated
from torch.nn.parameter import Parameter # A kind of Tensor that is to be considered a module parameter.

import math
import utils
import argparse
import data_loader

import rits # builds on this
from sklearn import metrics

from ipdb import set_trace

SEQ_LEN = 48 # 20000 # for 48 hr supervision of patients for Physionet imputation dataset
RNN_HID_SIZE = 64 # size of hidden layer of RNN


class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight):
        super(Model, self).__init__() # initialize the neural network model by Torch

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.rits_f = rits.Model(self.rnn_hid_size, self.impute_weight, self.label_weight) # forward RITS
        self.rits_b = rits.Model(self.rnn_hid_size, self.impute_weight, self.label_weight) # backward RITS

    def forward(self, data): 
#         print("IM_brits_for: ", data['forward']['masks'].size())
        ret_f = self.rits_f(data, 'forward') # the results from the forward RITS, when passed the data; self.rits_f is rits model, we are having __call__ function mapped to forward function in a nn module in torch
        ret_b = self.reverse(self.rits_b(data, 'backward')) # # the results from the backward RITS, when passed the data, in reverse manner

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss'] # forward loss
        loss_b = ret_b['loss'] # backward loss
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations']) # consistency loss between the imputation produced by the forward and backward passes

        loss = loss_f + loss_b + loss_c # total loss to optimize

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2 # average of forward, backward are the predictions
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2 # average of forward, backward are the imputations

        ret_f['loss'] = loss # store the loss
        ret_f['predictions'] = predictions # store the predictions
        ret_f['imputations'] = imputations # store the imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b): 
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1 # taking an error similar to mean absolute error, scaling doesn't change it much so ok
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_ # for 1 or 0 dimensional tensors, return them as it is
            indices = range(tensor_.size()[1])[::-1]  # along the second dimension of the tensor, reverse the indices (I think that it's the time axis, while the first axis was the bacth samples.)
            indices = Variable(torch.LongTensor(indices), requires_grad = False) # torch.LongTensor: 64-bit integer (signed) tensor out of the indices array; doesn't require grad

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices) # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor. So here, we are picking out the indices, along dim = 1, from where we had initially made the indices tensor

        for key in ret:
            ret[key] = reverse_tensor(ret[key]) # reverse every value, corresponding to a key in the ret dictionary

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
#         print("IM_brits: ", data['forward']['masks'].size())
        ret = self(data) # calling the forward function on the data; as forward is the only function which has only one parameter needed as argument: data

        if optimizer is not None:
            optimizer.zero_grad() # we need to explicitly set the gradients to zero before starting to do backpropragation (i.e., updation of Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.
            ret['loss'].backward() # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x. Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call optimizer.zero_grad() after each .step() call. Note that following the first .backward call, a second call is only possible after you have performed another forward pass.
            optimizer.step() # optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.

        return ret

