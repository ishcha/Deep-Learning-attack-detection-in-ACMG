import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 48 # For Physionet, the sequence length for each patient = 48

def binary_cross_entropy_with_logits(input1, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input1.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input1.size())) # target size must be same as input size

    max_val = (-input1).clamp(min=0) # Clamps all elements in -input into the range [ min, max ]; max = None: no upper bound
    loss = input1 - input1 * target + max_val + ((-max_val).exp() + (-input1 - max_val).exp()).log() # I dont understand how this loss is formed

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__() # initializing the NN module
        self.build(input_size) # build the model in terms of weights and biases and initialize as uniformly distributed random numbers

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size)) # define the weights of the NN using Parameter from torch, as a tensor 
        self.b = Parameter(torch.Tensor(input_size)) # define the biases of the NN using Parameter from torch, as a tensor 

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size) # complement of an identity matrix in binary system
        self.register_buffer('m', m) # Adds a persistent buffer to the module. This is typically used to register a buffer that should not to be considered a model parameter.

        self.reset_parameters()

    def reset_parameters(self): # to initialize or reset (if already initialized) model weights & biases
        stdv = 1. / math.sqrt(self.W.size(0)) # floating inverse of square root of input_size = W.size(0)
        self.W.data.uniform_(-stdv, stdv) # W.data: parameter tensor. uniform_: Fills self tensor with numbers sampled from the continuous uniform distribution
        if self.b is not None: 
            self.b.data.uniform_(-stdv, stdv) # again: fill b tensor with uniform values in the given interval. This should be when b is not none and is a Parameter

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b) # Applies a linear transformation to the incoming data; y=xA'+b
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__() # initialize the NN model
        self.diag = diag 

        self.build(input_size, output_size) # build the model in terms of weights and biases and initialize as uniformly distributed random numbers

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size)) # weights with given shape, as model parameters
        self.b = Parameter(torch.Tensor(output_size)) # biases with given shape, as model parameters

        if self.diag == True:
            assert(input_size == output_size) # assert keyword tests if a condition is true. If a condition is false, the program will stop with an optional message.
            m = torch.eye(input_size, input_size) # identity matrix with dimension given
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self): # to initialize or reset (if already initialized) model weights & biases
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d): # for eq 3 of the paper, gamma formed by this function
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b)) # Variable(self.m): means that the self.m is brought from the register_buffer and given register space as a variable
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight):
        super(Model, self).__init__() # initialize the NN model

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(35 * 2, self.rnn_hid_size) # A long short-term memory (LSTM) cell. Input size, hidden size are the 2 parameters of this cell

        self.temp_decay_h = TemporalDecay(input_size = 35, output_size = self.rnn_hid_size, diag = False) # a TemporalDecay model object with input_size = 35 (features), output_size = size of hidden layer; related to hidden layer
        self.temp_decay_x = TemporalDecay(input_size = 35, output_size = 35, diag = True) # a TemporalDecay model object with input_size = 35 (features), output_size = 35; related to input layer

        self.hist_reg = nn.Linear(self.rnn_hid_size, 35) # Applies a linear transformation to the incoming data; # in_features = self.rnn_hid_size; # out_features = 35
        self.feat_reg = FeatureRegression(35) # Make feature regression NN model with input_size = 35

        self.weight_combine = nn.Linear(35 * 2, 35) # another Linear unit

        self.dropout = nn.Dropout(p = 0.25) # Dropout layer with probability = 0.25
        self.out = nn.Linear(self.rnn_hid_size, 1) # Linear layer for output of regression (maybe)

    def forward(self, data, direct):
        ## Original sequence with 24 time steps
#         print("IM_rits_for: ", data['forward']['masks'].size())
        values = data[direct]['values'] # all of these features (values, masks, deltas, evals, eval_masks) are stored in the data object
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1) # Returns a new tensor with the same data as the self tensor but of a different shape
        is_train = data['is_train'].view(-1, 1) # Returns a new tensor with the same data as the self tensor but of a different shape

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size))) 
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
#             print("MAD")
#             print(type(values), masks.size())
            m = masks[:, t, :] # pull out all masks for time t
            d = deltas[:, t, :] # pull out all deltas for time t
            x = values[:, t, :] # pull out all values for time t
            
            

            gamma_h = self.temp_decay_h(d) # apply a forward pass on the deltas at time t, to get gamma for h
            gamma_x = self.temp_decay_x(d) # apply a forward pass on the deltas at time t, to get gamma for x

            h = h * gamma_h # initially 0

            x_h = self.hist_reg(h) # for eq 1
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5) # how much does x differ from x_h at the points of no-missing data

            x_c =  m * x +  (1 - m) * x_h # Eq2

            z_h = self.feat_reg(x_c) # Eq 7
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5) # how much does x differ from z_h at the points of no-missing data

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1)) # Eq8: not yet to beta, but beta is sigmoid(alpha)

            c_h = alpha * z_h + (1 - alpha) * x_h # Eq9
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5) # how much does x differ from c_h at the points of no-missing data

            c_c = m * x + (1 - m) * c_h # Eq10

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c)) # inputs, size of hidden layer

            imputations.append(c_c.unsqueeze(dim = 1)) # unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.

        imputations = torch.cat(imputations, dim = 1) # Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.

        y_h = self.out(h) # regression output
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False) # cross-entropy loss
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5) # put the loss only for training data

        y_h = torch.sigmoid(y_h) # sigmoid activation function on output

        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward') # forward pass on data (modified with backward in BRITS)

        if optimizer is not None:
            optimizer.zero_grad() # initialize the optimizer with 0 gradients
            ret['loss'].backward() # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x. Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call optimizer.zero_grad() after each .step() call. Note that following the first .backward call, a second call is only possible after you have performed another forward pass.
            optimizer.step() # optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.

        return ret
