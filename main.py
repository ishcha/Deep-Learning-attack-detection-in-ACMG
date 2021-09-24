import copy
import torch
import torch.nn as nn # neural network from PyTorch
import torch.nn.functional as F # contains many NN utility functions
import torch.optim as optim # torch.optim is a package implementing various optimization algorithms
from torch.optim.lr_scheduler import StepLR # provides several methods to adjust the learning rate based on the number of epochs.

import numpy as np
import sys
sys.path.append('./models')
import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json
import matplotlib.pyplot as plt


from sklearn import metrics

from ipdb import set_trace # ipdb: IPython-enabled Python Debugger; 

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000) # number of epochs
parser.add_argument('--batch_size', type=int, default=32) # batch_size: number of time series coming as 
parser.add_argument('--model', type=str) # model
parser.add_argument('--hid_size', type=int) # hidden layer size
parser.add_argument('--impute_weight', type=float) # to create a tradeoff between imputation and labels losses
parser.add_argument('--label_weight', type=float) # to create a tradeoff between imputation and labels losses
args = parser.parse_args()


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer with model parameters & learning rate

    data_iter = data_loader.get_loader(batch_size=args.batch_size) # making a data_loader object with given batch size

    auc = np.zeros((args.epochs))
    mae = np.zeros((args.epochs))
    mre = np.zeros((args.epochs))
    for epoch in range(args.epochs): # train the model for given number of epochs
        model.train() # train once

        run_loss = 0.0 # loss of the model

        for idx, data in enumerate(data_iter): 
            data = utils.to_var(data) # get the batch data
#             print("IM_main: ", data['forward']['masks'].size())
            ret = model.run_on_batch(data, optimizer, epoch) # run the model on the batch data given, with optimizer and epoch number

            run_loss += ret['loss'].item() # add the loss to run_loss

#             print ('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))
        print("Epoch:", epoch)
        auc[epoch], mae[epoch], mre[epoch] = evaluate(model, data_iter)
        print()
    epo = np.arange(args.epochs)
    plt.plot(epo, auc)
    plt.plot(epo, mae)
    plt.plot(epo, mre)
    plt.legend(['auc', 'mae', 'mre'])
    plt.xlabel('epoch number')
    plt.show()


def evaluate(model, val_iter): # val_iter: data_loader object with set batch_size
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter): 
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None) # just get the results of the model on the data, no optimization is needed now, as it's testing

        ## save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        ## collect test label & prediction
        pred = pred[np.where(is_train == 0)] # this is for the predictions where we were not training
        label = label[np.where(is_train == 0)] # this is for the labels where we were not training

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32') # the actual labels
    preds = np.asarray(preds) # the predicted labels
    auc = metrics.roc_auc_score(labels, preds)
    print ('AUC {}'.format(auc)) # AUC between actual labels and preds

    evals = np.asarray(evals) # evals are the actual values, which were obscured as validation points, and imputed by the algo
    imputations = np.asarray(imputations)
    mae = np.abs(evals - imputations).mean()
    print ('MAE', mae )
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    print ('MRE', mre)

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)

    np.save('./result/{}_data'.format(args.model), save_impute)
    np.save('./result/{}_label'.format(args.model), save_label)

    return auc, mae, mre

def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight) # defining the model; getattr(): from the models, get the model passed as args.model; initialize an object of brits model; this statement is initializing the brits model object
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # all the parameters of the model
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda() # if GPU there, make the model cuda-compatible

    train(model) # train the model


if __name__ == '__main__':
    run()

