import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # Dataset: An abstract class representing a Dataset; DataLoader: represents a Python iterable over a dataset
from itertools import tee
from copy import deepcopy as dc


class MySet(Dataset): # Myset: inherits from the Dataset class
    def __init__(self): 
        super(MySet, self).__init__() # initializing the Dataset object
        self.content = open('./json/json').readlines() # open this file and store the lines read in self.content

        indices = np.arange(len(self.content)) # make indices for the lines
        val_indices = np.random.choice(indices, len(self.content) // 5) # Generates a random sample from indices, of size len(self.content) // 5: choose 1/5th samples

        self.val_indices = set(val_indices.tolist()) # set() method is used to convert any of the iterable to sequence of iterable elements with distinct elements, commonly called Set. Here, the list iterable is made a set

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx]) # json.loads: parse a valid JSON string and convert it into a Python Dictionary
        if idx in self.val_indices:
            rec['is_train'] = 0 # val_indices are for testing
        else:
            rec['is_train'] = 1 # rest are for training
        return rec

def collate_fn(recs):
#     print("Recs: ",type(recs), "of length ", len(recs), "first item keys", recs[0].keys())
    forward = list(map(lambda x: x['forward'], recs)) # map() function returns a map object(which is an iterator) of the results after applying the given function (the lambda func here) to each item of a given iterable (list, tuple etc.) (recs here)
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
#         cop = dc(recs)
        masks = torch.from_numpy(np.array(list(map(lambda r: r['masks'], recs)), dtype = np.float32))
#         print(masks.size())
#         if recs != cop: 
#             print("ALARM!!")
        deltas = torch.from_numpy(np.array(list(map(lambda r: r['deltas'], recs)), dtype = np.float32))
        values = torch.from_numpy(np.array(list(map(lambda r: r['values'], recs)), dtype = np.float32)) # make a float type tensor from the values in recs    
        
        evals = torch.from_numpy(np.array(list(map(lambda r: r['evals'], recs)), dtype = np.float32))
        eval_masks = torch.from_numpy(np.array(list(map(lambda r: r['eval_masks'], recs)), dtype = np.float32))
        forwards = torch.from_numpy(np.array(list(map(lambda r: r['forwards'], recs)), dtype = np.float32))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks} # tensor_dictionary

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)} # dictionary of dictionaries

    ret_dict['labels'] = torch.from_numpy(np.array(list(map(lambda x: x['label'], recs)), dtype = np.float32)) # collecting labels as a FloatTensor
    ret_dict['is_train'] = torch.from_numpy(np.array(list(map(lambda x: x['is_train'], recs)), dtype = np.float32)) # collecting is_train bool as a FloatTensor

    return ret_dict

def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    ) # collate_fn: Used when using batched loading from a map-style dataset.

    return data_iter
