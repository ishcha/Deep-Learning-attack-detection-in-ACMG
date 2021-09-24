# 48 occurs in several ranges in this because for one patient, there are 48 hours of observation data

import os
import re # Its primary function is to offer a search, where it takes a regular expression and a string. provides regular expression support.
import numpy as np
import pandas as pd
import json # ultra-fast JSON encoder and decoder which written in pure C language

patient_ids = [] # a list of patient ids

for filename in os.listdir('../DCMG_data'): # os.listdir: method in python is used to get the list of all files and directories in the specified directory: ./raw here; not visible as this dataset is not loaded yet
    ## the patient data in PhysioNet contains 6-digits
#     match = re.search('\d{6}', filename) # re.search() method takes a regular expression pattern and a string and searches for that pattern within the string; \d -- decimal digit [0-9]; So '\d{6}' stands for 6 digit decimal, to match patient id
#     if match:
#         id_ = match.group() # match.group() is the matching text
#         patient_ids.append(id_)

out = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death'] # By using set_index(), you can assign an existing column of pandas.DataFrame to index (row label); this is pulling out 'In-hospital_death' records
def convert(o):
    if isinstance(o, np.int64): 
        return int(o)  


## we select 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

## mean and std of 35 attributes
mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
                 147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
                 141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
                 23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
                 119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
                 119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
                 191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])

std = np.array(
    [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
     7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
     1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
     0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
     3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
     9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
     133.96778334724377])

fs = open('./json/json', 'w')

def to_time_bin(x): # extracting hr and min from time x and outputting hr as we are making hourly bins, ig
    h, m = map(int, x.split(':'))
    return h 


def parse_data(x):
    x = x.set_index('Parameter').to_dict()['Value']

    values = []

    for attr in attributes: # attributes are the 35 features pulled out
        if attr in x:
            values.append(x[attr])
        else:
            values.append(np.nan) # if the attribute is not present in x, put nan. Missing value
    return values


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1] # reversing the mask list; traversing from end to beginning

    deltas = [] # empty list for the deltas

    # difference of 1 hr between all observations
    for h in range(48): # this is like taking the previous time step value of the mask
        if h == 0: # if its the first hr of observation, starting with delta for time step = 2 (hr = 1)
            deltas.append(np.ones(35)) # appends row of 35 1s to deltas; initially, all features are assumed to be present
        else: # at other hrs of observations
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1]) # at those points where previous time step had 1, their delta is 1, for those where it was 0, there delta = 1 + delta_of_previous

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    ## only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    ## imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_id(id_):
    data = pd.read_csv('./raw/{}.txt'.format(id_)) # read the txt file for a particular patient id
    # accumulate the records within one hour
    data['Time'] = data['Time'].apply(lambda x: to_time_bin(x)) # convert the time to time bins

    evals = []

    ## merge all the metrics within one hour
    for h in range(48): 
        evals.append(parse_data(data[data['Time'] == h])) # parsing the data for a given hr value, for a given patient. 

    evals = (np.array(evals) - mean) / std # standardize the evals with the mean and the std per feature

    shp = evals.shape 

    evals = evals.reshape(-1) # Keep all in a row; like a row-major

    ## randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist() # pick out the indices (row numbers) where evals entries are not nan, and convert to a list
    indices = np.random.choice(indices, len(indices) // 10) # Generates a random sample from a given 1-D array indices of size = len(indices) // 10 -> 10% of the values which were not nan

    values = evals.copy() # create a shallow copy of evals
    values[indices] = np.nan # put nan in the indices which are chosen out for validation

    masks = ~np.isnan(values) # assign the masks to the boolean indicating which values are not nan in values array
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals)) # evaluation masks; 1s are specifically those 10% non-nan indices in evals which were made nan, or removed for validation in values.  

    evals = evals.reshape(shp) # bring back evals to original shape
    values = values.reshape(shp) # bring values to shape of evals

    masks = masks.reshape(shp) # bring masks to shape of evals
    eval_masks = eval_masks.reshape(shp) # bring eval_masks to shape of evals

    label = out.loc[int(id_)] # find the label for this patient id

    rec = {'label': label} # put this in the record dictionary, which is getting initialized here

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

#     print("STARTING")
    rec = json.dumps(rec, default=convert) # converts a Python object into a json string
#     print("ENDING")

    fs.write(rec + '\n') # the json object is written to the file. 


# do the id parsing for all the patients as below.
for id_ in patient_ids:
    print('Processing patient {}'.format(id_))
    try:
        parse_id(id_)
    except Exception as e:
        print(e)
        continue

fs.close()

