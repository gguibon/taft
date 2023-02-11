import torch
import datetime
import pandas as pd


def tprint(s):
    '''
        print datetime and s
        @params: s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)


def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue
    return data


def select_subset(old_data, new_data, keys, idx, max_len=None, shape_max=1):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''

    for k in keys:
        new_data[k] = old_data[k][idx]
        if shape_max == 1:
            if max_len is not None and len(new_data[k].shape) > shape_max:
                new_data[k] = new_data[k][:,:max_len]
    return new_data


def batch_to_cuda(data, cuda, exclude_keys=[]):
    '''
        Move all values (tensors) to cuda if specified
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def get_device(cuda):
    '''
        Returns the torch device according to GPU available and cuda indice
    '''
    if torch.cuda.is_available() and cuda != -1:
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def save_classificationreport(report, filepathname):
    '''
        Saves a scikit learn classification report (dict) as a TSV file.
        Returns:
            the pandas dataframe of the report with cool styling (for notebooks)
    '''
    df = pd.DataFrame(report).T
    df['support'] = df.support.apply(int)
    df.style.background_gradient(cmap='viridis')
    df.to_csv(filepathname, sep="\t", float_format='%.4f')
    return df