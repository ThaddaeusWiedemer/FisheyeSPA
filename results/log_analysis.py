import json
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time


def kit():
    return sb.color_palette(['#179C7D', '#006E92', '#25BAE2', '#EB6A0A'])  # green, blue, light blue, orange


def kit_shades():
    return sb.color_palette(['#179C7D', '#52E5C3', '#8CEED7', '#C5F6EB'])  # green shades


# possible keys for metrics are:
#   'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'roi_loss_intra', 'roi_loss_inter', 'rcnn_loss_intra',
#   'rcnn_loss_inter', 'loss'
# To get fine-grained data on the losses `log_config` `interval` should be set low, e.g.
#   log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])


def keys_from_log_epoch(path: str, keys: list[str]):
    '''get all occurences of keys and time, epoch, and batch infromation from lines containing '- mmdet -INFO - Epoch'

    Arguments:
        path        (str): path to log file
        keys  (list[str]): keys to be extracted

    Returns:
        dict{'key': value}
    '''
    line_pattern = '(.*) - mmdet - INFO - Epoch \[(\d+)\]\[(\d+).*'
    line_pattern = re.compile(line_pattern)

    out = {'time': [], 'epoch': [], 'batch': []}

    key_patterns = []
    for key in keys:
        out.update({key: []})
        key_patterns.append(re.compile(f'{key}: (nan|-?\d*.?\d*e?-?\d*),?'))

    for i, line in enumerate(open(path, 'r')):
        for match in re.finditer(line_pattern, line):
            out['time'].append(match.group(1))
            out['epoch'].append(int(match.group(2)))
            out['batch'].append(int(match.group(3)))
            for key, pattern in zip(keys, key_patterns):
                out[key].append(float(re.findall(pattern, match.group(0))[0]))
    return out


def keys_from_log(path: str, keys: list[str], verbose=False):
    '''get all occurences of keys from a log file
    
    Arguments:
        path        (str): path to log file
        keys  (list[str]): keys to be extracted

    Returns:
        dict{'key': value}
    '''
    out = {}
    key_patterns = []
    for key in keys:
        out.update({key: []})
        key_patterns.append(re.compile(f'{key}: (nan|-?\d*.?\d*e?-?\d*),?'))

    for i, line in enumerate(open(path, 'r')):
        for key, pattern in zip(keys, key_patterns):
            for match in re.finditer(pattern, line):
                try:
                    out[key].append(float(match.group(1)))
                except:
                    out[key].append('nan')
                    if verbose:
                        print('couldn\'t interpret', match.group(1), 'in line')
                        print(line)
    out.update({'occurrence': range(len(out[keys[0]]))})
    return out


def df_from_log(cols: list[str], keys: list[str], logs: list[list[str]], get_nth: int = 2, get_max: bool = False):
    '''

    Arguments:
        cols        (list[str]): list of columns to be added to the dataframe
        keys        (list[str]): list of keys to be extracted from the log
        logs  (list[list[str]]): list of logs to be analyzed of the form ['path', 'value_1', 'value_2', ... ],
                                 where 'value_n' is the value for the n-th column in ``cols``
        get_nth           (int): only get every n-th occurence of each key
        get_max          (bool): get max and argmax for every metric in every log as additional output
    '''
    dfs = []
    maxs = []
    for log, *col_values in logs:
        # collect keys and save in long-form dataframe
        df = pd.DataFrame(keys_from_log(log, keys))
        df = df.melt(id_vars=['occurrence'], var_name='metric')
        # get only every n-th occurence and label them correctly
        if get_nth is not None:
            df['occurrence'] = df['occurrence'] / get_nth + 1
            df = df.drop(df[df['occurrence'] % 1 != 0].index)
        # insert additional columns
        for i, col in enumerate(cols):
            df.insert(loc=2 + i, column=col, value=col_values[i])
        # get maximum values and position of maximum value
        if get_max:
            _maxs = {}
            for key in keys:
                # i = df[df['metric']==key]['value'].idxmax()
                # idx = df['occurrence'][i]
                # value = df[df['metric']==key]['value'][i]
                if key == 'LAMR':
                    _max = df.loc[df[df['metric'] == key]['value'].idxmin()]
                    _maxs.update({key: (_max['occurrence'], _max['value'])})
                else:
                    _max = df.loc[df[df['metric'] == key]['value'].idxmax()]
                    _maxs.update({key: (_max['occurrence'], _max['value'])})
            maxs.append(_maxs)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, maxs


def smooth(y, window_length):
    '''smooth 1d data using a convolution filter of lenght ``window-length``'''
    window = np.ones(window_length) / window_length
    y_smooth = np.convolve(y, window, mode='same')
    return y_smooth
