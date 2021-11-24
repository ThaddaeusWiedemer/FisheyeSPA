import json
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time
import os.path


def cyan():
    return sb.color_palette(['#00AF90', '#2B7AC2', '#50AAE6', '#FF9929'])  # green, blue, light blue, orange


def palegreen():
    return sb.color_palette(['#00AF90', '#82BE3C', '#2B7AC2', '#FF9929'])  # green, light green, light blue, orange


def green():
    return sb.color_palette(['#00AF90', '#00AF90', '#2B7AC2', '#FF9929'])  # green, green, light blue, orange


def default():
    return sb.color_palette(['#00AF90', '#2B7AC2', '#FF9929'])  # green, blue, orange


def shades():
    return sb.color_palette(['#179C7D', '#52E5C3', '#8CEED7', '#C5F6EB'])  # green shades


def shades2():
    return sb.color_palette(['#179C7D', '#006E92', '#52E5C3', '#25BAE2'])  # green, blue + shades


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


def keys_from_log(path: str, keys: list[str], verbose=False, alt_pattern=False, **kwargs):
    '''get all occurences of keys from a log file
    
    Arguments:
        path        (str): path to log file
        keys  (list[str]): keys to be extracted
        alt_pattern (bool): whether to use <key', value> pattern instead of <key: value>

    Returns:
        dict{'key': value}
    '''
    out = {}
    key_patterns = []
    for key in keys:
        out.update({key: []})
        if alt_pattern:
            key_patterns.append(re.compile(f"{key}', (nan|-?\d*.?\d*e?-?\d*)"))
        else:
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


def df_from_log(cols: list[str],
                keys: list[str],
                logs: list[list[str]],
                get_nth: int = 2,
                get_max: bool = False,
                **kwargs):
    '''
    Get a dataframe containing all ``get_nth`` values corresponding to ``keys`` in the log.

    If `´get_max`` is set also return the maximum value per key.

    Arguments:
        cols        (list[str]): list of columns to be added to the dataframe
        keys        (list[str]): list of keys to be extracted from the log
        logs  (list[list[str]]): list of logs to be analyzed of the form ['path', 'value_1', 'value_2', ... ],
                                 where 'value_n' is the value for the n-th column in ``cols``
        get_nth           (int): only get every n-th occurence of each key
        get_max          (bool): get max and argmax for every metric in every log as additional output

    Returns:
        pandas.DataFrame: containing the value sequence for each key
        dict(tupel(int, float)), optional: containing the (argmax, max) of each sequence
    '''
    dfs = []
    maxs = []
    for log, *col_values in logs:
        # collect keys and save in long-form dataframe
        df = pd.DataFrame(keys_from_log(log, keys, **kwargs))
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
                    try:
                        _max = df.loc[df[df['metric'] == key]['value'].idxmin()]
                        _maxs.update({key: (_max['occurrence'], _max['value'])})
                    except ValueError:
                        print(f'no values in log {log}')
                else:
                    try:
                        _max = df.loc[df[df['metric'] == key]['value'].idxmax()]
                        _maxs.update({key: (_max['occurrence'], _max['value'])})
                    except ValueError:
                        print(f'no values in log {log}')
            maxs.append(_maxs)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, maxs


def df_from_sweep(dir, sizes, splits=['a', 'b', 'c'], all_size=2357, **kwargs):
    res = []
    for n in sizes:
        if n != 'all':
            for x in splits:
                if not os.path.isfile(f'{dir}/{n}{x}.log'):
                    print(f'{dir}/{n}{x}.log does not exist, skipping it ...')
                    continue
                log = [[f'{dir}/{n}{x}.log', n, x]]
                _, maxs = df_from_log(['size', 'split'], ['bbox_mAP', 'bbox_mAP_75', 'bbox_mAP_50', 'LAMR'],
                                      log,
                                      get_max=True,
                                      **kwargs)
                _res = {'size': n, 'split': x}
                for k, v in maxs[0].items():
                    _res.update({k: v[1]})
                res.append(_res)
        else:
            log = [[f'{dir}/all.log', all_size, 'a']]
            _, maxs = df_from_log(['size', 'split'], ['bbox_mAP', 'bbox_mAP_75', 'bbox_mAP_50', 'LAMR'],
                                  log,
                                  get_max=True,
                                  **kwargs)
            _res = {'size': all_size, 'split': 'a'}
            for k, v in maxs[0].items():
                _res.update({k: v[1]})
            res.append(_res)

    df = pd.DataFrame(res)
    df = df.melt(id_vars=['size', 'split'], var_name='metric')

    return df


def smooth(y, window_length):
    '''smooth 1d data using a convolution filter of lenght ``window-length``'''
    window = np.ones(window_length) / window_length
    y_smooth = np.convolve(y, window, mode='same')
    return y_smooth
