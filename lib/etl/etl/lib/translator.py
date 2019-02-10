#!/usr/bin/env python

"""Handles the Translate portion of the E-T-L pipeline

Usage:
  translator.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help         Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

import arrow
from functools import partial
from multiprocessing import Pool
import pandas as pd
import pywt
from scipy import signal as dsp

from .etl import ETL
from .extractor import Extractor
from .utils import etl_cli
from .utils import handle_config


def classification_matrix(df):
    return [
        df[df.peak_heights > 0].shape[0],
        df[df.peak_heights < 0].shape[0],
        df.widths.max(),
        df.widths.min(),
        abs(df.peak_heights).max(),
        abs(df.peak_heights).min(),
        df.widths.mean(),
        abs(df.peak_heights).mean()
    ]


def sort_and_reindex(df, col_name='peak_heights'):
    """Sort DataFrame base on column name and reindex"""
    df = df.sort_values(col_name, inplace=False)
    df = df.reset_index(inplace=False, drop=True)
    return df


def within_ratio(h1, h2, ratio):
    """Check if h2 (height2) has the opposite sign of h1 (height1)
    and is smaller by an amount within the given ratio
    """
    resp = False
    if (h1 < 0 and h2 > 0) or (h1 > 0 and h2 < 0):
        if (((abs(h1) * (1 - ratio)) < abs(h2))
                and ((abs(h1) * (1 + ratio)) > abs(h2))):
            resp = True
    return resp


def get_peak_info(cD, args):
    peaks, pinfo = dsp.find_peaks(cD, **args)
    pinfo.update({'peak_index': peaks})

    valleys, vinfo = dsp.find_peaks(-cD, **args)
    if 'peak_heights' in vinfo:
        vinfo.update({'peak_heights': -vinfo.get('peak_heights')})
    vinfo.update({'peak_index': valleys})

    return pd.concat([pd.DataFrame(pinfo), pd.DataFrame(vinfo)])


def remove_symmetric_pulses(df, height_ratio, max_dist, train_len):
    working_df = df.copy()
    to_remove = set()
    skip_until = -1
    train_count = 0
    for idx, row in df.iterrows():
        # If indexes were already added to to_remove then we don't
        # want to check the pulses
        if idx < skip_until:
            continue
        # if the next peak is within max_dist...
        try:
            if (row.peak_index + max_dist) >= df.iloc[idx + 1].peak_index:
                # ...and if the height is within height_ratio...
                if within_ratio(row.peak_heights,
                                df.iloc[idx + 1].peak_heights,
                                height_ratio):
                    # ...remove the symmetric pulses and the pulse train
                    to_remove.update([idx, idx + 1])
                    h2_index = df.iloc[idx + 1].peak_index
                    train = df[df.peak_index.between(h2_index, h2_index + train_len)]
                    train_count += len(train)
                    skip_until = train.index.values[-1]
                    to_remove.update(train.index.values)
        except IndexError:
            # End of df
            break
    for i in to_remove:
        working_df.drop(index=i, inplace=True)
    return working_df


def wavelet_transform(batch_meta, cfg):
    """Perform wavelet transform on batch of signals and return DataFrame"""
    batch, meta = batch_meta
    wavelet = cfg.get('mother_wavelet')
    level = cfg.get('decomposition_level')
    max_height = cfg.get('max_height')
    height_ratio = cfg.get('max_height_ratio')
    max_dist = cfg.get('max_distance')
    max_peaks = cfg.get('max_ticks_removal')
    peak_args = cfg.get('peak_finder_args')

    details = list()
    for i, signal in enumerate(batch):
        cA_, *cD_ = pywt.wavedec(signal, wavelet, level=level)
        details.append(cD_[0])

    dfs = [
        sort_and_reindex(get_peak_info(cD, peak_args))
        for cD in details
    ]
    trimmed_dfs = [
        remove_symmetric_pulses(df, height_ratio, max_dist, max_peaks)
        for df in dfs
    ]
    trimmed_dfs = [
        df[abs(df.peak_heights) < max_height]
        for df in trimmed_dfs
    ]
    trimmed_dfs = [sort_and_reindex(df) for df in trimmed_dfs]

    matrices = [classification_matrix(df) for df in trimmed_dfs]
    return matrices, meta


class Translator(ETL):

    def __init__(self, env_cfg={}):
        super(Translator, self).__init__(env_cfg)
        self.env_cfg = env_cfg

    def retrieve_data(self, ml_cfg):
        """Pass config file to retrieve generator for training data"""
        self.ml_cfg = ml_cfg
        self.wvt = partial(wavelet_transform, cfg=ml_cfg)

        pool = Pool(self.env_cfg.get('translator_pool', 1))
        data_gen = self.data_in.retrieve_data(self.ml_cfg)
        for data, meta in pool.imap(self.wvt, data_gen):
            yield data, meta


    def get_test_data(self):
        """Retrieve generator for test data based on previous config"""
        pool = Pool(self.env_cfg.get('translator_pool', 1))
        data_gen = self.data_in.get_test_data()
        for data, meta in pool.imap(self.wvt, data_gen):
            yield data, meta


if __name__ == '__main__':
    args = docopt(__doc__)
    translator = Translator()
    extractor = Extractor()
    translator.set_data_input(extractor)
    etl_cli(translator, args)
