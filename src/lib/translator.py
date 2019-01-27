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

from functools import partial
import pandas as pd
import pywt
from scipy import signal as dsp

from etl import ETL
from extractor import Extractor
from utils import etl_cli
from utils import handle_config
from utils import sort_and_reindex
from utils import within_ratio


class Translator(ETL):

    def retrieve_data(self, ml_cfg):
        self._ml_cfg = handle_config(ml_cfg)
        wavelet = self.ml_cfg.get('mother_wavelet')
        level = self.ml_cfg.get('decomposition_level')
        for data, meta in self.data_in.retrieve_data(ml_cfg):
            _, signals = self.wavelet_transform(data, wavelet, level)
            dfs = [self.get_peak_info(sig) for sig in signals]
            dfs = self.remove_false_peaks(dfs)
            yield dfs, meta

    def get_test_data(self):
        wavelet = self.ml_cfg.get('mother_wavelet')
        level = self.ml_cfg.get('decomposition_level')
        for data, meta in self.data_in.get_test_data():
            _, signals = self.wavelet_transform(data, wavelet, level)
            dfs = [self.get_peak_info(sig) for sig in signals]
            dfs = self.remove_false_peaks(dfs)
            yield dfs, meta

    def _sort_peaks(self, dfs):
        """Sort and reindex DataFrame base on 'peak_heights column"""
        return [sort_and_reindex(df, 'peak_heights') for df in dfs]

    def remove_false_peaks(self, dfs):
        dfs = self._sort_peaks(dfs)
        max_height = self.ml_cfg.get('max_height')
        height_ratio = self.ml_cfg.get('max_height_ratio')
        max_dist = self.ml_cfg.get('max_distance')
        max_peaks = self.ml_cfg.get('max_ticks_removal')
        dfs = [
            self.remove_symmetric_pulses(df, height_ratio, max_dist, max_peaks)
            for df in dfs
        ]
        dfs = [df[abs(df.peak_heights) < max_height] for df in dfs]
        return self._sort_peaks(dfs)

    def wavelet_transform(self, signals, wavelet, level):
        cAs, cDs = list(), list()
        for sig in signals:
            cA, *cD = pywt.wavedec(sig, wavelet, level=level)
            cAs.append(cA)
            cDs.append(cD[0])
        return cAs, cDs

    def get_peak_info(self, cD):
        args = self.ml_cfg.get('peak_finder_args')
        peaks, pinfo = dsp.find_peaks(cD, **args)
        pinfo.update({'peak_index': peaks})

        valleys, vinfo = dsp.find_peaks(-cD, **args)
        if 'peak_heights' in vinfo:
            vinfo.update({'peak_heights': -vinfo.get('peak_heights')})
        vinfo.update({'peak_index': valleys})

        return pd.concat([pd.DataFrame(pinfo), pd.DataFrame(vinfo)])

    def remove_symmetric_pulses(self, df, height_ratio, max_dist, train_len):
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
                        train = df[df.peak_index.between(h2_index,
                                                         h2_index + train_len)]
                        train_count += len(train)
                        skip_until = train.index.values[-1]
                        to_remove.update(train.index.values)
            except IndexError:
                # End of df
                break
        for i in to_remove:
            working_df.drop(index=i, inplace=True)
        return working_df


if __name__ == '__main__':
    args = docopt(__doc__)
    translator = Translator()
    extractor = Extractor()
    translator.set_data_input(extractor)
    etl_cli(translator, args)
