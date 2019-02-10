#!/usr/bin/env python

"""Handles the Loader portion of the E-T-L pipeline

Usage:
  loader.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help         Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
import pandas as pd

from .etl import ETL
from .extractor import Extractor
from .translator import Translator
from .utils import etl_cli
from .utils import handle_config


def df_to_one_hot(df, field_name, num_labels=2):
    return np.eye(num_labels)[df[field_name]].astype('int16')


class Loader(ETL):

    def __init__(self, env_cfg={}):
        super(Loader, self).__init__(env_cfg)

    def retrieve_data(self, ml_cfg):
        """Pass config file to retrieve generator for training data"""
        self.ml_cfg = ml_cfg
        data_list = list()
        meta_df = pd.DataFrame()
        batch_count = 1
        for data, meta in self.data_in.retrieve_data(self.ml_cfg):
            data_list.extend(data)
            meta_df = pd.concat([meta_df, meta])
            batch_count += 1
        print("Concatenated {} training batches".format(batch_count))
        data_resp = np.nan_to_num(np.array(data_list))
        meta_resp = df_to_one_hot(meta_df, 'target', 2)
        data_resp, meta_resp = ADASYN().fit_resample(data_resp, meta_resp)
        meta_resp = np.squeeze(np.eye(2)[meta_resp].astype('int16'))
        yield data_resp, meta_resp

    def get_test_data(self):
        """Retrieve generator for test data based on previous config"""
        data_list = list()
        meta_df = pd.DataFrame()
        batch_count = 1
        for data, meta in self.data_in.get_test_data():
            data_list.extend(data)
            meta_df = pd.concat([meta_df, meta])
            batch_count += 1
        print("Concatenated {} test batches".format(batch_count))
        data_resp = np.nan_to_num(np.array(data_list))
        yield data_resp, meta_df

if __name__ == '__main__':
    args = docopt(__doc__)
    loader = Loader()
    translator = Translator()
    extractor = Extractor()
    translator.set_data_input(extractor)
    loader.set_data_input(translator)
    etl_cli(loader, args)
