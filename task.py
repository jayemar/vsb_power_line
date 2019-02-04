#!/usr/bin/env python

"""Handles the Extract portion of the E-T-L pipeline

Usage:
  task.py <cfg_file>

Options:
  -h --help         Show this config
  <cfg_file>        Configuration file
"""
from docopt import docopt

import arrow
import joblib
import pdb
import pickle

from etl.dataloader import DataLoader
from etl.lib.utils import clean_args
from etl.lib.utils import handle_config

from model import build_model


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = clean_args(handle_config(args.get('<cfg_file>')))
    env_cfg = cfg.get('env_cfg', {})
    dl = DataLoader(cfg.get('env_cfg'))

    start_time = arrow.utcnow()
    print("Retrieving training data via task.py")
    print("Start time:   {}".format(start_time))
    train_gen = dl.retrieve_data(cfg.get('ml_cfg'))
    try:
        data_list, meta = next(train_gen)
        joblib.dump(meta, 'training_meta.pkl')
        joblib.dump(data_list, 'training_df.pkl')
    except Exception as err:
        print(err)
        pdb.set_trace()
    end_time = arrow.utcnow()
    print("End time:     {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))

    start_time = arrow.utcnow()
    print("Retrieving test data via task.py")
    print("Start time:   {}".format(start_time))
    test_gen = dl.get_test_data()
    try:
        data_list, meta = next(test_gen)
        joblib.dump(meta, 'test_meta.pkl')
        joblib.dump(data_list, 'test_df.pkl')
    except Exception as err:
        print(err)
        pdb.set_trace()
    end_time = arrow.utcnow()
    print("End time:     {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))
