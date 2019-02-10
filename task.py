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
import pandas as pd

from sklearn.model_selection import cross_val_score

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
    try:
        train_data = joblib.load('training_matrices.pkl')
        train_meta = joblib.load('training_meta.pkl')
        print("Re-using previously pulled training data")
    except FileNotFoundError:
        train_gen = dl.retrieve_data(cfg.get('ml_cfg'))
        try:
            train_data, train_meta = next(train_gen)
            joblib.dump(train_meta, 'training_meta.pkl')
            joblib.dump(train_data, 'training_matrices.pkl')
        except Exception as err:
            print(err)
            raise(err)

    end_time = arrow.utcnow()
    print("End time:     {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))

    start_time = arrow.utcnow()
    print("Retrieving test data via task.py")
    print("Start time:   {}".format(start_time))
    try:
        test_data = joblib.load('test_matrices.pkl')
        test_meta = joblib.load('test_meta.pkl')
        print("Re-using previously pulled test data")
    except FileNotFoundError:
        test_gen = dl.get_test_data()
        try:
            test_data, test_meta = next(test_gen)
            joblib.dump(test_meta, 'test_meta.pkl')
            joblib.dump(test_data, 'test_matrices.pkl')
        except Exception as err:
            print(err)
            raise(err)
    end_time = arrow.utcnow()
    print("End time:     {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))

    model = build_model(cfg.get('model_cfg'))

    '''
    cv_score = cross_val_score(
        model, train_data, train_meta, cv=3, scoring='recall_macro'
    )
    print(cv_score)
    '''

    training_result = model.fit(train_data, train_meta)

    predictions = model.predict(test_data).argmax(axis=1)
    # print(predictions)

    with open('submission.csv', 'w') as phil:
        phil.write('signal_id,target\n')
        for i, val in enumerate(predictions):
            phil.write("{},{}\n".format(i + 8712, val))
