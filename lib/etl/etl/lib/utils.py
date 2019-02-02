#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from pyarrow import parquet as pq
import yaml

DATA_DIR = "../../data/"
TEST_START = 8712  # First 8711 lines are training data


def handle_config(config):
    """Allow for th use of a file name or dict"""
    if type(config) == str:
        with open(config, 'r') as f:
            config = yaml.load(f)
    if not config:
        config = dict()
    return config


def clean_args(args):
    """Convert arg values into more pythonic names"""
    return {
        k.lstrip('-').lstrip('<').rstrip('>').replace('-', '_'):
        args[k] for k in args.keys()
    }


def get_data_generator(data_type, num_lines_per_batch):
    """data_type should be either 'train' or 'test'"""
    if data_type not in ['train', 'test']:
        raise ValueError("'data_type' must be one of 'train'/'test'")
    start = 0 if data_type == 'train' else 8712
    end = start + num_lines_per_batch
    meta_file = "metadata_{}.csv".format(data_type)
    meta_df = pd.read_csv(Path(DATA_DIR, meta_file))
    while True:
        data = pq.read_pandas(
            Path(DATA_DIR, data_type + '.parquet'),
            columns=[str(i) for i in range(start, end)]
        ).to_pandas().values.T
        meta = meta_df.iloc[start: end]
        start = end
        yield data, meta


def sort_and_reindex(df, col_name):
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


def etl_cli(instance, args):
    """Handles calling the E/T/L classes from the command line"""
    ml_cfg = handle_config(args.get('<ml_cfg>'))
    batch_size = ml_cfg.get('batch_size')

    count = 0
    max_count = int(args.get('-r', -1))
    if max_count not in [-1, 0]:
        try:
            gen = instance.retrieve_data(ml_cfg)
            for data, meta in gen:
                count += 1
                if max_count and count == max_count:
                    break
        except Exception as err:
            print("ERROR pulling training data: {}".format(err))
    print("{} training batch(es) of size {}".format(count, batch_size))

    count = 0
    max_count = int(args.get('-e', -1))
    if max_count not in [-1, 0]:
        try:
            gen = instance.get_test_data()
            for data, meta in gen:
                count += 1
                if max_count and count == max_count:
                    break
        except Exception as err:
            print("ERROR pulling test data: {}".format(err))
    print("{} test batch(es) of size {}".format(count, batch_size))

    count = 0
    max_count = int(args.get('-v', -1))
    if max_count not in [-1, 0]:
        try:
            gen = instance.get_validation_data()
            for data, meta in gen:
                count += 1
                if max_count and count == max_count:
                    break
        except Exception as err:
            print("ERROR pulling validation data: {}".format(err))
    print("{} test batch(es) of size {}".format(count, batch_size))
