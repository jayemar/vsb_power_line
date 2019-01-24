#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from pyarrow import parquet as pq

DATA_DIR = "../data/"
TEST_START = 8712  # First 8711 lines are training data


def load_data(data_type, num_lines_per_batch):
    """data_type should be either 'train' or 'test'"""
    if data_type not in ['train', 'test']:
        raise ValueError("'data_type' must be one of 'train'/'test'")
    start = 0 if data_type == 'train' else 8712
    end = start + num_lines_per_batch
    meta_file = "metadata_{}.csv".format(data_type)
    meta = pd.read_csv(Path(DATA_DIR, meta_file))
    while True:
        data = pq.read_pandas(
            Path(DATA_DIR, data_type + '.parquet'),
            columns=[str(i) for i in range(start, end)]
        ).to_pandas().values.T
        yield data, meta.iloc[start: end]


def too_tall(df, threshold, vals=False):
    """Return index of peaks above threshold value
    Include height of peaks if vals == True
    """
    field = 'peak_heights'
    return_fields = ['peak_index', field] if vals else ['peak_index']
    return df[(abs(df[field])) > threshold][return_fields]


def pos_neg_pairs(df, threshold, hdiff, vdiff, vals=False):
    """Return index of peaks with a matching peak of the opposite polarity
    Differnce in peak heights must be within hdiff
    Differnce in sample index must be within vdiff
    Include height of peaks if vals == True
    """
    field = 'peak_heights'
    return_fields = ['peak_index', field] if vals else ['peak_index']
    tmp = df.sort_values('peak_index')[['peak_index', 'peak_heights']]
