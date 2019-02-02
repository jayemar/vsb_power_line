#!/usr/bin/env python

"""Handles the Extract portion of the E-T-L pipeline

Usage:
  extractor.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help         Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

from pyarrow import parquet as pq

from .etl import ETL
from .utils import etl_cli
from .utils import get_data_generator
from .utils import handle_config


class Extractor(ETL):

    def retrieve_data(self, ml_cfg):
        self._ml_cfg = handle_config(ml_cfg)
        return get_data_generator('train', self.ml_cfg.get('batch_size'))

    def get_test_data(self):
        if not self.ml_cfg:
            raise Exception("ml_cfg not yet set via retrieve_data method")
        return get_data_generator('test', self.ml_cfg.get('batch_size'))


if __name__ == '__main__':
    args = docopt(__doc__)
    extractor = Extractor()
    etl_cli(extractor, args)
