#!/usr/bin/env python

"""Creates a DataLoader instance for retrieving train/test/validation data

Usage:
  dataloader.py <ml_cfg> [-r <train_count>] [-e <test_count>] [-v <validation_count>]

Options:
  -h --help             Show this config
  -r train_count        Upper limit on the number of training batches to pull
  -e test_count         Upper limit on the number of test batches to pull
  -v validation_count   Upper limit on the number of validation batches to pull
"""
from docopt import docopt

from .lib.etl import ETL
from .lib.extractor import Extractor
from .lib.loader import Loader
from .lib.translator import Translator
from .lib.utils import handle_config


class DataLoader(ETL):

    def __init__(self, env_cfg=None):
        super(DataLoader, self).__init__(env_cfg)
        self.extractor = Extractor(env_cfg)
        self.translator = Translator(env_cfg)
        self.loader = Loader(env_cfg)

        self.translator.set_data_input(self.extractor)
        self.loader.set_data_input(self.translator)

    def retrieve_data(self, ml_cfg):
        self.ml_cfg = handle_config(ml_cfg)


if __name__ == '__main__':
    args = docopt(__doc__)
    ml_cfg = args.get('<ml_cfg>')
    dataloader = DataLoader()

    count = 0
    max_count = args.get('<train_count>')
    try:
        gen = dataloader.retrieve_data(ml_cfg)
        for data, meta in gen:
            count += 1
            if max_count and count >= max_count:
                break
        batch_size = dataloader.ml_cfg.get('batch_size')
    except Exception as err:
        print("ERROR pulling training data: {}".format(err))
    print("{} training batches of size {}".format(count, batch_size))

    count = 0
    max_count = args.get('<test_count>')
    try:
        gen = dataloader.get_test_data()
        for data, meta in gen:
            count += 1
            if max_count and count >= max_count:
                break
    except Exception as err:
        print("ERROR pulling test data: {}".format(err))
    print("{} test batches of size {}".format(count, batch_size))

    count = 0
    max_count = args.get('<validation_count>')
    try:
        gen = dataloader.get_validation_data()
        for data, meta in gen:
            count += 1
            if max_count and count >= max_count:
                break
    except Exception as err:
        print("ERROR pulling validation data: {}".format(err))
    print("{} test batches of size {}".format(count, batch_size))
