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

from .etl import ETL
from .extractor import Extractor
from .utils import etl_cli
from .utils import handle_config
from .utils import sort_and_reindex
from .utils import within_ratio


class Translator(ETL):

    def retrieve_data(self, ml_cfg):
        self._ml_cfg = handle_config(ml_cfg)
        return self.data_in.retrieve_data(ml_cfg)


if __name__ == '__main__':
    args = docopt(__doc__)
    translator = Translator()
    extractor = Extractor()
    translator.set_data_input(extractor)
    etl_cli(translator, args)
