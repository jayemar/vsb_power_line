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
    print("Start time:   {}".format(start_time))
    train_gen = dl.retrieve_data(cfg.get('ml_cfg'))
    for i in range(3):
        data = next(train_gen)
        print("Retrieved training batch {}".format(i + 1))
    end_time = arrow.utcnow()
    print("End time:     {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))
