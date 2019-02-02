#!/usr/bin/env python

"""Handles the Extract portion of the E-T-L pipeline

Usage:
  task.py <cfg_file>

Options:
  -h --help         Show this config
  <cfg_file>        Configuration file
"""
from docopt import docopt

import yaml

from etl.dataloader import DataLoader
from etl.lib.utils import clean_args
from etl.lib.utils import handle_config

from model import build_model


if __name__ == '__main__':
    args = clean_args(docopt(__doc__))
    cfg = handle_config(args.get('cfg_file'))
    env_cfg = cfg.get('env_cfg', {})
    dl = DataLoader(args.get('<env_cfg>'))
    train_gen = dl.retrieve_data(args.get('<ml_cfg>'))
