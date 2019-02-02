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
from model import get_model
from .utils import handle_config


with open('../config.yaml', 'r') as f:
    cfg = yaml.load(f)

model = build_model(cfg.get('model_args', {}))


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = handl
    env_cfg = args
    dl = DataLoader(args.get('<env_cfg>'))
    train_gen = dl.retrieve_data(args.get('<ml_cfg>'))
