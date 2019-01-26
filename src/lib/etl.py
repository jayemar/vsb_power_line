#!/usr/bin/env python

"""Parent class/interface for ETL classes"""

import yaml


class ETL:
    def __init__(self, cfg=None, generator=None):
        if cfg and type(cfg) == str:
            with open(cfg, 'r') as f:
                self.cfg = yaml.load(f)

