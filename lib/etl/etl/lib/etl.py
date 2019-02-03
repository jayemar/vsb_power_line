#!/usr/bin/env python

"""Parent class/interface for ETL classes"""

from pathlib import Path

from .utils import handle_config


class ETL:
    def __init__(self, env_cfg=None):
        """env_cfg can be used for any misc settings unrelated to ML"""
        self.env_cfg = handle_config(env_cfg)
        self.data_dir = Path(self.env_cfg.get('data_dir', './data'))
        self.ml_cfg = None

    def set_data_input(self, data_obj):
        """data_obj should be the upstream object passing data to this class"""
        self.data_in = data_obj

    def retrieve_data(self, ml_cfg):
        """Pass config file to retrieve generator for training data"""
        self.ml_cfg = ml_cfg
        return self.data_in.retrieve_data(ml_cfg)

    def get_test_data(self):
        """Retrieve generator for test data based on previous config"""
        return self.data_in.get_test_data()

    def get_validation_data(self):
        """Retrieve generator for validation data based on previous config"""
        return self.data_in.get_validation_data()
