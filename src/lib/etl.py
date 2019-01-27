#!/usr/bin/env python

"""Parent class/interface for ETL classes"""

from utils import handle_config


class ETL:
    def __init__(self, env_cfg=None):
        """env_cfg can be used for any misc settings unrelated to ML"""
        self._env_cfg = handle_config(env_cfg)
        self._ml_cfg = dict()

    @property
    def env_cfg(self):
        return self._env_cfg

    @property
    def ml_cfg(self):
        return self._ml_cfg

    @property
    def data_in(self):
        return self._data_in

    def set_data_input(self, data_obj):
        """data_obj should be the upstream object passing data to this class"""
        self._data_in = data_obj

    def retrieve_data(self, ml_cfg):
        """Pass config file to retrieve generator for training data"""
        raise NotImplementedError("'retrieve_data' not implemented")

    def get_test_data(self):
        """Retrieve generator for test data based on previous config"""
        raise NotImplementedError("'get_test_data' not implemented")

    def get_validation_data(self):
        """Retrieve generator for validation data based on previous config"""
        raise NotImplementedError("'get_validation_data' not implemented")
