#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier


class Model(RandomForestClassifier):
    def __init__(self, args):
        super(RandomForestClassifier, self).__init__(**args)
