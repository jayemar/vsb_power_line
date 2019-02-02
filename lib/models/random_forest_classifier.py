#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier


def build_model(args={}):
    return RandomForestClassifier(**args)
