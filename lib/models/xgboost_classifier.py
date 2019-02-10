#!/usr/bin/env python

from xgboost import XGBClassifier


def build_model(args={}):
    if args:
        return XGBClassifier(args)
    else:
        return XGBClassifier()
