#!/usr/bin/env python

from lightgbm import LGBMClassifier


def build_model(args={}):
    if args:
        return LGBMClassifier(args)
    else:
        return LGBMClassifier()
