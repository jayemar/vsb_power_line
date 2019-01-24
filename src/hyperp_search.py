#!/usr/bin/env python

# # Hyper-Parameters
# 1. Wavelet
# 2. Peak height
# 3. Peak width

import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint as pp
from pyarrow import parquet as pq
import pywt
from scipy import signal as dsp
import seaborn as sns
import yaml

from utils import load_data
# from utils import pos_neg_pairs
# from utils import too_tall

sns.set()


with open("../config.yaml") as phil:
    cfg = yaml.load(phil)

for data, meta in load_data('train', cfg.get('misc_params').get('num_lines')):
    pass

idx = 0
sig = data[idx]
print("PD: {}".format(meta.iloc[idx].target))

cA, cD = pywt.dwt(sig, 'db3')

thresh_height = 2.5
thresh_width = 1
peaks, pinfo = dsp.find_peaks(cD, height=thresh_height, width=thresh_width)
valleys, vinfo = dsp.find_peaks(-cD, height=thresh_height, width=thresh_width)

pinfo.update({'peak_index': peaks})
dfp = pd.DataFrame(pinfo)

vinfo.update({'peak_heights': -vinfo.get('peak_heights')})
vinfo.update({'peak_index': valleys})
dfv = pd.DataFrame(vinfo)

df = pd.concat([dfp, dfv])
print(df.shape)
pp(df.columns)
