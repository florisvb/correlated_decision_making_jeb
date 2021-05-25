import numpy as np
import pickle
import os
import time
import datetime
from matplotlib import patches
import multi_tracker_analysis as mta
import scipy.stats
import figurefirst as fifi
import pandas
import copy


from numpy import random, histogram2d, diff
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d


def load_data():
    df = pandas.read_hdf('data/flydata_20210428_3cam.hdf', 'flydata_20210428_3cam')


    minimum_req_visits = 3
    flids_okay = []
    for flid in df.flid.unique():
        dfq = df[df.flid==flid]
        if len(dfq) >= minimum_req_visits:
            flids_okay.append(flid)
    df = df[df.flid.isin(flids_okay)]

    df.fraction_of_time_near_odor += 1e-4 # to help with logs

    # spatial novelty
    new_camera = []
    for ix in range(len(df)):
        nc = 1
        try:
            if df.iloc[ix].flid == df.iloc[ix-1].flid:
                if df.iloc[ix].camera != df.iloc[ix-1].camera:
                    nc = 2
            else:
                nc = 2
        except:
            nc = 2 # first one
        new_camera.append(nc)
    df['new_camera'] = new_camera

    # nth visit
    nth_visit = [1]
    for ix in range(1, len(df)):
        if df.iloc[ix].flid == df.iloc[ix-1].flid:
            nth_visit.append(nth_visit[-1]+1)
        else:
            nth_visit.append(1)
    df['nth_visit'] = nth_visit
    df['log_nth_visit'] = np.log(nth_visit)

    df['mean_interval'] = np.nanmean([df['interval'].values, df['interval2'].values], axis=0)
    df['approached_odor'] = np.sign(df['fraction_of_time_near_odor']-0.01)

    # logify
    log_df = df.copy()

    for col in log_df.columns:
        if col in ['flid', 'mean_xpos', 'mean_ypos', 'dates', 'color', 'approached_odor',
                   'fraction_of_time_near_edge', 'camera', 'all_trajec_ids']:
            continue
        else:
            log_df['log_' + col] = np.log(df[col])

    for col in log_df.columns:
        log_df[col].replace(np.NINF, 0, inplace=True)
        
    log_df['camera_num'] = (log_df['camera'] == 'center_camera')
    log_df['camera_num'] = (log_df['camera_num'] -1) /2

    return log_df



