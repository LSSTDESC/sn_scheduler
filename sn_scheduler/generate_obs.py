#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:23:25 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from sn_tools.sn_obs import season


def load_data(dirFiles):
    """
    Fonction to load files of dirFiles

    Parameters
    ----------
    dirFiles : str
        location dir of the files.

    Returns
    -------
    ddf : pandas df
        data from the files.

    """

    fis = glob.glob('{}/*.hdf5'.format(dirFiles))

    ddf = pd.DataFrame()
    for fi in fis:
        dd = pd.read_hdf(fi)
        ddf = pd.concat((ddf, dd))

    return ddf


def get_survey(field, survey, field_type='UD'):
    """
    Function to grab the survey corresponding to a field

    Parameters
    ----------
    field : str
        field name.
    survey : pandas df
        survey.
    field_type : str, optional
        Type of field. The default is 'UD'.

    Returns
    -------
    dfb : pandas df
        survey corresponding to the field.

    """

    idx = survey['target'] == field_type
    dfb = pd.DataFrame(survey[idx])
    dfb['target'] = field

    return dfb


def process(ddf_scheduler, survey, field_type):
    """
    Function to process the targets

    Parameters
    ----------
    ddf_scheduler : pandas df
        observations.
    survey : pandas df
        survey to consider.
    field_type : dict
        field types.

    Returns
    -------
    res : pandas df
        output schedule.

    """

    targets = ddf_scheduler['target'].unique()

    res = pd.DataFrame()
    for tt in targets:
        print('processing', tt)
        idx = ddf_scheduler['target'] == tt
        target_sched = ddf_scheduler[idx]
        # get corresponding survey
        target_survey = get_survey(tt, survey, field_type[tt])
        # process
        dd = process_target(target_sched, target_survey)

        res = pd.concat((res, dd))

    return res


def process_target(target_sched, target_survey):
    """
    Function to process a target

    Parameters
    ----------
    target_sched : pandas df
        Data to process
    target_survey : pandas df
        survey to use.

    Returns
    -------
    target_sched : pandas df
        processed target data.

    """

    # grab seasons
    seas = season(target_sched.to_records(index=False),
                  season_gap=20., mjdCol='mjd')

    target_sched = pd.DataFrame.from_records(seas)

    # match seasons with number of visits
    target_sched = target_sched.merge(target_survey,
                                      left_on=['target', 'season'],
                                      right_on=['target', 'season'],
                                      suffixes=['', ''])

    if 'index' in target_sched.columns:
        target_sched = target_sched.drop(['index'])
    # select target
    target_sched = select_target(target_sched)

    return target_sched


def select_target(target_sched):
    """
    Function to select the target

    Parameters
    ----------
    target_sched : pandas df
        Data to process.

    Returns
    -------
    tt : pandas df
        selected df.

    """

    # select obs with sufficient observing time
    bands = 'ugrizy'
    target_sched['obs_time_survey [h]'] = target_sched[list(bands)].sum(axis=1)
    target_sched['obs_time_survey [h]'] *= 30./3600.
    # print(target_sched.columns)
    idx = target_sched['obs_duration [h]'] >= target_sched['obs_time_survey [h]']
    sel_target = pd.DataFrame(target_sched[idx])
    print(len(sel_target)/len(target_sched))
    """
    if 'index' in sel_target.columns:
        sel_target = sel_target.drop(['index'])
    # reduce season length to 180 days
    sel_target = sel_target.groupby(['target', 'season']).apply(
        lambda x: reduce_season_length(x)).reset_index(drop=True)
    """
    return sel_target


def reduce_season_length(grp, mjdCol='mjd', sl_max=200.):
    """
    Function to reduce the number of observations acdcording to season length

    Parameters
    ----------
    grp : pandas df
        Data to process.
    mjdCol : str, optional
        col name to estimate season length. The default is 'mjd'.
    sl_max : float, optional
        max season length. The default is 180..

    Returns
    -------
    res : pandas df
        obs corresponding to the reduced season length.

    """

    grp = grp.sort_values(by=[mjdCol])
    # get season length
    mjd_min = grp[mjdCol].min()
    mjd_max = grp[mjdCol].max()

    season_length = mjd_max-mjd_min

    if season_length < sl_max:
        res = pd.DataFrame(grp)
    else:
        mjd_season = mjd_min+sl_max
        idx = grp['mjd'] <= mjd_season
        res = pd.DataFrame(grp[idx])

    return res


def plot_moon(df):
    """
    Function to plot the moon phase vs mjd

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    targets = df['target'].unique()

    for tt in targets:
        idx = df['target'] == tt
        sel = df[idx]
        fig, ax = plt.subplots()
        fig.suptitle(tt)
        ax.plot(sel['mjd'], sel['moon_phase_min_p1'], 'ko', mfc=None)
        ax.plot(sel['mjd'], sel['moon_phase_max_p1'], 'r*')

    plt.show()


def ana_moon(df, lunar_phase=20.):
    """
    Function to set the bands of observation according to the lunar phase

    Parameters
    ----------
    df : pandas df
        Data to process.
    lunar_phase : float, optional
        Lunar phase threshold. The default is 20..

    Returns
    -------
    dft : pandas df
        Output df.

    """

    df['bands'] = 'grizy'

    idx = df['moon_phase_max_p1'] <= lunar_phase

    dfa = df[~idx]
    dfb = pd.DataFrame(df[idx])
    dfb['bands'] = 'ugriz'

    dft = pd.concat((dfa, dfb))

    return dft
