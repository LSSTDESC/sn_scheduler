#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:15:56 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
from sn_tools.sn_fp_pixel import Pixels_in_FP
# from astropy.time import Time
from sn_scheduler.scheduler import StarAltTime
# import astropy.units as u
import pandas as pd
from sn_tools.sn_utils import multiproc
from sn_tools.sn_obs import getPix
from sn_holo_survey.holo_plot import plot_mjd


class HoloSurvey:
    def __init__(self, nside, deltaRA, deltaDec,
                 fp_level,
                 targetDir, targetFile,
                 dbDir, dbName, nproc=8,
                 show_Plot=False, outFigDir=''):
        """
        class to build a Holospec (AuxTel) survey

        Parameters
        ----------
        nside : int
            nside healpix parameter.
        deltaRA : float
            RA width to grab pixels around LSST pointing.
        deltaDec : float
            Dec width to grab pixels around LSST FP.
        fp_level : str
            LSST FP level (raft,sensor,).
        targetDir : str
            Data dir for targets.
        targetFile : str
            target file name.
        dbDir : str
            Data dir for LSST observations.
        dbName : str
            OS to process.
        nproc : int, optional
            Number of procs for multiprocessing. The default is 8.
        Returns
        -------
        None.

        """

        # get vars
        self.nside = nside
        self.nproc = nproc
        self.show_Plot = show_Plot

        # StarAltTime instance
        self.stars_alt = StarAltTime()

        # Pixels_in_FP instance
        self.pix_in_fp = Pixels_in_FP(nside, deltaRA, deltaDec, level=fp_level)

        # load targets
        self.targets = self.load_targets(targetDir, targetFile)

        # target-> pixel
        self.target_pixels = self.target_to_pixel()

        print(self.target_pixels)

        # some variables

        self.df_var = ['healpixID', 'pixRA', 'pixDec', 'raft']

        if fp_level == 'ccd':
            self.df_var += ['ccd']

        if fp_level == 'sensor':
            self.df_var += ['ccd', 'sensor']

        # load observations
        self.load_obs(dbDir, dbName)

    def __call__(self):
        """
        Main method to process data

        Returns
        -------
        None.

        """

        params = {}

        res = multiproc(self.nnights, params, self.process_night, self.nproc)

        return res

    def process_night(self, nights, params, j, output_q=None):
        """
        Method to process a set of nights using multiptocessing

        Parameters
        ----------
        nights : list(int)
            List of nights to process.
        params : dict
            parameters.
        j : int
            internal tag for multiprocessing.
        output_q : multiprocessing queue, optional
            where to put the results. The default is None.

        Returns
        -------
        pandas df
            selected targets.

        """

        res = pd.DataFrame()
        # loop on nights
        for night in nights:
            # selec obs for this night
            sel_dd = self.select_obs(night)

            if sel_dd is None:
                continue

            # loop on obs and grab nearest targets
            for dd in sel_dd:
                sel_targets, ppixels = self.process_obs(dd)
                sel_targets['night'] = night
                res = pd.concat((res, sel_targets))
                if self.show_Plot:
                    plot_mjd(dd, self.nside,
                             ppixels, self.target_pixels,
                             self.stars_alt, sel_targets)

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def load_obs(self, dbDir, dbName):
        """
        Method to load observations from LSST simulation

        Parameters
        ----------
        dbDir : str
            Data dir.
        dbName : str
            OS name.

        Returns
        -------
        None.

        """

        fName = '{}/{}.npy'.format(dbDir, dbName)

        self.obs = np.load(fName)

        idx = self.obs['night'] <= 365
        self.obs = self.obs[idx]

        self.nnights = self.obs['night'].tolist()

    def load_targets(self, targetDir, targetFile):
        """
        Method to load (Gaia) targets

        Parameters
        ----------
        targetDir : str
            Target dir.
        targetFile : str
            Target file.

        Returns
        -------
        targets : pandas df
            array with targets.

        """

        # load targets

        targets = pd.read_parquet('{}/{}'.format(targetDir, targetFile))

        targets = targets.rename(
            columns={'phot_g_mean_mag': 'g_mag',
                     'phot_variable_flag': 'var_flag'})

        targets = targets.round({'ra': 2, 'dec': 2, 'g_mag': 2})
        targets['var_flag'] = targets['var_flag'].str.replace(
            'NOT_AVAILABLE', 'NA')

        return targets

    def target_to_pixel(self):
        """
        Method to grab pixels from targets

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        healpixID, pixRA, pixDec = getPix(self.nside,
                                          self.targets['ra'],
                                          self.targets['dec'])

        df = pd.DataFrame(healpixID, columns=['healpixID'])
        df['pixRA'] = pixRA
        df['pixDec'] = pixDec

        return df

    def select_obs(self, night):
        """
        Method to select observations of the night

        Parameters
        ----------
        night : int
            night number.

        Returns
        -------
        sel_dd : array
            selected obs.

        """

        idx = self.obs['night'] == night
        sel_night = self.obs[idx]

        tt = np.unique(sel_night['target_name']).tolist()
        tt = list(filter(None, tt))

        list_dd = list(filter(lambda x: 'DD' in x, tt))

        # night with no ddf
        if len(list_dd) == 0:
            return None

        # select only observations with these DDFs
        idxa = np.in1d(sel_night['target_name'], list_dd)

        sel_dd = sel_night[idxa]

        # sort by mjd
        sel_dd = np.sort(sel_dd, order=['mjd'])

        return sel_dd

    def process_obs(self, dd):
        """
        Method to process observations

        Parameters
        ----------
        dd : array
            observations.

        Returns
        -------
        targets_nearest : pandas df
            Nearest targets vs obs.

        """

        RA = dd['RA']
        Dec = dd['Dec']
        band = dd['filter']
        mjd = np.round(dd['mjd'], 3)
        field = dd['target_name'].split(':')[-1]

        # get pixels in FP
        ppixels = self.pix_in_fp(dd, RA, Dec)

        # pix_in_fp.plot_pixels_in_FP(ppixels)

        ppixels = ppixels[self.df_var]

        # grab nearest targets

        targets_nearest = self.get_targets(ppixels)

        targets_nearest['mjd'] = mjd
        targets_nearest['field'] = field

        return targets_nearest, ppixels

    def get_targets(self, pixels):
        """
        Method to grab the targets nearest to LSST FP

        Parameters
        ----------
        pixels : pandas df
            .

        Returns
        -------
        sel_targets : pandas df
            selected targets.

        """

        tt = self.targets[['source_id', 'ra', 'dec']]

        # make all possible combinations pixels/targettarget
        combis = pixels.merge(tt, how='cross')

        combis['deltaRA'] = (combis['pixRA']-combis['ra']) * \
            np.cos(np.deg2rad(combis['pixRA']))
        combis['deltaDec'] = combis['pixDec']-combis['dec']
        combis['dist'] = np.sqrt(combis['deltaRA']**2+combis['deltaDec']**2)

        combis = combis.sort_values(by='dist')

        # print(combis[['healpixID', 'source_id', 'dist']])

        tt = combis.groupby(['source_id']).apply(
            lambda x: min_dist_source(x), include_groups=False).reset_index()

        idx = tt['dist'] <= 5

        res = pd.DataFrame(tt[idx])

        res['target'] = res['source_id']

        vv = ['source_id', 'healpixID', 'pixRA', 'pixDec',
              'raft', 'deltaRA', 'deltaDec', 'dist', 'target']

        res = res[vv]

        ll = res['source_id'].to_list()

        idx = self.targets['source_id'].isin(ll)

        sel_targets = self.targets[idx]

        sel_targets = sel_targets.merge(res, left_on=['source_id'],
                                        right_on=['source_id'],
                                        suffixes=['', ''])
        del res
        return sel_targets


def min_dist_source(grp):
    """
    Function to estimate the minimal source distance

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        Output data.

    """

    grp = grp.sort_values(by=['dist'])

    return pd.DataFrame(grp[:1])
