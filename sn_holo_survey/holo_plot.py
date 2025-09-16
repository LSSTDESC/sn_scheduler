#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:19:17 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import healpy as hp
import astropy.units as u
from astropy.time import Time
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from astropy.visualization import astropy_mpl_style, quantity_support
plt.style.use(astropy_mpl_style)
quantity_support()

plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"

# plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['axes.facecolor'] = 'blue'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
# the line width around the marker symbol
plt.rcParams['lines.markeredgewidth'] = 0.3
plt.rcParams['lines.markersize'] = 5  # markersize, in points
plt.rcParams['grid.alpha'] = 0.75  # transparency, between 0.0 and 1.0
plt.rcParams['grid.linestyle'] = '-'  # simple line
plt.rcParams['grid.linewidth'] = 0.4  # in points
plt.rcParams['font.size'] = 13


def plot_Mollview(pixels, axa, fig, night, nside, comment='', n=6):
    """
    Method to display a Mollweid view

    Parameters
    --------------
    pixels: pandas df
      data to plots
    axa: matplotlib axis
      axis to use for the plot
    fig: matplotlib figure
      figure to use for plot
    night: int
      night number
    """

    xmin = 0.99999
    xmax = np.max([np.max(pixels['color']), 1])

    norm = plt.cm.colors.Normalize(xmin, xmax)
    # cmap = plt.get_cmap('jet', int(xmax))
    # n = int(xmax)+1
    # n = 6
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cmap = from_list(None, plt.cm.Set1(range(1, n)), n-1)
    cmap.set_under('w')

    npixels = hp.nside2npix(nside)
    hpxmap = np.zeros(npixels, dtype=int)
    hpxmap = np.full(hpxmap.shape, -2)
    hpxmap[pixels['healpixID']] = pixels['color'].astype(int)

    plt.axes(axa)
    hp.mollview(hpxmap, nest=True, cmap=cmap,
                min=xmin, max=n, norm=norm, cbar=False,
                title=comment, hold=True, badcolor='white', xsize=800)

    hp.graticule(verbose=False)

    axb = axa.inset_axes([39., -7.2, 7, 5], projection='mollweide')

    axa.indicate_inset_zoom(axb, edgecolor="black")


def plot_star_alt(stars_alt, mjd, targets, fig, ax, plt=None):
    """

    Function to plot star alts vs time

    Parameters
    ----------
    stars_alt : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    plot_it : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    targets_info : TYPE
        DESCRIPTION.

    """
    """
    Function to process targets

    Parameters
    ----------
    stars_alt : StarAltTime instance
        The class where the calculation is made.
    year : int
        year of observation.
    month : int
        month of observation.
    day : int
        day of observation.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    targets_info : pandas df
        Targets with obs. info.

    """

    # date
    year, month, day = get_date(mjd)

    # grab the targets
    stars_alt.target_location(targets=targets)

    # get stars alt
    stars_alt(year=year, month=month, day=day)

    # grab star infos
    alt_min = 25.
    alt_max = 86.5
    airmass_max = 2.5

    targets_info = stars_alt.target_info(star_alt_min=alt_min*u.deg,
                                         star_alt_max=alt_max*u.deg,
                                         star_airmass_max=airmass_max)

    # plot result here
    stars_alt.plot(star_alt_min=alt_min*u.deg,
                   star_alt_max=alt_max*u.deg,
                   star_airmass_max=airmass_max,
                   time_obs=Time(mjd, format='mjd'),
                   hour_min=-6,
                   hour_max=8,
                   fig=fig, ax=ax, myplt=plt)

    return targets_info


def plot_flat(pixels_FP, target_pixels, ra, dec,
              width_ra=3.,
              width_dec=3., fig=None, ax=None):
    """
    Function to plot pixels FP and pixels target

    Parameters
    ----------
    pixels_FP : pandas df
        LSST FP pixels.
    target_pixels : pandas df
        target pixels.
    ra : float
        ra center of the plot.
    dec : float
        dec center of the plot.
    width_ra : float, optional
        RA width for the plot. The default is 3..
    width_dec : float, optional
        Dec width for the plot. The default is 3..
    fig : matplotlib figure, optional
        figure of the plot. The default is None.
    ax : matplotlib axis, optional
        axis of the plot. The default is None.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(pixels_FP['pixRA'], pixels_FP['pixDec'], marker='o', color='r',
            mfc='None', linestyle='None', label='VRO FP')
    ax.plot(target_pixels['pixRA'],
            target_pixels['pixDec'], 'b*', label='target')

    ax.set_xlim([ra-width_ra, ra+width_ra])

    ax.set_ylim([dec-width_dec, dec+width_dec])

    ax.grid(visible=True)

    ax.set_xlabel(r'Right Ascension [deg]')
    ax.set_ylabel(r'Declination [deg]')

    ax.legend(loc='upper right', fontsize=10, frameon=False)
    """,
              bbox_to_anchor=(1.3, 0.8), fontsize=10,
              frameon=False)
    """


def get_date(mjd):
    """
    Function to get data from mjd

    Parameters
    ----------
    mjd : float
        MJD.

    Returns
    -------
    year : int
        year corresponding to mjd.
    month : int
        month corresponding to mjd.
    day : int
        day correxsponding to mjd.

    """

    ttime = Time('{}'.format(mjd), format='mjd')
    tdate = '{}'.format(ttime.datetime64)
    spa = tdate.split('T')[0].split('-')
    year = int(spa[0])
    month = int(spa[1])
    day = int(spa[2])

    return year, month, day


def plot_mjd(dd, nside,
             ppixels, target_pixels, stars_alt, targets_nearest,
             outDir='', outName=''):
    """
    Function to make a set of plots at mjd

    Parameters
    ----------
    dd: numpy array
      observation corresponding to this mjd
    ppixels : pandas df
        LSST FP pixels.
    target_pixels : pandas df
        target pixels.
    stars_alt : StartAlt class instance
        to estimate alt of targets.
    targets_nearest : pandas df
        Nearest targets to FP.

    Returns
    -------
    None.

    """

    # obs info
    RA = dd['RA']
    Dec = dd['Dec']
    band = dd['filter']
    mjd = np.round(dd['mjd'], 3)
    field = dd['target_name'].split(':')[-1]
    night = dd['night']

    # band index
    bbands = dict(zip('ugrizy', [1, 2, 3, 4, 5, 6]))

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(
        10, 10))

    fig.subplots_adjust(wspace=0.4)

    ppixels['color'] = bbands[band]

    comment = '{},night={},mjd={},{}-band'.format(field, night, mjd, band)
    target_pixels['color'] = 7
    all_pixels = pd.concat((ppixels, target_pixels))
    plot_Mollview(all_pixels, ax[0][0], fig, night, nside, comment, n=7)

    # plot pixels (flat mode)
    plot_flat(ppixels, target_pixels, RA, Dec, fig=fig, ax=ax[1][0])

    # grab the targets
    # targets_b = make_df([field], [RA], [Dec])

    # plot the targets
    rr = plot_star_alt(stars_alt, mjd, targets_nearest, fig, ax[0][1], plt)
    # set_size(5, 5, ax=ax[0][1])

    # print target results

    targets_nearest = targets_nearest.sort_values(by=['dist'])

    tp = ['source_id', 'ra', 'dec', 'g_mag', 'var_flag', 'sp_type']

    axc = ax[1][1]

    axc.axis('off')

    rr = pd.DataFrame(targets_nearest[tp])
    rr= rr.round({'ra': 2, 'dec': 2, 'g_mag': 2})
    rr['g_mag'] = rr['g_mag'].astype(str)
    #rr = rr.drop_duplicates(subset='source_id')
    print(rr)
    # rr.style.hide(axis='index')
    # rr = rr.reset_index()
    # rr = rr.to_string(index=False)
    # rr = rr.set_index('source_id')
    table = pd.plotting.table(axc, rr,
                              loc='center', cellLoc='center',
                              colWidths=[0.35]+[0.1]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.8, 1.8)
    """
    axc.table(cellText=rr.values, colLabels=rr.keys(),
              loc='center', cellLoc='center',
              colWidths=[0.35]+[0.1]*4, fontsize=20, scale=(2, 2))
    """
    # plt.tight_layout()

    if outDir == '':
        plt.show()
    else:
        fName = '{}/{}.png'.format(outDir, outName)
