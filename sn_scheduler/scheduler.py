#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 08:45:23 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pytz
import timezonefinder
from datetime import datetime
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astroplan import Observer
from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rf


class StarAltTime:
    def __init__(self, site_name='Cerro Pachon'):
        """
        Class to estimate star alt vs altz
        Code adapted from the example:
        https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html

        Parameters
        ----------
        site_name : str, optional
            Site name for observation. The default is 'Cerro Pachon'.

        Returns
        -------
        None.

        """

        self.timezone_str, self.site_location, self.gbt = self.time_zone(
            site_name)

    def time_zone(self, site_name):
        """
        method to get the time zone

        Parameters
        ----------
        site_name : str
            Site name.

        Returns
        -------
        timezone_str : str
            time zone.
        site_location : EarthLocation
            Site location.

        """

        obs_location = EarthLocation.of_site(site_name)
        longitude = obs_location.lon
        latitude = obs_location.lat
        altitude = obs_location.height
        site_location = EarthLocation(
            lat=latitude.degree*u.deg,
            lon=longitude.degree*u.deg, height=altitude)

        # time zone
        tf = timezonefinder.TimezoneFinder()
        timezone_str = tf.certain_timezone_at(
            lat=latitude.degree, lng=longitude.degree)

        # observer to estimate moon phases
        gbt = Observer(location=site_location, elevation=altitude,
                       name='gbt', timezone=timezone_str)

        return timezone_str, site_location, gbt

    def target_location(self, targets=pd.DataFrame()):
        """
        method to estimate target location

        Parameters
        ----------
        all_target_names : list(str), optional
            List of targets. The default is ["hd111980", "hd101452",
                                               "hd115169", "hd142331"].

        Returns
        -------
        None.

        """

        # targets
        # NTargets = len(all_target_names)

        self.all_target_locations = []
        for i, row in targets.iterrows():
            self.all_target_locations.append(
                SkyCoord(row['ra'], row['dec'], unit='deg'))

        """
        self.all_target_locations= [SkyCoord.from_name(
            target_name) for target_name in all_target_names]
        """

        self.all_target_names = targets['target'].to_list()
        self.all_target_ras = targets['ra'].to_list()
        self.all_target_decs = targets['dec'].to_list()

    def __call__(self, year=2025, month=1, day=8):
        """
        method to estimate alt vs z

        Parameters
        ----------
        year : int, optional
            Year of observation. The default is 2025.
        month : int, optional
            Month of observation. The default is 1.
        day : int, optional
            Day of observation. The default is 8.

        Returns
        -------
        None.

        """

        night_obs = datetime(year, month, day, 0, 0)
        night_obs_midnight = datetime(year, month, day, 23, 59, 59)
        night_obs_midnight_str = night_obs_midnight.strftime(
            "%Y-%m-%d %H:%M:%S")

        # datetime.datetime(year, month, day, 23, 59, 59)
        self.night_obs_str = night_obs.strftime("%Y-%m-%d %H:%M:%S")

        self.utcoffset = self.utc_offset(self.timezone_str, night_obs_midnight)

        # print('utc offset', self.utcoffset)
        # obs frame
        self.obs_frame(night_obs_midnight_str)

        # sun frame
        self.sun_frame()

        # moon frame
        self.moon_frame()

        # target frame
        self.target_frame()

        # grab mjd
        tt = Time({'year': year, 'month': month, 'day': day,
                  'hour': 0, 'minute': 0, 'second': 0.}, scale='utc')
        self.mjd = tt.mjd
        self.year = year
        self.month = month
        self.day = day
        """
        test = Time(self.mjd, format='mjd')
        print(test.ymdhms['month'])
        """

    def utc_offset(self, timezone_str, night_obs_midnight):
        """
        method to estimate the UTC offset

        Parameters
        ----------
        timezone_str : str
            time zone.
        night_obs_midnight : str
            midnight for the night of observation.

        Returns
        -------
        utcoffset : TYPE
            DESCRIPTION.

        """

        timezone = pytz.timezone(timezone_str)
        # timezone = pytz.timezone('UTC')

        dt = night_obs_midnight

        """
        print("The UTC Time at observation %s" % dt)
        print("The actual time in %s is %s" %
              (timezone_str, dt + timezone.utcoffset(dt)))
        """
        try:
            utcoffset = timezone.utcoffset(dt)
        except pytz.exceptions.AmbiguousTimeError:
            utcoffset = timezone.utcoffset(dt, is_dst=True)

        # utcoffset = timezone.utcoffset(dt).total_seconds()
        # print("The UTC offset in Chile is ", utcoffset, " hours")
        utcoffset = utcoffset.total_seconds()/60./60.
        utcoffset = utcoffset*u.hour  # need hours units

        return utcoffset

    def obs_frame(self, night_obs_midnight_str):
        """
        method to set the frame for observations

        Parameters
        ----------
        night_obs_midnight_str : str
            Midnight of the observing night.

        Returns
        -------
        None.

        """

        midnight_utc = Time(night_obs_midnight_str) - self.utcoffset

        delta_midnight = np.linspace(-2, 10, 100)*u.hour
        self.frame_night = AltAz(obstime=midnight_utc +
                                 delta_midnight, location=self.site_location)

        delta_midnight = np.linspace(-12, 12, 500)*u.hour
        times_evening_to_morning = midnight_utc + delta_midnight
        self.frame_evening_to_morning = AltAz(
            obstime=times_evening_to_morning, location=self.site_location)

        self.times_evening_to_morning = times_evening_to_morning
        self.delta_midnight = delta_midnight

    def sun_frame(self):
        """
        Method to set the Sun frame

        Returns
        -------
        None.

        """

        from astropy.coordinates import get_sun
        self.sunaltazs_evening_to_morning = get_sun(
            self.times_evening_to_morning).transform_to(self.frame_evening_to_morning)

    def moon_frame(self):
        """
        Method to set the Moon frame.

        Returns
        -------
        None.

        """

        from astropy.coordinates import get_body
        moon_evening_to_morning = get_body(
            "moon", self.times_evening_to_morning)
        self.moonaltazs_evening_to_morning = moon_evening_to_morning.transform_to(
            self.frame_evening_to_morning)

        # self.moon_phase = self.gbt.moon_phase(self.times_evening_to_morning)

        self.moon_phase = pd.DataFrame(
            (self.times_evening_to_morning-self.utcoffset).mjd, columns=['mjd'])
        self.moon_phase['moon_phase'] = self.gbt.moon_phase(
            self.times_evening_to_morning-1)/u.rad*100./np.pi

    def target_frame(self, alt_min=20.):
        """
        Method to set the target frame

        Returns
        -------
        None.

        """

        self.all_target_altazs_evening_to_morning = [tt.transform_to(
            self.frame_evening_to_morning) for tt in self.all_target_locations]

    def target_info(self, sun_alt_night=-18*u.deg,
                    star_alt_min=20.*u.deg,
                    star_alt_max=86.5*u.deg,
                    star_airmass_max=2.5):
        """
        method to get infos for targets: night_duration,obs_duration
        according to input criteria

        Parameters
        ----------
        sun_alt_night : float*u.deg, optional
            min Sun alt for the night to begin/end. The default is -18*u.deg.
        star_alt_min : float*u.deg, optional
            min alt for star observation. The default is 20.*u.deg.
        star_alt_max : float*u.deg, optional
            max alt for star observation. The default is 86.5*u.deg.
        star_airmass_max : float, optional
            Max airmass for star observation. The default is 2.0.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """

        # get begining and end of the night
        idxs = Angle(self.sunaltazs_evening_to_morning.alt) < sun_alt_night

        night_min = np.min(self.delta_midnight[idxs])
        night_max = np.max(self.delta_midnight[idxs])
        night_duration = (night_max-night_min)/u.hour

        # print('night', night_min, night_max, night_duration)

        # select obs time of the night
        idxb = self.delta_midnight >= night_min
        idxb &= self.delta_midnight <= night_max
        obs_time = self.delta_midnight[idxb]

        tmid = self.delta_midnight/u.hour

        idx = tmid < 0
        obs_time_neg = self.mjd+(tmid[idx]+24.)/24.-self.utcoffset/u.hour/24.
        obs_time_pos = self.mjd+1+tmid[~idx]/24.-self.utcoffset/u.hour/24.

        obs_time_mjd = np.concatenate((obs_time_neg, obs_time_pos))

        obs_time_mjd.sort()
        obs_time_mjd = obs_time_mjd[idxb]

        # print('night mjd', np.min(obs_time_mjd), np.max(obs_time_mjd))
        ntargets = len(self.all_target_altazs_evening_to_morning)

        res = pd.DataFrame()
        for i in np.arange(ntargets):
            target = self.all_target_altazs_evening_to_morning[i][idxb]
            # alt min and max selection
            alt = Angle(target.alt)
            idx = alt >= star_alt_min
            idx &= alt <= star_alt_max
            # airmass selection
            idx &= target.secz <= star_airmass_max

            #
            fi_sel = target[idx]
            if len(fi_sel) > 0:
                sel_time = obs_time[idx]
                sel_time_mjd = obs_time_mjd[idx]
                df_res = self.analyze_alt(sel_time, sel_time_mjd,
                                          target.alt[idx],
                                          target.secz[idx],
                                          night_min, night_max)

                df_res['target'] = self.all_target_names[i]
                df_res['ra'] = self.all_target_ras[i]
                df_res['dec'] = self.all_target_decs[i]
                df_res['night_duration [h]'] = night_duration
                df_res['obs_duration [h]'] = df_res['obs_duration_p1 [h]'] + \
                    df_res['obs_duration_p2 [h]']
                df_res['moon_phase_min'] = df_res[[
                    'moon_phase_min_p1', 'moon_phase_min_p2']].T.min()
                df_res['moon_phase_max'] = df_res[[
                    'moon_phase_max_p1', 'moon_phase_max_p2']].T.max()
                df_res['year'] = self.year
                df_res['month'] = self.month
                df_res['day'] = self.day
                df_res['mjd'] = self.mjd
                res = pd.concat((res, df_res))

                """
                min_time = np.min(sel_time)
                max_time = np.max(sel_time)
                obs_duration = (max_time-min_time)/u.hour
                med_airmass = np.median(fi_sel.secz)
                min_time_h = min_time/u.hour
                max_time_h = max_time/u.hour
                mjd_min = get_mjd(self.year, self.month, self.day, min_time_h)
                mjd_max = get_mjd(self.year, self.month, self.day, max_time_h)
            else:
                obs_duration = -1.
                med_airmass = -1.
                mjd_min = -1
                mjd_max = -1

            r.append(
                (self.all_target_names[i], self.all_target_ras[i],
                 self.all_target_decs[i],
                 night_duration, np.round(obs_duration, 3), med_airmass, mjd_min, mjd_max))
                 """

        """
        import pandas as pd
        res = pd.DataFrame(
            r, columns=['target', 'ra', 'dec', 'night_duration[h]',
                        'obs_duration[h]', 'med_airmass', 'mjd_min', 'mjd_max'])
        """

        return res

    def analyze_alt(self, obs_time, obs_time_mjd, alt, airmass,
                    night_min, night_max):
        """
        Method to analyze a night

        Parameters
        ----------
        obs_time : array
            observing time.
        obs_time_mjd : array
            obbs time mjd.
        alt : array
            alt values.
        airmass : array
            airmass value.
        night_min : float
            night min.
        night_max : float
            night max.

        Returns
        -------
        res : array
            Result.

        """

        obs_time_min, obs_time_max = np.min(
            obs_time/u.hour), np.max(obs_time/u.hour)

        obs_h = obs_time/u.hour
        alt_deg = Angle(alt)/u.deg

        # make a dfof data
        df = pd.DataFrame(obs_h.value.tolist(), columns=['time_h'])
        df['alt'] = alt_deg.value.tolist()
        df['airmass'] = airmass.value.tolist()
        df['mjd'] = obs_time_mjd.value.tolist()

        pers = periods(df.to_records(index=False))

        ddict = {}
        for pp in [1, 2]:
            ddict['mjd_per_min_p{}'.format(pp)] = 0
            ddict['mjd_per_max_p{}'.format(pp)] = 0.
            ddict['mjd_alt_max_p{}'.format(pp)] = 0.
            ddict['alt_max_p{}'.format(pp)] = 0.
            ddict['mjd_airmass_min_p{}'.format(pp)] = 0.
            ddict['airmass_min_p{}'.format(pp)] = 0.
            ddict['moon_phase_min_p{}'.format(pp)] = 1.
            ddict['moon_phase_max_p{}'.format(pp)] = 0.

        for pp in np.unique(pers['period']):
            idx = pers['period'] == pp
            sel_per = pers[idx]
            """
            min_time = np.min(sel_per['time_h'])
            max_time = np.max(sel_per['time_h'])
            print('alll', max_time-min_time)
            """
            min_time = np.min(sel_per['mjd'])
            max_time = np.max(sel_per['mjd'])

            # grab the moon phases
            idxm = self.moon_phase['mjd'] >= min_time
            idxm &= self.moon_phase['mjd'] <= max_time
            sel_moon = self.moon_phase[idxm]

            ddict['mjd_per_min_p{}'.format(pp)] = min_time
            ddict['mjd_per_max_p{}'.format(pp)] = max_time
            idx = np.argmax(sel_per['alt'])
            ddict['mjd_alt_max_p{}'.format(pp)] = sel_per[idx]['mjd']
            ddict['alt_max_p{}'.format(pp)] = sel_per[idx]['alt']
            idx = np.argmin(sel_per['airmass'])
            ddict['mjd_airmass_min_p{}'.format(pp)] = sel_per[idx]['mjd']
            ddict['airmass_min_p{}'.format(pp)] = sel_per[idx]['airmass']
            ddict['moon_phase_min_p{}'.format(
                pp)] = sel_moon['moon_phase'].min()
            ddict['moon_phase_max_p{}'.format(
                pp)] = sel_moon['moon_phase'].max()
            """
            ddict['mjd_per_min_p{}'.format(pp)] = get_mjd(
                self.year, self.month, self.day, min_time)
            ddict['mjd_per_max_p{}'.format(pp)] = get_mjd(
                self.year, self.month, self.day, max_time)
            idx = np.argmax(sel_per['alt'])
            ddict['mjd_alt_max_p{}'.format(pp)] = get_mjd(
                self.year, self.month, self.day, sel_per[idx]['time_h'])
            ddict['alt_max_p{}'.format(pp)] = sel_per[idx]['alt']
            idx = np.argmin(sel_per['airmass'])
            ddict['mjd_airmass_min_p{}'.format(pp)] = get_mjd(
                self.year, self.month, self.day, sel_per[idx]['time_h'])
            ddict['airmass_min_p{}'.format(pp)] = sel_per[idx]['airmass']
            """
        # res = pd.DataFrame.from_dict(ddict, index=[0])
        res = pd.DataFrame([ddict])
        for i in [1, 2]:
            vva = 'obs_duration_p{} [h]'.format(i)
            vvb = 'mjd_per_max_p{}'.format(i)
            vvc = 'mjd_per_min_p{}'.format(i)
            res[vva] = (res[vvb]-res[vvc])*24.
        res = res.round(3)
        return res

    def get_infos_deprecated(self, obs_time_mjd=60690.916):
        """

        Method to get infos for stars during the night

        Parameters
        ----------
        obs_time_mjd : float, optional
            MJD obs. The default is 60690.916.

        Returns
        -------
        res : pandas df
            Output data: target, mjd_altmax,alt_altmax,
            airmass_altmax,mjd_obs,alt_obs,airmass_obs
        None.

        """

        # for each target, get:

        ntargets = len(self.all_target_altazs_evening_to_morning)

        # time_obs = Time(obs_time, format='fits', out_subfmt='date_hms')
        time_obs = Time(obs_time_mjd, format='mjd')
        tag_obs = time_obs.ymdhms[3]+time_obs.ymdhms[4]/60.
        if tag_obs > 12:
            tag_obs -= 24.

        r = []
        for idx in np.arange(ntargets):

            target = self.all_target_names[idx]
            alt = self.all_target_altazs_evening_to_morning[idx].alt.deg
            airmass = self.all_target_altazs_evening_to_morning[idx].secz
            midnight_time = self.delta_midnight/u.h

            idx = np.argmax(alt)
            idxb = (np.abs(midnight_time - tag_obs)).argmin()

            midmax = midnight_time[idx].value
            day = self.day
            if midmax > 0:
                day += 1
            else:
                midmax += 24.

            max_h = int(midmax)
            max_min = int((midmax-max_h)*60.)
            max_sec = int((midmax-max_h-max_min/60.)*3600.)

            t = Time(datetime(self.year, self.month, day,
                              max_h, max_min, max_sec), scale='utc')

            r.append((target, t.mjd, alt[idx], airmass[idx].value,
                     time_obs.mjd, alt[idxb], airmass[idxb].value))

        res = pd.DataFrame(r, columns=['target', 'mjd_altmax', 'alt_altmax',
                                       'airmass_altmax', 'mjd_obs', 'alt_obs',
                                       'airmass_obs'])
        res = res.round(3)
        return res

    def plot(self, plotName='',
             sun_alt_night=-18*u.deg,
             star_alt_min=20.*u.deg,
             star_alt_max=86.5*u.deg,
             star_airmass_max=2.5,
             time_obs=None):
        """
        Method to plot the result: alt vs time

        Returns
        -------
        None.

        """

        if 'plt' not in self.__dict__.keys():
            self.activate_plot()

        plt = self.plt

        # plots
        fig, ax = plt.subplots(figsize=(16, 8))

        # draw the Sun and the Moon
        ax.plot(self.delta_midnight, self.sunaltazs_evening_to_morning.alt,
                color='r', ls=":", label='Sun')
        ax.plot(self.delta_midnight, self.moonaltazs_evening_to_morning.alt,
                color=[0.75]*3, ls='--', label='Moon')

        ax.fill_between(self.delta_midnight, 0*u.deg, 90*u.deg,
                        self.sunaltazs_evening_to_morning.alt < -0*u.deg,
                        color='0.5', zorder=0)
        ax.fill_between(self.delta_midnight, 0*u.deg, 90*u.deg,
                        self.sunaltazs_evening_to_morning.alt < -18*u.deg,
                        color='k', zorder=0)

       # selections

        idxs = Angle(self.sunaltazs_evening_to_morning.alt) < sun_alt_night

        night_min = np.min(self.delta_midnight[idxs])
        night_max = np.max(self.delta_midnight[idxs])
        idxb = self.delta_midnight >= night_min
        idxb &= self.delta_midnight <= night_max

        # draw targets

        ntargets = len(self.all_target_altazs_evening_to_morning)
        for idx in np.arange(ntargets):
            ax.plot(self.delta_midnight,
                    self.all_target_altazs_evening_to_morning[idx].alt,
                    label=self.all_target_names[idx], lw=2)

            # select target with spec
            target = self.all_target_altazs_evening_to_morning[idx][idxb]
            # alt min and max selection
            alt = Angle(target.alt)
            idxc = alt >= star_alt_min
            idxc &= alt <= star_alt_max
            # airmass selection
            idxc &= target.secz <= star_airmass_max
            ax.plot(self.delta_midnight[idxb][idxc],
                    target[idxc].alt, linestyle='None', marker='*')

        # indicate time_obs (if any)
        if time_obs is not None:
            tag_obs = time_obs.ymdhms[3]+time_obs.ymdhms[4]/60.
            if tag_obs > 12:
                tag_obs -= 24.
            ax.plot([tag_obs]*2, [0., 90.], color='w', lw=3)

        # plt.legend(loc='upper left')
        ax.legend(fontsize=12)
        ax.set_xlim(-12*u.hour, 12*u.hour)
        ax.set_xticks((np.arange(13)*2-12)*u.hour)
        ax.set_ylim(0*u.deg, 90*u.deg)
        ax.set_xlabel('Hours from Midnight local time')
        ax.set_ylabel('Altitude [deg]')
        title = "observations at Cerro Pachon - night " + \
            self.night_obs_str.split(" ")[0]
        fig.suptitle(title)
        if plotName != '':
            plt.savefig(plotName)

    def plot_airmass(self, plotName='', time_obs=None):
        """
        Method to plot the result: alt vs time

        Returns
        -------
        None.

        """

        if 'plt' not in self.__dict__.keys():
            self.activate_plot()

        plt = self.plt

        # plots
        fig, ax = plt.subplots(figsize=(16, 8))

        print(self.sunaltazs_evening_to_morning.secz)

        ax.plot(self.delta_midnight, self.sunaltazs_evening_to_morning.secz,
                color='r', ls=":", label='Sun')
        ax.plot(self.delta_midnight, self.moonaltazs_evening_to_morning.secz,
                color=[0.75]*3, ls='--', label='Moon')

        ntargets = len(self.all_target_altazs_evening_to_morning)
        for idx in np.arange(ntargets):
            ax.plot(self.delta_midnight,
                    self.all_target_altazs_evening_to_morning[idx].secz,
                    label=self.all_target_names[idx], lw=2)

        idx_sun = self.sunaltazs_evening_to_morning.alt < -0*u.deg
        ax.fill_between(self.delta_midnight[idx_sun], 1., 3.,
                        self.sunaltazs_evening_to_morning.secz[idx_sun], color='0.5', zorder=0)
        idx_night = self.sunaltazs_evening_to_morning.alt < -18*u.deg
        ax.fill_between(self.delta_midnight[idx_night], 1., 3.,
                        self.sunaltazs_evening_to_morning.secz[idx_night], color='k', zorder=0)

        # indicate time_obs (if any)
        if time_obs is not None:
            tag_obs = time_obs.ymdhms[3]+time_obs.ymdhms[4]/60.
            if tag_obs > 12:
                tag_obs -= 24.
            ax.plot([tag_obs]*2, [0., 90.], color='w', lw=3)

        # plt.legend(loc='upper left')
        ax.legend(fontsize=12)
        ax.set_xlim(-12*u.hour, 12*u.hour)
        ax.set_xticks((np.arange(13)*2-12)*u.hour)
        ax.set_ylim(1., 3.)
        ax.set_xlabel('Hours from Midnight local time')
        ax.set_ylabel('airmass')
        title = "observations at Cerro Pachon - night " + \
            self.night_obs_str.split(" ")[0]
        fig.suptitle(title)
        if plotName != '':
            plt.savefig(plotName)

    def activate_plot(self):
        """
        Method to activate matplotlib with some options

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        from astropy.visualization import astropy_mpl_style, quantity_support

        plt.style.use(astropy_mpl_style)
        quantity_support()

        plt.rcParams["axes.labelsize"] = "large"
        plt.rcParams["axes.linewidth"] = 2.0
        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["ytick.major.size"] = 8
        plt.rcParams["ytick.minor.size"] = 5
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"

        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams['axes.titlesize'] = 16
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

        self.plt = plt

        # list of self variables in self.__dict__.keys())


def get_mjd(year, month, day, time_h):
    """
    Function to estimate mjd from local hour

    Parameters
    ----------
    year : int
        year of observation.
    month : int
        month of observation.
    day : int
        day of observation.
    time_h : float
        local time of observation.

    Returns
    -------
    float
        modified julian date.

    """

    if time_h < 0.:
        th, tm, ts = get_hms(time_h+24.)
        ti = Time(datetime(year, month, day, th, tm, ts), scale='utc')
    else:
        th, tm, ts = get_hms(time_h)
        ti = Time(datetime(year, month, day+1, th, tm, ts), scale='utc')

    return ti.mjd


def get_hms(time_h):
    """
    function to get hms time from time in hour

    Parameters
    ----------
    time_h : float
        time in hour.

    Returns
    -------
    th : int
        hour.
    tm : int
        minutes.
    ts : float
        seconds.

    """

    th = int(time_h)
    tm = int((time_h-th)*60.)
    ts = int((time_h-th-tm/60.)*3600.)

    return th, tm, ts


def findIntersection(fun1, fun2, x0, xmin, xmax):
    """
    Function to find intersection using fslove

    Parameters
    ----------
    fun1 : function
        First function.
    fun2 : function
        Second function.
    x0 : float
        X value for the first iteration.
    xmin : float
        min x bound value.
    xmax : float
        max x bound value.


    Returns
    -------
    float
        The solution (None if no solution found in [xmin,xmax]).

    """
    xsol = fsolve(lambda x: fun1(x) - fun2(x), x0)

    if xsol >= xmin and xsol <= xmax:
        return xsol
    else:
        return None


def periods(obs, period_gap=0.1, colName='time_h'):
    """
    Function to estimate periods

    Parameters
    --------------
    obs: numpy array
      array of observations
    period_gap: float, opt
       minimal gap required to define a period (default: 0.1 h)
    colName : str, optional
        Name of the column to estimate the diff.
        The default is 'time_h'.
    Returns
    ----------
    original numpy array with period appended

    """

    seasoncalc = np.ones(obs.size, dtype=int)
    obs.sort(order=colName)

    if len(obs) > 1:
        diff = np.diff(obs[colName])
        flag = np.where(diff > period_gap)[0]

        if len(flag) > 0:
            for i, indx in enumerate(flag):
                seasoncalc[indx+1:] = i+2

    obs = rf.append_fields(obs, 'period', seasoncalc)
    return obs


def process_night(stars_alt, year, month, day, targets, plot_it=False):
    """


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

    """
    print(targets_info.columns)
    print(targets_info[['target', 'mjd', 'mjd_per_min_p1',
          'mjd_per_max_p1', 'obs_duration [h]']])
    """
    # plot result here
    if plot_it:
        stars_alt.plot(star_alt_min=alt_min*u.deg,
                       star_alt_max=alt_max*u.deg,
                       star_airmass_max=airmass_max)
        # stars_alt.plot_airmass()
        stars_alt.plt.show()

    return targets_info


def process_mjd(stars_alt, mjd, targets, plot_it=False):
    """
    Function to process a night (mjd)

    Parameters
    ----------
    stars_alt : StarAltTime class
        instance of the StarAltTime used for calculations.
    mjd : float
        Modified Julian Date of the night to process.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    rr : pandas df
        Result of the night processing.

    """

    tm = Time('{}'.format(mjd), format='mjd')

    print(tm.ymdhms, mjd)
    year = tm.ymdhms[0]
    month = tm.ymdhms[1]
    day = tm.ymdhms[2]

    rr = process_night(stars_alt, year, month, day, targets, plot_it=plot_it)

    return rr


def process(mjd_min, mjd_max, stars_alt, targets, plot_it=False, outDir=''):
    """
    Function to process data

    Parameters
    ----------
    mjd_min : float
        min Modified Julian Date.
    mjd_max : float
        max Modified Julian Date.
    stars_alt : StarAltTime class
        Instance of the StarAltTime class.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.
    outDir: str, optional.
        Output dir. The default is ''

    Returns
    -------
    None.

    """

    mjds = np.arange(mjd_min, mjd_max+1, 1)
    res = pd.DataFrame()
    for mjd in mjds:
        rr = process_mjd(stars_alt, mjd, targets, plot_it=plot_it)
        res = pd.concat((res, rr))

    outName = '{}/ddf_scheduler_{}_{}.hdf5'.format(
        outDir, int(mjd_min), int(mjd_max))

    res.to_hdf(outName, key='schedule')


def process_multiproc(toproc, params, j=0, output_q=None):
    """
    Function to process data using multiprocessing

    Parameters
    ----------
    toproc : list((float,float))
        list of mjds to process.
    params : dict
        parameters.
    j : int, optional
        internal int for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the results. The default is None.

    Returns
    -------
    int
        output value.

    """

    stars_alt = params['star_alt']
    targets = params['targets']
    outDir = params['outDir']

    for vv in toproc:
        print(vv[0], vv[1])
        process(vv[0], vv[1], stars_alt, targets, plot_it=False, outDir=outDir)

    if output_q is not None:
        return output_q.put({j: [1]})
    else:
        return [1]
