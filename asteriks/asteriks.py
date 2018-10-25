"Makes light curves of moving objects in K2 data"

import numpy as np
import pandas as pd
import K2ephem
import K2fov
from tqdm import tqdm
import pickle
import logging
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from matplotlib import animation

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from scipy.interpolate import interp1d

from lightkurve import LightCurve

from . import PACKAGEDIR
from . import utils
from . import plotting
from . import query
from . import web
from .version import __version__

WEBSITE_DIR = os.path.join('/'.join(PACKAGEDIR.split('/')[0:-1]), 'docs/', 'pages/')
log = logging.getLogger('\tASTERIKS ')

class WCSFailure(Exception):
    # There are no WCS files in asteriks to use for this object
    pass

class NotBuiltError(Exception):
    # The user needs to run `build`
    pass

class NotFetchedError(Exception):
    # The user needs to run `fetch`
    pass


class object(object):
    '''Moving object
    '''

    def __init__(self, name, campaign=None, aperture_radius=8, nlagged=6, minvel_cap=0.1*u.pix/u.hour):
        self.name = name
        self.campaign = campaign
        self.time, self.cadenceno = self._get_meta()
        self.alternate_names = query.find_alternate_names_using_CAF(self.name)
        if campaign is None:
            self.campaign = self._get_campaign_number()
        self.aperture_radius = aperture_radius
        self.nlagged = nlagged
        self.minvel_cap = minvel_cap

    def __repr__(self):
        return 'asteriks.object ({})'.format(self.name)

    def fetch(self):
        ''' Fetch the asteroid data from JPL small bodies and MAST
        '''
        log.info('Fetching JPL data for {}'.format(self.name))
        self.jpl_data = self._get_radec(aperture_radius=self.aperture_radius,
                              nlagged=self.nlagged)
        self.ra = np.asarray(self.jpl_data[0].ra)
        self.dec = np.asarray(self.jpl_data[0].dec)
        self.velocity = np.asarray(self.jpl_data[0].velocity)
        log.info('Fetching MAST data for {}'.format(self.name))
        self.file_data = query.get_mast(self.name, self.campaign, timetables=self.jpl_data)
        self.nfiles = len(self.file_data)

    def _get_meta(self, cadence='long'):
        '''Load the time axis from the package meta data.

        There are stored cadenceno and JD arrays in the data directory.

        Parameters
        ----------
        cadence: str
            'long' or 'short'

        Returns
        -------
        time : numpy.ndarray
            Array of time points in JD
        cadeceno : numpy.ndarray
            Cadence numbers for campaign
        '''
        timefile = query.LC_TIME_FILE
        if cadence in ['short', 'sc']:
            log.warning('Short cadence is not currently supported. Expect odd behaviour')
            timefile = SC_TIME_FILE
        meta = pickle.load(open(timefile, 'rb'))
        if ('{}'.format(self.campaign) in meta.keys()):
            time = meta['{}'.format(self.campaign)]['time']
            cadenceno = meta['{}'.format(self.campaign)]['cadenceno']
        else:
            time = np.zeros(0)
            cadenceno = np.zeros(0)
            for m in meta.items():
                if '{}'.format(self.campaign) in m[0][0:-1]:
                    time = np.append(time, m[1]['time'])
                    cadenceno = np.append(cadenceno, m[1]['cadenceno'])
        return time, cadenceno


    def _get_campaign_number(self):
        '''Finds which campaign an object was observed in. Will return the FIRST hit.

        Parameters
        ----------
        name : str
            Asteroid name.

        Returns
        -------
        campaign : int
            Campaign number observed in. Will return the FIRST campaign.
        '''
        campaigns = []
        log.info('Finding campaign number for {}'.format(self.name))
        for c in tqdm(np.arange(1, 18), desc='Checking campaigns'):
            with utils.silence():
                df = K2ephem.get_ephemeris_dataframe(self.name, c, c, step_size=1)
            k = K2fov.getKeplerFov(c)
            onsil = np.zeros(len(df), dtype=bool)
            for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
                try:
                    onsil[i] = k.isOnSilicon(r, d, c)
                except:
                    continue
            if np.any(onsil):
                campaigns.append(c)
                log.debug('\n\tMoving object found in campaign {}'.format(c))
                break
        if len(campaigns) == 0:
            raise ValueError('{} never on Silicon'.format(self.name))
        campaign = campaigns[0]
        return campaign

    def _get_radec(self, nlagged=0, aperture_radius=3, minvel_cap=0.1*u.pix/u.hour):
        '''Finds RA and Dec of moving object using K2 ephem.

        When nlagged is specified, will interpolate the RA and Dec and find the specified
        number of lagged or leading apertures.

        This is used to create a moving aperture before and behind the target aperture.

        Parameters
        ----------
        name : str
            Name of object, resolved by JPL small bodies
        campaign : int or None
            Campaign number in K2. If None, campaigns will be stepped through until
            a campaign containing the object is reached.
        nlagged : int
            Number of lagged apertures to create. Must be even.
        aperture_radius: int
            Maximum size of aperture. This is used to ensure lagged apertures never
            overlap.
        plot : bool
            Whether or not to create a mp4 movie of the field. This will take a much
            longer to run if True.
        img_dir: str
            Path to store mp4 file
        cadence : str
            'long' or 'short'

        Returns
        -------
        dfs : list of pandas.DataFrame
            List of dataframes containing Julian Date, RA, Dec, Campaign, and channel.
            Returns one dataframe per aperture. First dataframe is always the asteroid
            aperture.
        '''

        with utils.silence():
            for altname in self.alternate_names:
                try:
                    df = K2ephem.get_ephemeris_dataframe(altname, self.campaign, self.campaign, step_size=1./(8))
                except:
                    continue
                if 'df' in locals():
                    break
        if 'df' not in locals():
            raise ValueError('Could not find ephemeris for {} in C{}.'.format(self.name, self.campaign))

        # Interpolate to the time values for the campaign.
        dftimes = [t[0:23] for t in np.asarray(df.index, dtype=str)]
        df['jd'] = Time(dftimes, format='isot').jd
        log.debug('Creating RA, Dec values for all times in campaign')
        f = interp1d(df.jd, df.ra, fill_value='extrapolate')
        ra = f(self.time) * u.deg
        f = interp1d(df.jd, df.dec, fill_value='extrapolate')
        dec = f(self.time) * u.deg
        df = pd.DataFrame(np.asarray([self.time, self.cadenceno, ra, dec]).T,
                          columns=['jd', 'cadenceno', 'ra', 'dec'])
        df['campaign'] = self.campaign

        # Find where asteroid is on silicon.
        log.debug('Finding on silicon values')
        k = K2fov.getKeplerFov(self.campaign)
        onsil = np.zeros(len(df), dtype=bool)
        for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
            try:
                onsil[i] = k.isOnSilicon(r, d, self.campaign)
            except:
                continue
        if not np.any(onsil):
            raise ValueError('{} never on Silicon in campaign {}'
                             ''.format(self.name, self.campaign))
        df['onsil'] = onsil
        onsil[np.where(onsil == True)[0][0]:np.where(onsil == True)[0][-1]] = True
        df['incampaign'] = onsil

        # Find the channel the asteroid is on at each point in time.
        log.debug('Finding channels')
        x = np.zeros((len(df), 3))
        for idx, r, d in zip(range(len(df)), df.ra, df.dec):
            try:
                x[idx, :] = k.getChannelColRow(r, d)
            except:
                continue
        df['channel'] = x[:, 0]

        # Find the lagged apertures
        df, lag, lagspacing = utils.find_lagged_apertures(df, nlagged, aperture_radius, minvel_cap)
        self.lagspacing = lagspacing
        log.debug('\n\tLag found \n {} (days)'.format(np.atleast_2d(lag).T))

        # Build a dataframe for every lagged aperture.
        dfs = []
        for l in lag:
            df1 = df.copy()
            f = interp1d(df1.jd, df1.ra, fill_value='extrapolate')
            ra = f(df1.jd + l) * u.deg
            f = interp1d(df1.jd, df1.dec, fill_value='extrapolate')
            dec = f(df1.jd + l) * u.deg
            df1['jd'] += l
            df1['ra'] = ra
            df1['dec'] = dec
            dfs.append(df1)

        # We don't need anything that wasn't in the campaign
        # Remove anything where there is no incampaign flag.
        dfs = [o[dfs[0].incampaign].reset_index(drop=True) for o in dfs]

        # Find the pixel position for every channel.
        c = np.asarray(['{:02}'.format(self.campaign) in c[0:2] for c in utils.campaign_strb])
        for jdx in range(len(dfs)):
            for b in utils.campaign_strb[c]:
                for ch in np.unique(dfs[jdx].channel).astype(int):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            wcs = pickle.load(open('{}c{}_{:02}.p'.format(utils.WCS_DIR, b, int(ch)), 'rb'))
                    except FileNotFoundError:
                        continue
                    X, Y = wcs.wcs_world2pix([[r, d] for r, d in zip(dfs[jdx].ra, dfs[jdx].dec)], 1).T
                    dfs[jdx]['Row_{0:02}_{1:02}'.format(int(b), ch)] = Y.astype(int)
                    dfs[jdx]['Column_{0:02}_{1:02}'.format(int(b), ch)] = X.astype(int)
        return dfs



    def build(self, diff_tol=5, difference=True):
        '''Download data from MAST and make arrays.
        '''
        if isinstance(self.aperture_radius, int):
            PIXEL_TOL = (self.aperture_radius**2+self.aperture_radius**2)**0.5
        if hasattr(self.aperture_radius, '__iter__'):
            if len(self.aperture_radius.shape) >= 1:
                PIXEL_TOL = (self.aperture_radius.shape[0]**2 + self.aperture_radius.shape[1]**2)**0.5
            else:
                PIXEL_TOL = (self.aperture_radius[0]**2 + self.aperture_radius[1]**2)**0.5
        log.debug('PIXEL_TOL set to {}'.format(PIXEL_TOL))
        can_difference = True
        xaper, yaper, aper = utils.build_aperture(self.aperture_radius)
        log.debug('Aperture\n {}'.format(aper))

        # Arrays to store final results
        ar = np.zeros((len(self.jpl_data[0]), aper.shape[0], aper.shape[1], len(self.jpl_data))) * np.nan
        er = np.zeros((len(self.jpl_data[0]), aper.shape[0], aper.shape[1], len(self.jpl_data))) * np.nan
        diff_ar = np.zeros((len(self.jpl_data[0]), aper.shape[0], aper.shape[1], len(self.jpl_data))) * np.nan
        diff_er = np.zeros((len(self.jpl_data[0]), aper.shape[0], aper.shape[1], len(self.jpl_data))) * np.nan
        log.debug('Arrays sized {}'.format(ar.shape))

        current_channel = -1

        # Find the positions of all the apertures at all times.
        tablecoords = [None] * len(self.jpl_data)
        for idx, obj in enumerate(self.jpl_data):
            tablecoords[idx] = SkyCoord(self.ra, self.dec, unit=(u.deg, u.deg))

        # For every file that we need to open...
        for file in tqdm(np.arange(len(self.file_data)), desc='Inflating Files\t'):
            # Find the campaign and channel
            campaign = int(self.file_data.campaign[file])
            channel = int(self.file_data.channel[file])

            # If we've switched channels load a new WCS
            if channel != current_channel:
                current_channel = np.copy(channel)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wcs = pickle.load(open('{}c{:02}_{:02}.p'.format(
                            utils.WCS_DIR, int(campaign), int(channel)), 'rb'))
                except FileNotFoundError:
                    log.error('There is no WCS file for Campaign {} Channel {}'
                              ''.format(campaign, channel))
                    raise WCSFailure

            # Open the file
            url = self.file_data.url[file]
            try:
                with utils.silence():
                    cadence, flux, error, column, row, poscorr1, poscorr2 = utils.open_tpf(url)
                if can_difference:
                    if np.all(~np.isfinite(poscorr1)) & np.all(~np.isfinite(poscorr1)):
                        can_difference = False
                        log.warning('\nThere is no POS_CORR information. Can not use difference imaging.\n')
                else:
                    if np.any(np.isfinite(poscorr1)) & np.any(np.isfinite(poscorr1)):
                        can_difference = True
                        log.warning('\nThere is POS_CORR information. Difference imaging turned on.\n')
            except OSError:
                continue

            # Get all the coordinates of the pixels
            pixel_coordinates = np.asarray(['{}, {}'.format(i, j)
                                            for i, j in zip(column.ravel(), row.ravel())])
            r, d = np.asarray(wcs.wcs_pix2world(column.ravel(), row.ravel(), 1))
            coords = SkyCoord(r, d, unit=(u.deg, u.deg))
            r, d = coords.ra, coords.dec
            # For every aperture...
            for idx, obj in enumerate(self.jpl_data):
                # Get all the coordinates across all time.
                tablecoord = tablecoords[idx]
                ok = np.zeros(len(tablecoord)).astype(bool)

                # Only use the times where we are close to the aperture.
                for coord in coords:
                    ok |= tablecoord.separation(coord) < PIXEL_TOL*4*u.arcsec
    #            if ok.any():
    #                import pdb;pdb.set_trace()
                # Pair down the table.
                tab = obj[['cadenceno', 'Column_{:02}_{:02}'.format(
                    campaign, channel), 'Row_{:02}_{:02}'.format(int(campaign), int(channel)), 'velocity']][ok]
                # log.debug('{} Near Aperture {}'.format(len(tab), idx))
                # For every time that we are near to the aperture...
                for t in tab.iterrows():
                    # Check if it's in aperture.
                    inaperture = np.asarray(['{}, {}'.format(int(i), int(j))
                                             for i, j in zip(xaper - self.aperture_radius + t[1][1], yaper - self.aperture_radius + t[1][2])])
                    mask_1 = np.asarray([i in pixel_coordinates for i in inaperture])
                    # If nothing is in aperture, then move on.
                    if not np.any(mask_1):
                        continue
                    mask_2 = np.asarray([i in inaperture for i in pixel_coordinates])

                    # Find which cadence number we're at.
                    c = np.where(cadence == int(t[1][0]))[0]
                    if len(c) == 0:
                        continue

                    # If we can difference image...then do so.
                    if can_difference & difference:
                        v = t[1][3]*u.pix/u.hour
                        timetolerance = np.round(((2 * self.aperture_radius * u.pix)/(v * 0.5*u.hour)).value)
                        clip = np.arange(c[0] - timetolerance, c[0] + timetolerance, 1).astype(int)
                        hits, flag = utils.find_overlapping_cadences(
                            c, poscorr1, poscorr2, mask=clip, tol=diff_tol)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if flag[0] == 1:
                                diff = np.nanmedian(flux[hits[0], :, :], axis=0)
                                ediff = (1./(len(hits[0]))) * \
                                    np.nansum(error[hits[0], :, :]**2, axis=0)**0.5
                                diff_ar[int(t[0]), xaper[mask_1], yaper[mask_1],
                                        idx] = (diff.ravel()[mask_2])
                                diff_er[int(t[0]), xaper[mask_1], yaper[mask_1],
                                        idx] = (ediff.ravel()[mask_2])
                    with warnings.catch_warnings():
                        # Build an undifferenced array
                        ar[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (flux[c[0]].ravel()[mask_2])
                        er[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (error[c[0]].ravel()[mask_2])
        diff_ar[diff_ar == 0] = np.nan
        diff_er[diff_er == 0] = np.nan

        self.data_array = ar
        self.error_array = er
        self.diff_array = diff_ar
        self.diff_error_array = diff_er
        self.thumb = np.nanmedian(self.data_array[:, :, :, 0] - self.diff_array[:, :, :, 0], axis=0)
        # Get rid of bright patches
        nans = np.isfinite(self.data_array[:, :, :, 0] - self.diff_array[:, :, :, 0]).sum(axis=0)
        nans[nans > np.percentile(nans[nans!=0], 20)] = 0
        nans[nans != 0] = 1
        nans = nans.astype(bool)
        self.thumb[nans] = np.nan
        # Do we need to stack the array to increase signal to noise?
        self.stack = 1
        if np.nanmax(self.thumb) < 100:
            log.warning('Faint asteroid.')
            self.thumb = np.nanmedian(plotting.stack_array(self.data_array[:, :, :, 0] - self.diff_array[:, :, :, 0]), axis=0)
            self.stack = 20
        self._build_lightcurves()

    def _build_lightcurves(self):
        '''Build light curves out of the data array.
        '''

        percs = np.arange(80, 100, 2)[::-1]
        final_lcs = {}
        apers = np.zeros((self.data_array.shape[1], self.data_array.shape[2], len(percs)))
        ts = np.asarray([self.jpl_data[i].jd for i in range(self.data_array.shape[-1])])

        apermean = np.zeros(len(percs))
        apernpoints = np.zeros(len(percs))
        for idx, perc in enumerate(percs):
            lead_lag_correction = True
            # Build the aperture out of the percentiles
            aper = (np.nan_to_num(self.thumb) > np.nanpercentile(self.thumb, perc))
            utils.fix_aperture(aper)
            if aper.sum() == 0:
                aper = (self.thumb > np.nanpercentile(self.thumb, perc))
            apers[:, :, idx] = aper
            npix = np.nansum(aper)

            # Find how many pixels drop out due to nans or traveling over the edge of the tpf
            npix_a = np.asarray([np.sum(np.isfinite(self.data_array[:, :, :, i] - self.diff_array[:, :, :, i]) * np.atleast_3d(aper).transpose([2, 0, 1]), axis=(1, 2)) for i in range(self.data_array.shape[-1])], dtype=float)
            all_pixels = npix_a[0] >= np.nanmax(npix_a)*0.8
    #        npix_a /= np.nanmax(npix_a)

            # Build all light curves
            lcs = np.asarray([np.nansum((self.data_array[:, :, :, i] - self.diff_array[:, :, :, i]) *
                                        np.atleast_3d(aper).T, axis=(1, 2)) for i in range(self.data_array.shape[-1])])
            elcs = np.asarray([np.nansum((self.error_array[:, :, :, i]**2 + self.diff_error_array[:, :, :, i]**2) *
                                        np.atleast_3d(aper).T, axis=(1, 2)) for i in range(self.data_array.shape[-1])])**0.5

            lcs[lcs==0] = np.nan
            elcs[elcs==0] = np.nan

            # Build background
            bkg_perc = np.nanpercentile((self.data_array), 15)
            bkg_aper = np.nan_to_num(self.data_array) < bkg_perc
            bkgs = np.asarray([np.nansum((self.data_array[:, :, :, i] - self.diff_array[:, :, :, i]) *
                                        (bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(self.data_array.shape[-1])])
            ebkgs = np.asarray([np.nansum((self.error_array[:, :, :, i]**2 + self.diff_error_array[:, :, :, i]**2) *
                                        (bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(self.data_array.shape[-1])])**0.5
            bkgs /= np.asarray([np.nansum((bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(self.data_array.shape[-1])])
            ebkgs /= np.asarray([np.nansum((bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(self.data_array.shape[-1])])

            # Interpolate the remaining apertures onto the same time frame as the object
            interp_lcs = np.asarray([np.interp(ts[0, :], t, lc)
                                     for t, lc in zip(ts, lcs)])
            interp_elcs = np.asarray([np.interp(ts[0, :], t, elc)
                                     for t, elc in zip(ts, elcs)])
            interp_npix_a = np.asarray([np.interp(ts[0, :], t, npix_a1)
                             for t, npix_a1 in zip(ts, npix_a)])


            # Can you USE the lead/lag apertures?
            # Must have enough pixels in the aperture (80%)
            # Must have at least 50% of lead/lag apertures available
            lead_quality = np.ones(self.data_array.shape[0], dtype=bool)
            if lead_lag_correction:
                max_pix = interp_npix_a[1:].max(axis=1)
                at_least_80 = interp_npix_a[1:] > np.atleast_2d(interp_npix_a[1:].max(axis=1)).T * 0.8
                test = np.sum(at_least_80, axis=0) >= (len(lcs) - 1) * 0.5
                if test.sum() < (0.3 * (npix_a[0] > 1).sum()):
                    log.warn('Aperture {}. Lead lag correction looks poor. Turning off.'.format(idx))
                    lead_lag_correction = False
                else:
                    lead_quality = test

            # remove timestamps where the leading/lagging apertures are
            # contaminating the object/each other.
            lead_quality &= ~self.jpl_data[0]['CONTAMINATEDAPERTUREFLAG']

            # Do the lag apertures pass?
            background_quality = np.ones(lcs.shape[1], dtype=bool)
            if lead_lag_correction:
                median = np.nanmedian(interp_lcs[1:, :], axis=0)#, lead_quality & all_pixels], axis=0)
                median_err =  (np.nansum(interp_elcs[1:, :]**2, axis=0)**0.5)/np.nansum(np.isfinite(interp_lcs[1:, :]), axis=0)#, lead_quality & all_pixels], axis=0)
                # Too much flux in the background is BAD, clip it out
                background_quality[np.abs(np.nan_to_num(median)) > 1000] = False

                # What's left? Any outliers?
                _, median1, std1 = sigma_clipped_stats(median, sigma=3, iters=2, mask = ~(lead_quality & all_pixels & background_quality))
                background_quality &= np.abs(np.nan_to_num(median - median1)) < 3 * std1

                # Are there noisy time stamps?
                std = np.nanstd(interp_lcs[1:,:], axis=0)
                _, median1, std1 = sigma_clipped_stats(std, sigma=3, iters=2, mask= ~(lead_quality & all_pixels & background_quality))
                background_quality &= np.abs(np.nan_to_num(std - median1)) < 3 * std1
            else:
                median = np.zeros(len(lcs[0, :])) * np.nan
                median_err = np.zeros(len(lcs[0, :])) * np.nan

            apermean[idx] = np.nansum(lcs[0, lead_quality & all_pixels & background_quality])
            apernpoints[idx] = len(lcs[0, lead_quality & all_pixels & background_quality])
            final_lcs[idx] = {'t': ts[0, :], 'lc': lcs[0, :],
                              'elc': elcs[0, :], 'npix': npix, 'perc': perc, 'average_background':median,
                              'average_background_err':median_err,
                              'background_quality' : background_quality, 'all_pixels' : all_pixels,
                              'lead_quality' : lead_quality, 'npix_in_aper':npix_a[0,:], 'aper':aper,
                              'lead_lag_correction':lead_lag_correction}


        self.penalty = ((1 - apernpoints/apernpoints.max())**2 + (1 - apermean/apermean.max())**2)**0.5
        self.best = self.penalty.argmin()
        self.best_aper = apers[:, : ,self.best]
        final_lcs['BEST'] = final_lcs[self.best]

        self.apers = apers
        self.lcs = final_lcs
        self.percs = percs
        self.apermean = apermean

    def writeLightCurve(self, output=None, dir=''):
        '''Write light curves to a fits file.

        Parameters
        ----------
        output : str or None
            Output file. If None, will be generated automatically.
        dir : str
            Output directory. Default is current directory.
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')

        if output is None:
            output = 'hlsp_k2movingbodies_k2_lightcurve_{1}_c{2:02}_v{3}.fits'.format(dir, self.name.replace(' ',''), self.campaign, __version__)
        ra_ar = np.interp(self.lcs[0]['t'], self.jpl_data[0].jd, self.jpl_data[0].ra)
        dec_ar = np.interp(self.lcs[0]['t'], self.jpl_data[0].jd, self.jpl_data[0].dec)

        i = 'BEST'
        hdr = fits.Header()
        hdr['ORIGIN'] = 'NASA/Ames'
        hdr['DATE'] = Time.now().isot
        hdr['CREATOR'] = 'asteriks'
        hdr['TELESCOP'] = 'Kepler'
        hdr['INSTRUME'] = 'Kepler Photometer'
        hdr['OBJECT'] = '{}'.format(self.name)
        hdr['HLSPNAME'] = 'K2MovingBodies'
        hdr['HLSPLEAD'] = 'Kepler/K2 GO Office'
        hdr['EXPSTART'] = Time(self.lcs[i]['t'][0], format='jd').isot
        hdr['EXPEND'] = Time(self.lcs[i]['t'][-1], format='jd').isot
        hdr['VERSION'] = __version__


        # BKG_QUAL : If there is evidence from lead lag that there is a background contaminant, will be False
        # LEAD_QUAL : If the lead/lag test cannot be completed, will be False
        # NPIX_QUAL : If there is not at least 80% of the aperture as non-nans, will be False
        # NPIX_APER : Number of non-NaN pixels in aperture.

        primary_hdu = fits.PrimaryHDU(header=hdr)

        hdus = [primary_hdu]
        cols = []
        cols.append(fits.Column(name='TIME', array=(self.lcs[i]['t']), format='D', unit='JD'))
        cols.append(fits.Column(name='FLUX', array=(self.lcs[i]['lc']), format='E', unit='e-/s'))
        cols.append(fits.Column(name='FLUX_ERR', array=(self.lcs[i]['elc']), format='E', unit='e-/s'))
        cols.append(fits.Column(name='BKG_FLUX', array=(self.lcs[i]['average_background']), format='E', unit='e-/s'))
        cols.append(fits.Column(name='BGFL_ERR', array=(self.lcs[i]['average_background_err']), format='E', unit='e-/s'))

        cols.append(fits.Column(name='RA_OBJ', array=ra_ar, format='E', unit='deg'))
        cols.append(fits.Column(name='DEC_OBJ', array=dec_ar, format='E', unit='deg'))
        cols.append(fits.Column(name='LEAD_QUAL', array=self.lcs[i]['lead_quality'], format='L'))
        cols.append(fits.Column(name='NPIX_QUAL', array=self.lcs[i]['all_pixels'], format='L'))
        cols.append(fits.Column(name='BKG_QUAL', array=self.lcs[i]['background_quality'], format='L'))
        cols.append(fits.Column(name='NPIX_APER', array=self.lcs[i]['npix_in_aper'], format='I'))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['EXTNAME'] = 'BESTAPER'
        hdu.header['PERC'] = '{}'.format(self.lcs[i]['perc'])
        hdu.header['NPIX'] = '{}'.format(self.lcs[i]['npix'])
        hdu.header['LEADFLAG'] = '{}'.format(self.lcs[i]['lead_lag_correction'])
        hdus.append(hdu)

        for i in range(list(self.lcs.keys())[-2]):
            cols = []
            cols.append(fits.Column(name='TIME', array=(self.lcs[i]['t']), format='D', unit='JD'))
            cols.append(fits.Column(name='FLUX', array=(self.lcs[i]['lc']), format='E', unit='e-/s'))
            cols.append(fits.Column(name='FLUX_ERR', array=(
                self.lcs[i]['elc']), format='E', unit='e-/s'))
            cols.append(fits.Column(name='BKG_FLUX', array=(self.lcs[i]['average_background']), format='E', unit='e-/s'))
            cols.append(fits.Column(name='BGFL_ERR', array=(
                self.lcs[i]['average_background_err']), format='E', unit='e-/s'))

            cols.append(fits.Column(name='RA_OBJ', array=ra_ar, format='E', unit='deg'))
            cols.append(fits.Column(name='DEC_OBJ', array=dec_ar, format='E', unit='deg'))
            cols.append(fits.Column(name='LEAD_QUAL', array=self.lcs[i]['lead_quality'], format='L'))
            cols.append(fits.Column(name='NPIX_QUAL', array=self.lcs[i]['all_pixels'], format='L'))
            cols.append(fits.Column(name='BKG_QUAL', array=self.lcs[i]['background_quality'], format='L'))
            cols.append(fits.Column(name='NPIX_APER', array=self.lcs[i]['npix_in_aper'], format='I'))

            cols = fits.ColDefs(cols)
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.header['EXTNAME'] = 'PERC{}'.format(self.lcs[i]['perc'])
            hdu.header['PERC'] = '{}'.format(self.lcs[i]['perc'])
            hdu.header['NPIX'] = '{}'.format(self.lcs[i]['npix'])
            hdu.header['LEADFLAG'] = '{}'.format(self.lcs[i]['lead_lag_correction'])
            hdus.append(hdu)
        hdul = fits.HDUList(hdus)
        hdul.writeto(
            '{0}{1}'.format(dir, output), overwrite=True)


    def writeTPF(self, output=None, dir=''):
        '''Write data arrays to a moving TPF fits file.

        Parameters
        ----------
        output : str or None
            Output file. If None, will be generated automatically.
        dir : str
            Output directory. Default is current directory.
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')

        if output is None:
            output = 'hlsp_k2movingbodies_k2_tpf_{1}_c{2:02}_v{3}.fits'.format(dir, self.name.replace(' ',''), self.campaign, __version__)
        r = {'ar':self.data_array, 'er':self.error_array, 'diff':self.diff_array, 'ediff':self.diff_error_array}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tpf = utils.build_tpf(r, self.lcs['BEST']['t'], self.name, self.best_aper)
            tpf.hdu.verify('silentfix')
            tpf.to_fits('{0}{1}'.format(dir, output), overwrite=True)

    def plotLightCurves(self):
        '''Plot all the light curves for the object
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')

        fig, ax = plt.subplots(figsize=(15, 4))
        for idx, lc in self.lcs.items():
            if idx == 'BEST':
                continue
            ax.errorbar(lc['t'], lc['lc'], lc['elc'], ls='', c='k', zorder=1*idx, alpha=0.5)
            im = ax.scatter(lc['t'], lc['lc'], c=np.zeros(len(lc['t'])) + lc['perc'], s=1, vmin=70, vmax=100, zorder=2*idx)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Aperture Percentile', fontsize=13)
        ax.set_xlabel('Time (days) [JD]', fontsize=13)
        ax.set_ylabel('Flux [counts]', fontsize=13)
        ax.set_title('{}'.format(self.name), fontsize=17)


    def plotLightCurveQuality(self, save=True, dir=''):
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')

        fig = plt.figure(figsize=(13.33, 7.5))
        ax = plt.subplot2grid((6, 3), (1, 0), colspan=2, rowspan=4)
        plt.scatter(self.percs, self.apermean, c='#9b59b6')
        plt.scatter(self.percs[self.best], self.apermean[self.best], c='#16a085')
        plt.axvline(self.percs[self.best], c='#16a085', ls='--', zorder=-1)
        plt.text(self.percs[self.best]*1.005, self.apermean[self.best]*1.005, 'Best Aperture')

        plt.title('Mean Flux in Aperture')
        plt.xlabel('Aperture Percentile (%)', fontsize=13)
        plt.ylabel('Total Light Curve Flux (Counts) [e$^-$/s]', fontsize=13)
        plt.subplots_adjust(left=0.16)
        ax = plt.subplot2grid((6, 3), (1, 2), rowspan=4)
        plt.imshow(self.thumb, origin='bottom')
        plt.axis('off')
        plt.contour(self.apers[:, :, self.best], colors='white', levels=[0.5])
        if save:
            fig.savefig('{}{}_aperture_selection.png'.format(
                        dir, self.name.replace(' ', '')), dpi=150)

    def plotBestLightCurve(self, save=True, dir=''):
        '''Plot the best light curve
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')
        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        mask = self.lcs['BEST']['background_quality'] & self.lcs['BEST']['lead_quality'] & self.lcs['BEST']['all_pixels']
        ax.errorbar(self.lcs['BEST']['t'][mask], self.lcs['BEST']['lc'][mask], self.lcs['BEST']['elc'][mask],
                    label='Best Data Quality', marker='.', ls='', markersize=2, color='#9b59b6', zorder=2)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        ax.errorbar(self.lcs['BEST']['t'], self.lcs['BEST']['lc'], self.lcs['BEST']['elc'],
                    label='Compromised Data Quality', marker='.', ls='', markersize=2, color='#16a085', zorder=1)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.legend()

        ax.set_xlabel('Time (Julian Date)', fontsize=16)
        ax.set_ylabel('Flux [e$^-$/s]', fontsize=16)
        ax.set_title('{}'.format(self.name), fontsize=20)
        plt.subplots_adjust(left=0.16)
        if save:
            fig.savefig('{}{}_lc.png'.format(dir, self.name.replace(' ', '')), dpi=150)

    def plotTPF(self):
        '''Plot the median TPF image in time
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.imshow(self.thumb)
        ax.set_xticks([])
        ax.set_yticks([])


    def animateTPF(self, dir=''):
        '''Make a movie of the asteroid in time.
        '''
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')
        if self.stack > 1:
            log.warning('Object is faint, stacking {} frames together to build movie.'.format(self.stack))
        ok = np.nansum(self.data_array[:, :, :, 0], axis=(1, 2)) != 0
        ok[np.where(ok == True)[0][0]:np.where(ok == True)[0][-1]] = True
        plotting.two_panel_movie(self.data_array[ok, :, :, 0], self.data_array[ok, :, :, 0] - self.diff_array[ok, :, :, 0],
                        title='', out='{}{}.mp4'.format(dir, self.name.replace(' ','')), scale='linear', vmin=0,
                        vmax=np.max([np.nanmax(self.thumb), 300]), stack=self.stack)

    def info(self):
        if 'ra' in self.__dir__():
            str = ('{}\n{}\nCampaign:\t{}\nMean RA:\t{}\nMean Dec:\t{}'
                   '\nAlternate Names: {}\nObs Start:\t{}\nObs End:\t{}'
                   '\n'
                   '\nNumber of Files: \t{}'
                   '\nNumber of Apertures: \t{}'
                   '\nAperture Radius:\t{} Pixel(s)'.format(self.name, ''.join(['-' for i in range(len(self.name))]),
                                                            self.campaign, self.ra.mean(), self.dec.mean(),
                                                            self.alternate_names, Time(self.time[0], format='jd').isot,
                                                            Time(self.time[-1], format='jd').isot, self.nfiles, self.nlagged,
                                                            self.aperture_radius))
        else:
            str = ('{}\n{}\nCampaign:\t{}'
                   '\nAlternate Names: {}\nObs Start:\t{}\nObs End:\t{}'
                   '\n'
                   '\nNumber of Apertures: \t{}'
                   '\nAperture Radius:\t{} Pixel(s)'.format(self.name, ''.join(['-' for i in range(len(self.name))]),
                                                            self.campaign,
                                                            self.alternate_names, Time(self.time[0], format='jd').isot,
                                                            Time(self.time[-1], format='jd').isot, self.nlagged,
                                                            self.aperture_radius))

        print(str)

    def plotTrack(self):
        '''Diagnostic plot for moving object.'''
        if 'ra' not in self.__dir__():
            raise NotFetchedError('The metadata for this object has not been fetched. Use the `fetch` method to obtain meta data (e.g. asteriks.object(NAME).fetch()).')
        aspect_ratio = (np.nanmax(self.dec) - np.nanmin(self.dec)) / (np.nanmax(self.ra) - np.nanmin(self.ra))
        if aspect_ratio < 1:
            fig, ax = plt.subplots(1, figsize=(10, 10*aspect_ratio))
        if aspect_ratio > 1:
            fig, ax = plt.subplots(1, figsize=(10*aspect_ratio, 10))
        plt.subplots_adjust(wspace=1, hspace=1)
        xlims, ylims = plotting.campaign_base_plot(ax=ax, campaigns=[self.campaign])

        plt.plot(self.ra, self.dec, zorder=-1, lw=1, color='grey', label=self.name)
        plt.xlim(self.ra.min(),
                 self.ra.max())
        plt.ylim(self.dec.min(),
                 self.dec.max())
        plt.gca().set_aspect(1)
        plt.xlabel('RA', fontsize=20)
        plt.ylabel('Declination', fontsize=20)
        plt.title('{} Aperture Masks'.format(self.name), fontsize=20)
        plt.legend()
#        return ax

    def animateTrack(self, output='out.mp4'):
        '''Animation of apertures along the path of the moving object.
        '''
        if 'ra' not in self.__dir__():
            raise NotFetchedError('The metadata for this object has not been fetched. Use the `fetch` method to obtain meta data (e.g. asteriks.object(NAME).fetch()).')
        plotting.plot_aperture_movie(self.jpl_data, self.name, self.campaign, self.lagspacing, self.aperture_radius, output=output)

    def _build_website(self, dir=''):
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')
        page_dir = "{0}{1}/".format(WEBSITE_DIR, self.name.replace(' ', ''))
        if not os.path.isdir(page_dir):
            os.mkdir(page_dir)
        self.writeTPF(dir=page_dir)
        self.writeLightCurve(dir=page_dir)
        with plt.style.context(('ggplot')):
            self.plotBestLightCurve(dir=page_dir)
            self.animateTPF(dir=page_dir)
        sd = (Time(self.lcs[0]['t'][0], format='jd').isot)
        start_date = datetime.strptime(sd, '%Y-%m-%dT%H:%M:%S.%f').strftime('%d %B %Y')
        ed = (Time(self.lcs[0]['t'][-1], format='jd').isot)
        end_date = datetime.strptime(ed, '%Y-%m-%dT%H:%M:%S.%f').strftime('%d %B %Y')


        web.create_asteroid_page_html(self.name, self.campaign, start_date=start_date, end_date=end_date)

    def to_lightcurve(self, aper='BEST'):
        if 'data_array' not in self.__dir__():
            raise NotBuiltError('The data for this object has not been built. Use the `build` method to obtain meta data (e.g. asteriks.object(NAME).build()).')
        mask = self.lcs['BEST']['background_quality'] & self.lcs['BEST']['lead_quality'] & self.lcs['BEST']['all_pixels']
        lc = LightCurve(time=self.lcs[aper]['t'][mask], flux=self.lcs[aper]['lc'][mask], flux_err=self.lcs[aper]['elc'][mask])
        return lc
