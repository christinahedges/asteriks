"Makes light curves of moving objects in K2 data"
import numpy as np
import pandas as pd
import K2ephem
import K2fov
from tqdm import tqdm
import pickle
import os

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from scipy.interpolate import interp1d

from .utils import *
from .plotting import *
from .query import *
from .web import *
from asteriks.version import __version__


class WCSFailure(Exception):
    # There are no WCS files in asteriks to use for this object
    pass


def get_meta(campaign, cadence='long'):
    '''Load the time axis from the package meta data.

    There are stored cadenceno and JD arrays in the data directory.

    Parameters
    ----------
    campaign : int
        Campaign number
    cadence: str
        'long' or 'short'

    Returns
    -------
    time : numpy.ndarray
        Array of time points in JD
    cadeceno : numpy.ndarray
        Cadence numbers for campaign
    '''
    timefile = LC_TIME_FILE
    if cadence in ['short', 'sc']:
        log.warning('Short cadence is not currently supported. Expect odd behaviour')
        timefile = SC_TIME_FILE
    meta = pickle.load(open(timefile, 'rb'))
    if ('{}'.format(campaign) in meta.keys()):
        time = meta['{}'.format(campaign)]['time']
        cadenceno = meta['{}'.format(campaign)]['cadenceno']
    else:
        time = np.zeros(0)
        cadenceno = np.zeros(0)
        for m in meta.items():
            if '{}'.format(campaign) in m[0][0:-1]:
                time = np.append(time, m[1]['time'])
                cadenceno = np.append(cadenceno, m[1]['cadenceno'])
    return time, cadenceno


def get_campaign_number(name):
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
    log.debug('Finding campaign number for {}'.format(name))
    for c in tqdm(np.arange(1, 18), desc='Checking campaigns'):
        with silence():
            df = K2ephem.get_ephemeris_dataframe(name, c, c, step_size=1)
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
        raise ValueError('{} never on Silicon'.format(name))
    campaign = campaigns[0]
    return campaign


def find_lagged_apertures(df, nlagged=0, aperture_radius=3, minvel_cap=0.1*u.pix/u.hour):
    '''Finds the lag time for apertures based on a dataframe of asteroid positions.

    Apertures are built to never overlap. However, if the asteroid goes below some
    minimum velocity, they will be allowed to overlap and a flag will be added.
    This is to ensure that asteroids with a slow turning point still have apertures.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of asteroid times, ras, decs and on silicon flags
    nlagged : int
        Number of lagged apertures to find. If odd, will add one to create an
        even number.
    minvel_cap : float * astropy.units.pix/astropy.units.hour
        Minimum asteroid velocity in pixels/hour.
    Returns
    -------
    lag : numpy.ndarray

    '''
    log.debug('Creating lagged apertures')
    if not hasattr(minvel_cap, 'value'):
        minvel_cap *= u.pix/u.hour

    if nlagged % 2 is 1:
        log.warning('\n\tOdd value of nlagged set ({}). '
                    'Setting to nearest even value. ({})'.format(nlagged, nlagged + 1))
        nlagged += 1

    ok = df.onsil == True
    dr = (np.asarray(df[ok].ra[1:]) - np.asarray(df[ok].ra[0:-1])) * u.deg
    dd = (np.asarray(df[ok].dec[1:]) - np.asarray(df[ok].dec[0:-1])) * u.deg
    t = np.asarray(df[ok].jd[1:]) * u.day
    dt = (np.asarray(df[ok].jd[1:]) - np.asarray(df[ok].jd[0:-1])) * u.day
    dr, dd, t = dr[dt == np.median(dt)], dd[dt == np.median(dt)], t[dt == np.median(dt)]
    dt = np.median(dt)
    velocity = np.asarray(((dr**2 + dd**2)**0.5).to(u.arcsec).value/4)*u.pixel/dt.to(u.hour)
    minvel = np.min(velocity)
    log.debug('\n\tMinimum velocity of {} found'.format(np.round(minvel, 2)))
    velocity = np.interp(np.asarray(df.jd), t.value, velocity.value)
    df['velocity'] = velocity
    df['CONTAMINATEDAPERTUREFLAG'] = velocity < minvel_cap.value
    if minvel < minvel_cap:
        log.warning('\n\tMinimum velocity ({}) less than '
                    'minimum velocity cap ({})! Setting to '
                    'minimum velocity cap.'.format(np.round(minvel, 2), np.round(minvel_cap, 2)))
        minvel = minvel_cap
    lagspacing = np.arange(-nlagged - 2, nlagged + 4, 2)
    lagspacing = lagspacing[np.abs(lagspacing) != 2]
    lagspacing = lagspacing[np.argsort(lagspacing**2)]
    lag = (aperture_radius * u.pixel * lagspacing/minvel).to(u.day).value
    return df, lag, lagspacing


def get_radec(name, campaign=None, nlagged=0, aperture_radius=3, plot=False,
              img_dir='', cadence='long', minvel_cap=0.1*u.pix/u.hour, trim=True):
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
    time, cadenceno = get_meta(campaign, cadence)

    if campaign is None:
        campaign = get_campaign_number(name)

    # Get the ephemeris data from JPL
    alternate_names = find_alternate_names_using_CAF(name)
    with silence():
        for altname in alternate_names:
            try:
                df = K2ephem.get_ephemeris_dataframe(altname, campaign, campaign, step_size=1./(8))
            except:
                continue
            if 'df' in locals():
                break
    if 'df' not in locals():
        log.error('Could not find ephemeris.')

    # Interpolate to the time values for the campaign.
    dftimes = [t[0:23] for t in np.asarray(df.index, dtype=str)]
    df['jd'] = Time(dftimes, format='isot').jd
    log.debug('Creating RA, Dec values for all times in campaign')
    f = interp1d(df.jd, df.ra, fill_value='extrapolate')
    ra = f(time) * u.deg
    f = interp1d(df.jd, df.dec, fill_value='extrapolate')
    dec = f(time) * u.deg
    df = pd.DataFrame(np.asarray([time, cadenceno, ra, dec]).T,
                      columns=['jd', 'cadenceno', 'ra', 'dec'])
    df['campaign'] = campaign

    # Find where asteroid is on silicon.
    log.debug('Finding on silicon values')
    k = K2fov.getKeplerFov(campaign)
    onsil = np.zeros(len(df), dtype=bool)
    for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
        try:
            onsil[i] = k.isOnSilicon(r, d, campaign)
        except:
            continue
    if not np.any(onsil):
        raise ValueError('{} never on Silicon in campaign {}'
                         ''.format(name, campaign))
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
    df, lag, lagspacing = find_lagged_apertures(df, nlagged, aperture_radius, minvel_cap)
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

    if plot:
        log.info('Creating an mp4 of apertures, this will take a few minutes. '
                 'To turn this feature off set plot to False.')
        plot_aperture_movie(dfs, name=name, campaign=campaign, lagspacing=lagspacing,
                            aperture_radius=aperture_radius, dir=img_dir)
        log.debug('Saved mp4 to {}{}_aperture.mp4'.format(img_dir, name.replace(' ', '')))

    # We don't need anything that wasn't in the campaign
    # Remove anything where there is no incampaign flag.
    if trim:
        dfs = [o[dfs[0].incampaign].reset_index(drop=True) for o in dfs]

    # Find the pixel position for every channel.
    c = np.asarray(['{:02}'.format(campaign) in c[0:2] for c in campaign_strb])
    for jdx in range(len(dfs)):
        for b in campaign_strb[c]:
            for ch in np.unique(dfs[jdx].channel).astype(int):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wcs = pickle.load(open('{}c{}_{:02}.p'.format(WCS_DIR, b, int(ch)), 'rb'))
                except FileNotFoundError:
                    continue
                X, Y = wcs.wcs_world2pix([[r, d] for r, d in zip(dfs[jdx].ra, dfs[jdx].dec)], 1).T
                dfs[jdx]['Row_{0:02}_{1:02}'.format(int(b), ch)] = Y.astype(int)
                dfs[jdx]['Column_{0:02}_{1:02}'.format(int(b), ch)] = X.astype(int)
    return dfs


def find_overlapping_cadences(cadences, poscorr1, poscorr2, tol=5, distance_tol=0.02, mask=None):
    '''Finds cadences where observations are almost exactly aligned in the telescope
       despite K2 motion.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mask is None:
            mask = []

        if not hasattr(cadences, '__iter__'):
            cadences = [cadences]

        hits = []
        flags = []
        for i in cadences:
            dist = np.sqrt(((poscorr1 - poscorr1[i]))**2 + ((poscorr2 - poscorr2[i]))**2)
            pos = np.where(dist < distance_tol)[0]
            pos = (np.asarray(list(set(pos) - set([i]) - set(mask))))
            if len(pos) <= tol:
                pos = np.argsort(dist)
                pos = pos[pos != i]
                for m in mask:
                    pos = pos[pos != m]
                pos = pos[0:tol]
                flags.append(0)
            else:
                flags.append(1)
            hits.append(pos)

        if len(hits) == 0:
            return hits[0], flags[0]
    return np.asarray(hits), np.asarray(flags)


def make_arrays(objs, mast, n, diff_tol=5, difference=True):
    '''Make moving TPFs
    '''
    if isinstance(n, int):
        PIXEL_TOL = (n**2+n**2)**0.5
    if hasattr(n, '__iter__'):
        if len(n.shape) >= 1:
            PIXEL_TOL = (n.shape[0]**2+n.shape[1]**2)**0.5
        else:
            PIXEL_TOL = (n[0]**2+n[1]**2)**0.5
    log.debug('PIXEL_TOL set to {}'.format(PIXEL_TOL))
    can_difference = True
    xaper, yaper, aper = build_aperture(n)
    log.debug('Aperture\n {}'.format(aper))

    # Arrays to store final results
    ar = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    er = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    diff_ar = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    diff_er = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    log.debug('Arrays sized {}'.format(ar.shape))

    current_channel = -1

    # Find the positions of all the apertures at all times.
    tablecoords = [None] * len(objs)
    for idx, obj in enumerate(objs):
        tablecoords[idx] = SkyCoord(obj.ra, obj.dec, unit=(u.deg, u.deg))

    # For every file that we need to open...
    for file in tqdm(np.arange(len(mast)), desc='Inflating Files\t'):
        # Find the campaign and channel
        campaign = int(mast.campaign[file])
        channel = int(mast.channel[file])

        # If we've switched channels load a new WCS
        if channel != current_channel:
            current_channel = np.copy(channel)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wcs = pickle.load(open('{}c{:02}_{:02}.p'.format(
                        WCS_DIR, int(campaign), int(channel)), 'rb'))
            except FileNotFoundError:
                log.error('There is no WCS file for Campaign {} Channel {}'
                          ''.format(campaign, channel))
                raise WCSFailure

        # Open the file
        url = mast.url[file]
        try:
            with silence():
                cadence, flux, error, column, row, poscorr1, poscorr2 = open_tpf(url)
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
        for idx, obj in enumerate(objs):
            # Get all the coordinates across all time.
            tablecoord = tablecoords[idx]
            ok = np.zeros(len(tablecoord)).astype(bool)

            # Only use the times where we are close to the aperture.
            for coord in coords:
                ok |= tablecoord.separation(coord) < PIXEL_TOL*4*u.arcsec

            # Pair down the table.
            tab = obj[['cadenceno', 'Column_{:02}_{:02}'.format(
                campaign, channel), 'Row_{:02}_{:02}'.format(int(campaign), int(channel)), 'velocity']][ok]
            # log.debug('{} Near Aperture {}'.format(len(tab), idx))
            # For every time that we are near to the aperture...
            for t in tab.iterrows():
                # Check if it's in aperture.
                inaperture = np.asarray(['{}, {}'.format(int(i), int(j))
                                         for i, j in zip(xaper - n + t[1][1], yaper - n + t[1][2])])
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
                    timetolerance = np.round(((2*n*u.pix)/(v * 0.5*u.hour)).value)
                    clip = np.arange(c[0] - timetolerance, c[0] + timetolerance, 1).astype(int)
                    hits, flag = find_overlapping_cadences(
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
    return ar, er, diff_ar, diff_er


def build_products(name, campaign, dir, movie=False):
    output_dir = '{}{}/'.format(dir, name.replace(' ', ''))
    timetables = pickle.load(open('{}{}_timetables.p'.format(
        output_dir, name.replace(' ', '')), 'rb'))
    r = pickle.load(open('{}{}_tpfs.p'.format(output_dir, name.replace(' ', '')), 'rb'))
    ar, er, diff, ediff = r['ar'], r['er'], r['diff'], r['ediff']
    diff[diff==0] = np.nan
    ediff[diff==0] = np.nan

    thumb = np.nanmedian(ar[:, :, :, 0] - diff[:, :, :, 0], axis=0)

    stack = 1
    if np.nanmax(thumb) < 100:
        log.warning('Faint asteroid.')
#        log.warning('Not using the lead/lag correction.')
        log.warning('Stacking movie')
        thumb = np.nanmedian(stack_array(ar[:, :, :, 0] - diff[:, :, :, 0]), axis=0)
#        lead_lag_correction = False
        stack = 20

    if movie:
        with plt.style.context(('ggplot')):
            ok = np.nansum(ar[:, :, :, 0], axis=(1, 2)) != 0
            ok[np.where(ok == True)[0][0]:np.where(ok == True)[0][-1]] = True
            two_panel_movie(ar[ok, :, :, 0], ar[ok, :, :, 0] - diff[ok, :, :, 0],
                            title='', out='{}{}.mp4'.format(output_dir, name.replace(' ', '')), scale='linear', vmin=0,
                            vmax=np.max([np.nanmax(thumb), 300]), stack=stack)

    percs = np.arange(80, 100, 2)[::-1]
    final_lcs = {}
    apers = np.zeros((ar.shape[1], ar.shape[2], len(percs)))
    ts = np.asarray([timetables[i].jd for i in range(ar.shape[-1])])


    apermean = np.zeros(len(percs))
    apernpoints = np.zeros(len(percs))
    for idx, perc in enumerate(percs):
        lead_lag_correction = True
        # Build the aperture out of the percentiles
        aper = (thumb > np.nanpercentile(thumb, perc))
        fix_aperture(aper)
        if aper.sum() == 0:
            aper = (thumb > np.nanpercentile(thumb, perc))
        apers[:, :, idx] = aper
        npix = np.nansum(aper)

        # Find how many pixels drop out due to nans or traveling over the edge of the tpf
        npix_a = np.asarray([np.sum(np.isfinite(ar[:, :, :, i] - diff[:, :, :, i]) * np.atleast_3d(aper).transpose([2, 0, 1]), axis=(1, 2)) for i in range(ar.shape[-1])], dtype=float)
        all_pixels = npix_a[0] >= np.nanmax(npix_a)*0.8
#        npix_a /= np.nanmax(npix_a)



        # Build all light curves
        lcs = np.asarray([np.nansum((ar[:, :, :, i] - diff[:, :, :, i]) *
                                    np.atleast_3d(aper).T, axis=(1, 2)) for i in range(ar.shape[-1])])
        elcs = np.asarray([np.nansum((er[:, :, :, i]**2 + ediff[:, :, :, i]**2) *
                                    np.atleast_3d(aper).T, axis=(1, 2)) for i in range(ar.shape[-1])])**0.5

        lcs[lcs==0] = np.nan
        elcs[elcs==0] = np.nan

        # Build background
        bkg_perc = np.nanpercentile((ar), 15)
        bkg_aper = (ar) < bkg_perc
        bkgs = np.asarray([np.nansum((ar[:, :, :, i] - diff[:, :, :, i]) *
                                    (bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(ar.shape[-1])])
        ebkgs = np.asarray([np.nansum((er[:, :, :, i]**2 + ediff[:, :, :, i]**2) *
                                    (bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(ar.shape[-1])])**0.5
        bkgs/= np.asarray([np.nansum((bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(ar.shape[-1])])
        ebkgs/= np.asarray([np.nansum((bkg_aper[:, :, :, i] & ~np.atleast_3d(aper).T) , axis=(1, 2)) for i in range(ar.shape[-1])])

        # Interpolate the remaining apertures onto the same time frame as the object
        interp_lcs = np.asarray([np.interp(ts[0, :], t, lc)
                                 for t, lc in zip(ts, lcs)])
        interp_elcs = np.asarray([np.interp(ts[0, :], t, elc)
                                 for t, elc in zip(ts, elcs)])
        interp_npix_a = np.asarray([np.interp(ts[0, :], t, npix_a1)
                         for t, npix_a1 in zip(ts, npix_a)])


#        import pdb; pdb.set_trace()
        # Can you USE the lead/lag apertures?
        # Must have enough pixels in the aperture (80%)
        # Must have at least 50% of lead/lag apertures available
        lead_quality =  np.ones(ar.shape[0], dtype=bool)
        if lead_lag_correction:
            test = (interp_npix_a[1:] > np.atleast_2d(interp_npix_a[1:].max(axis=1)).T*0.8).sum(axis=0) > (len(lcs) - 1)*0.5
            if test.sum() < 0.3 * (npix_a[0] > 1).sum():
                log.warn('Lead lag correction looks poor. Turning off.')
                lead_lag_correction=False
            else:
                lead_quality = test

        # Do the lag apertures pass?
        background_quality = np.ones(lcs.shape[1], dtype=bool)
        if lead_lag_correction:
            median = np.nanmedian(interp_lcs[1:, :], axis=0)#, lead_quality & all_pixels], axis=0)
            # Too much flux in the background is BAD, clip it out
            background_quality[np.abs(median) > 1000] = False

            # What's left? Any outliers?
            _, median1, std1 = sigma_clipped_stats(median, sigma=3, iters=2, mask = ~(lead_quality & all_pixels & background_quality))
            background_quality &= np.abs(median - median1) < 3 * std1

            # Are there noisy time stamps?
            std = np.nanstd(interp_lcs[1:,:], axis=0)
            _, median1, std1 = sigma_clipped_stats(std, sigma=3, iters=2, mask= ~(lead_quality & all_pixels & background_quality))
            background_quality &= np.abs(std - median1) < 3 * std1

        apermean[idx] = np.nansum(lcs[0, lead_quality & all_pixels & background_quality])
        apernpoints[idx] = len(lcs[0, lead_quality & all_pixels & background_quality])
        final_lcs[idx] = {'t': ts[0, :], 'lc': lcs[0, :],
                          'elc': elcs[0, :], 'npix': npix, 'perc': perc,
                          'background_quality' : background_quality, 'all_pixels' : all_pixels,
                          'lead_quality' : lead_quality, 'npix_in_aper':npix_a[0,:], 'aper':aper,
                          'lead_lag_correction':lead_lag_correction}

    grad = np.gradient(apermean/np.nanmin(apermean[apermean!=0]))
    best_mean = np.where(percs == percs[grad
                                        <= np.median(grad[grad!=0])][0])[0][0]
    # Shouldn't waste all the data points...
    npoints = np.where(percs == np.min(percs[apernpoints/np.max(apernpoints) > 0.7]))[0][0]
    best = np.min([best_mean, npoints])
    final_lcs['BEST'] = {'t': final_lcs[best]['t'], 'lc': final_lcs[best]['lc'],
                              'elc': final_lcs[best]['elc'], 'npix': final_lcs[best]['npix'], 'perc': final_lcs[best]['perc'],
                              'background_quality':final_lcs[best]['background_quality'], 'all_pixels':final_lcs[best]['all_pixels'],
                              'lead_quality':final_lcs[best]['lead_quality'], 'npix_in_aper':final_lcs[best]['npix_in_aper'],
                              'aper':final_lcs[best]['aper'],'lead_lag_correction':final_lcs[best]['lead_lag_correction']}


    pickle.dump(final_lcs, open('{}{}_lcs.p'.format(output_dir, name.replace(' ', '')), 'wb'))
    pickle.dump(apers, open('{}{}_apers.p'.format(output_dir, name.replace(' ', '')), 'wb'))

    ra_ar = np.interp(final_lcs[0]['t'], timetables[0].jd, timetables[0].ra)
    dec_ar = np.interp(final_lcs[0]['t'], timetables[0].jd, timetables[0].dec)

    i = 'BEST'
    hdr = fits.Header()
    hdr['ORIGIN'] = 'NASA/Ames'
    hdr['DATE'] = Time.now().isot
    hdr['CREATOR'] = 'asteriks'
    hdr['TELESCOP'] = 'Kepler'
    hdr['INSTRUME'] = 'Kepler Photometer'
    hdr['OBJECT'] = '{}'.format(name)
    hdr['HLSPNAME'] = 'K2MovingBodies'
    hdr['HLSPLEAD'] = 'Kepler/K2 GO Office'
    hdr['EXPSTART'] = Time(final_lcs[i]['t'][0], format='jd').isot
    hdr['EXPEND'] = Time(final_lcs[i]['t'][-1], format='jd').isot
    hdr['LDLGCORR'] = lead_lag_correction
    hdr['VERSION'] = __version__


    # BKG_QUAL : If there is evidence from lead lag that there is a background contaminant, will be False
    # LEAD_QUAL : If the lead/lag test cannot be completed, will be False
    # NPIX_QUAL : If there is not at least 80% of the aperture as non-nans, will be False
    # NPIX_APER : Number of non-NaN pixels in aperture.

    primary_hdu = fits.PrimaryHDU(header=hdr)

    hdus = [primary_hdu]
    cols = []
    cols.append(fits.Column(name='TIME', array=(final_lcs[i]['t']), format='D', unit='JD'))
    cols.append(fits.Column(name='FLUX', array=(final_lcs[i]['lc']), format='E', unit='e-/s'))
    cols.append(fits.Column(name='FLUX_ERR', array=(final_lcs[i]['elc']), format='E', unit='e-/s'))
    cols.append(fits.Column(name='RA_OBJ', array=ra_ar, format='E', unit='deg'))
    cols.append(fits.Column(name='DEC_OBJ', array=dec_ar, format='E', unit='deg'))
    cols.append(fits.Column(name='LEAD_QUAL', array=final_lcs[i]['lead_quality'], format='L'))
    cols.append(fits.Column(name='NPIX_QUAL', array=final_lcs[i]['all_pixels'], format='L'))
    cols.append(fits.Column(name='BKG_QUAL', array=final_lcs[i]['background_quality'], format='L'))
    cols.append(fits.Column(name='NPIX_APER', array=final_lcs[i]['npix_in_aper'], format='I'))

    cols = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header['EXTNAME'] = 'BESTAPER'
    hdu.header['PERC'] = '{}'.format(final_lcs[i]['perc'])
    hdu.header['NPIX'] = '{}'.format(final_lcs[i]['npix'])
    hdu.header['LEADFLAG'] = '{}'.format(final_lcs[i]['lead_lag_correction'])
    hdus.append(hdu)

    for i in range(list(final_lcs.keys())[-2]):
        cols = []
        cols.append(fits.Column(name='TIME', array=(final_lcs[i]['t']), format='D', unit='JD'))
        cols.append(fits.Column(name='FLUX', array=(final_lcs[i]['lc']), format='E', unit='e-/s'))
        cols.append(fits.Column(name='FLUX_ERR', array=(
            final_lcs[i]['elc']), format='E', unit='e-/s'))
        cols.append(fits.Column(name='RA_OBJ', array=ra_ar, format='E', unit='deg'))
        cols.append(fits.Column(name='DEC_OBJ', array=dec_ar, format='E', unit='deg'))
        cols.append(fits.Column(name='LEAD_QUAL', array=final_lcs[i]['lead_quality'], format='L'))
        cols.append(fits.Column(name='NPIX_QUAL', array=final_lcs[i]['all_pixels'], format='L'))
        cols.append(fits.Column(name='BKG_QUAL', array=final_lcs[i]['background_quality'], format='L'))
        cols.append(fits.Column(name='NPIX_APER', array=final_lcs[i]['npix_in_aper'], format='I'))

        cols = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header['EXTNAME'] = 'PERC{}'.format(final_lcs[i]['perc'])
        hdu.header['PERC'] = '{}'.format(final_lcs[i]['perc'])
        hdu.header['NPIX'] = '{}'.format(final_lcs[i]['npix'])
        hdu.header['LEADFLAG'] = '{}'.format(final_lcs[i]['lead_lag_correction'])
        hdus.append(hdu)
    hdul = fits.HDUList(hdus)
    hdul.writeto(
        '{0}{1}/hlsp_k2movingbodies_k2_lightcurve_{1}_c{2:02}_v{3}.fits'.format(dir, name.replace(' ',''), campaign, __version__), overwrite=True)
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(13.33, 7.5))
        ax = plt.subplot2grid((6, 3), (1, 0), colspan=2, rowspan=4)
        plt.scatter(percs, apermean, c='#9b59b6')
        plt.scatter(percs[best], apermean[best], c='#16a085')
        plt.axvline(percs[best], c='#16a085', ls='--', zorder=-1)
        plt.text(percs[best]*1.005, apermean[best]*1.005, 'Best Aperture')

        plt.title('Mean Flux in Aperture')
        plt.xlabel('Aperture Percentile (%)', fontsize=13)
        plt.ylabel('Total Light Curve Flux (Counts) [e$^-$/s]', fontsize=13)
        plt.subplots_adjust(left=0.16)
        ax = plt.subplot2grid((6, 3), (1, 2), rowspan=4)
        plt.imshow(thumb, origin='bottom')
        plt.axis('off')
        plt.contour(apers[:, :, best], colors='white', levels=[0, 1])

        fig.savefig('{}{}_aperture_selection.png'.format(
            output_dir, name.replace(' ', '')), dpi=150)

        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        mask = final_lcs[best]['background_quality'] & final_lcs[best]['lead_quality'] & final_lcs[best]['all_pixels']
        ax.errorbar(final_lcs[best]['t'][mask], final_lcs[best]['lc'][mask], final_lcs[best]['elc'][mask],
                    label='Best Data Quality', marker='.', ls='', markersize=2, color='#9b59b6', zorder=2)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        ax.errorbar(final_lcs[best]['t'], final_lcs[best]['lc'], final_lcs[best]['elc'],
                    label='Compromised Data Quality', marker='.', ls='', markersize=2, color='#16a085', zorder=1)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.legend()

        ax.set_xlabel('Time (Julian Date)', fontsize=16)
        ax.set_ylabel('Flux [e$^-$/s]', fontsize=16)
        ax.set_title('{}'.format(name), fontsize=20)
        plt.subplots_adjust(left=0.16)
        fig.savefig('{}{}_lc.png'.format(output_dir, name.replace(' ', '')), dpi=150)
        create_asteroid_page_html(name, dir)


def run(name, campaign=None, aperture_radius=8, dir='/Users/ch/K2/projects/hlsp-asteriks/output/'):
    log.info('Running {}, Campaign {}'.format(name, campaign))
    output_dir = '{}{}/'.format(dir, name.replace(' ', ''))
    log.debug('Output to {}'.format(output_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if campaign is None:
        mast, campaign = find_mast_files_using_CAF(name, campaign)
    timetables = get_radec(name, campaign, aperture_radius, plot=False, img_dir=output_dir)
    pickle.dump(timetables, open('{}{}_timetables.p'.format(
        output_dir, name.replace(' ', '')), 'wb'))
    mast = get_mast(name, campaign, timetables=timetables)
    # Make sure that MAST hasn't returned channels that aren't anywhere near
    # the asteroid!
    in_asteroid_track = timetables[0].channel.unique().astype(int)
    in_mast_list = mast.channel.unique().astype(int)
    ok_channels = in_mast_list[np.in1d(in_mast_list, in_asteroid_track)]
    ok = np.zeros(len(mast), dtype=bool)
    for o in ok_channels:
        ok |= np.asarray(mast.channel, dtype=int) == o
    mast = mast[ok]
    mast.to_csv('{}{}_mast.csv'.format(output_dir, name.replace(' ', '')), index=False)
    ar, er, diff, ediff = make_arrays(timetables, mast, aperture_radius)
    import pdb;pdb.set_trace()
    if np.all(~np.isfinite(diff)):
        diff[:, :, :, :] = 0
        ediff[:, :, :, :] = 0
    t = timetables[0].jd
    cadenceno = timetables[0].cadenceno
    results = {'ar': ar, 'er': er, 'diff': diff, 'ediff': ediff, 't': t, 'cadenceno': cadenceno}
    pickle.dump(results, open('{}{}_tpfs.p'.format(output_dir, name.replace(' ', '')), 'wb'))
    build_products(name, campaign, dir, movie=True)
