"Makes light curves of moving objects in K2 data"
import logging
import numpy as np
import pandas as pd
import K2ephem
import K2fov
from tqdm import tqdm
import pickle
import os

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats

from scipy.interpolate import interp1d


from .utils import *
from .plotting import *
from .query import *

from time import time
from . import PACKAGEDIR

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
    timefile=LC_TIME_FILE
    if cadence in ['short', 'sc']:
        log.warning('Short cadence is not currently supported. Expect odd behaviour')
        timefile=SC_TIME_FILE
    meta = pickle.load(open(timefile,'rb'))
    if ('{}'.format(campaign) in meta.keys()):
        time = meta['{}'.format(campaign)]['time']
        cadenceno = meta['{}'.format(campaign)]['cadenceno']
    else:
        time = np.zeros(0)
        cadenceno = np.zeros(0)
        for m in meta.items():
            if '{}'.format(campaign) in m[0][0:-1]:
                time = np.append(time, m[1]['time'])
                cadenceno = np.append(cadenceno, m[1]['time'])
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
    for c in tqdm(np.arange(1,18), desc='Checking campaigns'):
        with silence():
            df = K2ephem.get_ephemeris_dataframe(name, c, c, step_size=1)
        k = K2fov.getKeplerFov(c)
        onsil = np.zeros(len(df), dtype=bool)
        for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
            try:
                onsil[i] = k.isOnSilicon(r, d, 1)
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
        nlagged+=1

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
    df['velocity'] =  velocity
    df['CONTAMINATEDAPERTUREFLAG'] =  velocity < minvel_cap.value
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
    with silence():
        df = K2ephem.get_ephemeris_dataframe(name, campaign, campaign, step_size=1./(8))

    # Interpolate to the time values for the campaign.
    dftimes = [t[0:23] for t in np.asarray(df.index, dtype=str)]
    df['jd'] = Time(dftimes,format='isot').jd
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
            onsil[i] = k.isOnSilicon(r, d, 1)
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
    x = np.zeros((len(df),3))
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
        log.debug('Saved mp4 to {}{}_aperture.mp4'.format(img_dir, name.replace(' ','')))

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
                dfs[jdx]['Row_{}_{}'.format(b, ch)] = Y.astype(int)
                dfs[jdx]['Column_{}_{}'.format(b, ch)] = X.astype(int)
    return dfs


def find_overlapping_cadences(cadences, poscorr1, poscorr2, tol=5, distance_tol=0.02, mask=None):
    '''Finds cadences where observations are almost exactly aligned.

    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mask is None:
            mask=[]

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
    ar = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    er = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    diff_ar = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    diff_er = np.zeros((len(objs[0]), aper.shape[0], aper.shape[1], len(objs))) * np.nan
    difflags = np.zeros(len(objs[0]))
    log.debug('Arrays sized {}'.format(ar.shape))

    mastcoord = SkyCoord(mast.RA, mast.Dec, unit=(u.deg, u.deg))
    current_channel = -1

    tablecoords = [None] * len(objs)
    for idx, obj in enumerate(objs):
        tablecoords[idx] = SkyCoord(obj.ra, obj.dec, unit=(u.deg, u.deg))

    for file in tqdm(np.arange(len(mast)), desc='Inflating Files\t'):
        campaign = mast.campaign[file]
        channel = mast.channel[file]
        if channel != current_channel:
            current_channel = np.copy(channel)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wcs = pickle.load(open('{}c{}_{:02}.p'.format(WCS_DIR, campaign, int(channel)), 'rb'))
            except FileNotFoundError:
                log.error('There is no WCS file for Campaign {} Channel {}'
                          ''.format(campaign, channel))
                raise WCSFailure
        url = mast.url[file]
        try:
            with silence():
                cadence, flux, error, column, row, poscorr1, poscorr2 = open_tpf(url)
            if can_difference:
                if np.all(~np.isfinite(poscorr1)) & np.all(~np.isfinite(poscorr1)):
                    can_difference=False
                    log.warning('\nThere is no POS_CORR information. Can not use difference imaging.\n')
            else:
                if np.any(np.isfinite(poscorr1)) & np.any(np.isfinite(poscorr1)):
                    can_difference=True
                    log.warning('\nThere is POS_CORR information. Difference imaging turned on.\n')
        except OSError:
            continue
        pixel_coordinates = np.asarray(['{}, {}'.format(i, j) for i, j in zip(column.ravel(), row.ravel())])

        r, d = np.asarray(wcs.wcs_pix2world(column.ravel(), row.ravel(), 1))
        coords = SkyCoord(r, d, unit=(u.deg, u.deg))
        r, d = coords.ra, coords.dec

        for idx, obj in enumerate(objs):
            tablecoord = tablecoords[idx]
            ok = np.zeros(len(tablecoord)).astype(bool)
            for coord in coords:
                ok |= tablecoord.separation(coord) < PIXEL_TOL*4*u.arcsec
            tab = obj[['cadenceno','Column_{}_{}'.format(campaign, channel),'Row_{}_{}'.format(campaign, channel), 'velocity']][ok]
            for t in tab.iterrows():

                inaperture = np.asarray(['{}, {}'.format(int(i), int(j)) for i, j in zip(xaper - n + t[1][1], yaper - n  + t[1][2])])
                mask_1 = np.asarray([i in pixel_coordinates for i in inaperture])
                if not np.any(mask_1):
                    continue
                mask_2 = np.asarray([i in inaperture for i in pixel_coordinates])
                c = np.where(cadence == int(t[1][0]))[0]
                if len(c) == 0:
                    continue
                if can_difference & difference:
                    v = t[1][3]*u.pix/u.hour
                    timetolerance = np.round(((2*n*u.pix)/(v * 0.5*u.hour)).value)
                    clip = np.arange(c[0] - timetolerance, c[0] + timetolerance, 1).astype(int)
                    hits, flag = find_overlapping_cadences(c, poscorr1, poscorr2, mask=clip, tol=diff_tol)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if flag[0] == 1:
                            diff = np.nanmedian(flux[hits[0],:,:], axis=0)
                            ediff = (1./(len(hits[0]))) * np.nansum(error[hits[0],:,:]**2, axis=0)**0.5
                            diff_ar[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (diff.ravel()[mask_2])
                            diff_er[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (ediff.ravel()[mask_2])
                with warnings.catch_warnings():
                    ar[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (flux[c[0]].ravel()[mask_2])
                    er[int(t[0]), xaper[mask_1], yaper[mask_1], idx] = (error[c[0]].ravel()[mask_2])
    return ar, er, diff_ar, diff_er


def run(name, campaign, aperture_radius=8, dir='/Users/ch/K2/projects/hlsp-asteriks/output/'):
    log.info('Running {}, Campaign {}'.format(name, campaign))
    output_dir = '{}{}/'.format(dir, name.replace(' ',''))
    log.debug('Output to {}'.format(output_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    timetables = get_radec(name, campaign, aperture_radius, plot=True, img_dir=output_dir)
    pickle.dump(timetables, open('{}{}_timetables.p'.format(output_dir, name.replace(' ','')), 'wb'))
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
    mast.to_csv('{}{}_mast.csv'.format(output_dir, name.replace(' ','')), index=False)
    ar, er, diff, ediff = make_arrays(timetables, mast, aperture_radius)
    if np.all(~np.isfinite(diff)):
        diff[:,:,:,:] = 0
        ediff[:,:,:,:] = 0

    fig = plt.figure()
    thumb = np.nanmedian(ar[:, :, :, 0] - diff[:,:,:,0], axis=0)
    plt.imshow(thumb, origin='bottom', cmap='inferno')
    plt.axis('off')
    fig.savefig('{}{}_thumb.png'.format(output_dir, name.replace(' ','')), bbox_inches='tight', dpi=150)

    t = timetables[0].jd
    cadenceno = timetables[0].cadenceno
    results = {'ar':ar, 'er':er, 'diff':diff, 'ediff':diff, 't':t, 'cadenceno':cadenceno}
    pickle.dump(results, open('{}{}_tpfs.p'.format(output_dir, name.replace(' ','')), 'wb'))

    ok = np.nansum(ar[:,:,:,0], axis=(1,2)) != 0
    ok[np.where(ok == True)[0][0]:np.where(ok == True)[0][-1]] = True
    movie(ar[ok,:,:,0] - diff[ok,:,:,0], out='{}{}.mp4'.format(output_dir, name.replace(' ','')),
                   title='Motion Differenced Flux', vmax = np.nanmax(thumb), vmin=0, cmap='inferno', facecolor='grey')
    movie(ar[ok,:,:,0], out='{}{}_RAW.mp4'.format(output_dir, name.replace(' ','')),
                   title='TPF Flux', vmax = np.nanmax(thumb), vmin=0, cmap='inferno', facecolor='grey')
    percs = np.arange(80, 100, 2)[::-1]
    npanels=len(percs)
    final_lcs = {}
    apers = np.zeros((ar.shape[1], ar.shape[2], len(percs)))
    #Build a plot and lcs
    with plt.style.context(('ggplot')):
        ts = np.asarray([timetables[i].jd for i in range(ar.shape[-1])])
        xlims = [1e13,0]
        ylims = [1e13,0]
        for idx, perc in enumerate(percs):
            aper = (thumb > np.nanpercentile(thumb, perc))
            fix_aperture(aper)
            apers[:,:,idx] = aper
            npix = np.nansum(aper)
            naper = np.asarray([np.nansum(np.isfinite(ar[:,:,:,i] - diff[:,:,:,i])*np.atleast_3d(aper).T, axis=(1,2)) == npix for i in range(ar.shape[-1])])

            lcs = np.asarray([np.nansum((ar[:,:,:,i] - diff[:,:,:,i]) * np.atleast_3d(aper).T, axis=(1,2)) for i in range(ar.shape[-1])])
            elcs = np.asarray([(1./(npix))*np.nansum((er[:,:,:,i]**2 + ediff[:,:,:,i]**2)**0.5, axis=(1,2)) for i in range(ar.shape[-1])])

            lcs = np.asarray([np.interp(ts[0], t, lc) for t, lc in zip(ts, lcs)])
            lcs[~naper] = np.nan
            elcs[~naper] = np.nan

            t = ts[0]
            lc = lcs[0]
            elc = elcs[0]

            lagged = np.nanstd(lcs[1:], axis=0)
            _, median, std = sigma_clipped_stats(lagged[np.isfinite(lagged)], sigma=5, iters=3)
            lagged -= median
            mask = np.abs(lagged) < 3*std

            lagged = np.nanmedian(lcs[1:], axis=0)
            _, median, std = sigma_clipped_stats(lagged[np.isfinite(lagged)], sigma=5, iters=3)
            lagged -= median
            mask &= np.abs(lagged) < 3*std

            t, lc, elc = t[mask][np.isfinite(lc[mask])], lc[mask][np.isfinite(lc[mask])], elc[mask][np.isfinite(lc[mask])]

            final_lcs[idx] = {'t':t, 'lc':lc, 'elc':elc, 'npix':npix, 'perc':perc}

        pickle.dump(final_lcs, open('{}{}_lcs.p'.format(output_dir, name.replace(' ','')), 'wb'))
        pickle.dump(apers, open('{}{}_apers.p'.format(output_dir, name.replace(' ','')), 'wb'))

        fig = plt.figure(figsize=(12, 3))
        ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        ax.errorbar(final_lcs[2]['t'], final_lcs[2]['lc'], final_lcs[2]['elc'], label='N Pixels {} Perc {}'.format(final_lcs[2]['npix'], final_lcs[2]['perc']), marker='.', ls='', markersize=2)
        ax.set_xlabel('Time (Julian Date)')
        ax.set_ylabel('Counts e$^-$/s')
        ax.set_title('{}'.format(name))
        ax.legend()
        ax = plt.subplot2grid((1, 4), (0, 3))
        ax.imshow(thumb, origin='bottom')
        ax.axis('off')
        ax.contour(apers[:,:,2], colors='white', levels=[0,1])
        fig.savefig('{}{}_lc.png'.format(output_dir, name.replace(' ','')), bbox_inches='tight', dpi=150)
