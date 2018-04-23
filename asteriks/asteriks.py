"Makes light curves of moving objects in K2 data"
PACKAGEDIR = '/Users/ch/K2/projects/asteriks/python/' #For now

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

from scipy.interpolate import interp1d

from lightkurve import KeplerTargetPixelFile

from .utils import *
from .plotting import *

from . import PACKAGEDIR
campaign_stra = np.asarray(['1', '2', '3','4', '5', '6', '7' ,'8', '91', '92',
                            '101', '102', '111', '112', '12', '13', '14', '15',
                            '16', '17', '18'])
campaign_strb = np.asarray(['01', '02', '03','04', '05', '06', '07' ,'08', '91',
                            '92', '101', '102', '111', '112', '12', '13', '14',
                            '15', '16', '17', '18'])

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
LC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'lc_meta.p')
SC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'sc_meta.p')

PIXEL_TOL = 100

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
            df = K2ephem.get_ephemeris_dataframe(name, c, c, step_size=1.)
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


def find_lagged_apertures(df, nlagged=0, minvel_cap=0.1*u.pix/u.hour):
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
    return df, lag


def get_radec(name, campaign=None, nlagged=0, aperture_radius=3, plot=False,
              img_dir='', cadence='long', minvel_cap=0.1*u.pix/u.hour):
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
    df, lag = find_lagged_apertures(df, nlagged, minvel_cap)
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
        make_aperture_movie(dfs, name=name, campaign=campaign, lagspacing=lagspacing,
                           aperture_radius=aperture_radius, dir=img_dir)
        log.debug('Saved mp4 to {}{}_aperture.mp4'.format(img_dir, name.replace(' ','')))

    # We don't need anything that wasn't in the campaign
    # Remove anything where there is no incampaign flag.
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


def get_mast(objs, search_radius = (PIXEL_TOL*4.)/60.):
    '''Queries MAST for all files near a moving object.

    Parameters
    ----------
    objs : list of pandas.DataFrame's
        Result from `get_radec` function
    search_radius : float
        MAST API search radius in arcmin. Default is 4. Increase this to be more
        robust if TPFs in the channel are larger than PIXEL_TOL pixels.

    Returns
    -------
    mast : pandas.DataFrame
        A dataframe with all the target pixel files near the object at any time.
    '''
    obj = objs[0]
    ra, dec, channel = np.asarray(obj.ra), np.asarray(obj.dec), np.asarray(obj.channel)
    ra_chunk = list(chunk(ra, int(np.ceil(len(ra)/100))))
    dec_chunk = list(chunk(dec, int(np.ceil(len(ra)/100))))
    channel_chunk = list(chunk(channel, int(np.ceil(len(ra)/100))))

    MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
    extra = 'outputformat=CSV&action=Search'
    columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel'
    mast = pd.DataFrame(columns=['RA','Dec','EPIC', 'channel'])
    campaign = np.unique(obj.campaign)[0]
    for idx in tqdm(range(len(ra_chunk)), desc='Querying MAST \t'):
        r, d, ch = ra_chunk[idx], dec_chunk[idx], channel_chunk[idx]
        query = 'RA={}&DEC={}&radius={}&sci_campaign={}&sci_channel={}&max_records=100&'.format(
                        ",".join(list(np.asarray(r, dtype='str'))),
                        ",".join(list(np.asarray(d, dtype='str'))),
                        search_radius,
                        campaign,
                        ",".join(list(np.asarray(np.asarray(ch, dtype=int),
                        dtype='str'))))
        chunk_df = pd.read_csv(MAST_API + query + extra + columns,
                               error_bad_lines=False,
                               names=['RA','Dec','EPIC','Investigation ID', 'channel'])
        chunk_df = chunk_df.dropna(subset=['EPIC']).reset_index(drop=True)
        chunk_df = chunk_df.loc[chunk_df.RA != 'RA (J2000)']
        chunk_df = chunk_df.loc[chunk_df.RA != 'ra']
        mast = mast.append(chunk_df.drop_duplicates(['EPIC']).reset_index(drop=True))
    mast = mast.drop_duplicates(['EPIC']).reset_index(drop='True')
    ids = np.asarray(mast.EPIC, dtype=str)
    c = np.asarray(['{:02}'.format(campaign) in c[0:2] for c in campaign_strb])
    m = pd.DataFrame(columns = mast.columns)
    for a, b in zip(campaign_stra[c], campaign_strb[c]):
        m1 = mast.copy()
        urls = ['http://archive.stsci.edu/missions/k2/target_pixel_files/c{}/'.format(a)+i[0:4] +
                '00000/'+i[4:6]+'000/ktwo' + i +
                '-c{}_lpd-targ.fits.gz'.format(b) for i in ids]
        m1['url'] = urls
        m1['campaign'] = b
        m = m.append(m1)
    coord = SkyCoord(m.RA, m.Dec, unit=(u.hourangle, u.deg))
    m['RA'] = coord.ra.deg
    m['Dec'] = coord.dec.deg

    m = m.reset_index(drop=True)
    return m


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

def build_aperture(n = 5, shape='square', xoffset=0, yoffset=0):
    '''Create the X and Y locations for a circular aperture.

    Parameters
    ----------
    n : float
        Aperture radius
    '''
    log.debug('Building apertures')
    x, y = np.meshgrid(np.arange(np.ceil(n).astype(int) * 2 + 1, dtype=float), np.arange(np.ceil(n).astype(int) * 2 + 1, dtype=float))
    if isinstance(shape, str):
        if shape == 'circular':
            aper = ((x - n + 1 + xoffset)**2 + (y - n + 1 + yoffset)**2) < (n**2)
        if shape == 'square':
            aper = np.ones(x.shape, dtype=bool)
    else:
        aper = shape
    x[~aper] = np.nan
    y[~aper] = np.nan
    xaper, yaper = np.asarray(x), np.asarray(y)
    xaper, yaper = np.asarray(xaper[np.isfinite(xaper)], dtype=int), np.asarray(yaper[np.isfinite(yaper)], dtype=int)
    return xaper, yaper

def make_arrays(objs, mast, xaper, yaper, n, diff_tol=5):
    '''Make moving TPFs
    '''
    can_difference = True

    ar = np.zeros((len(objs[0]), xaper.max() + 1, yaper.max() + 1, len(objs))) * np.nan
    er = np.zeros((len(objs[0]), xaper.max() + 1, yaper.max() + 1, len(objs))) * np.nan
    diff_ar = np.zeros((len(objs[0]), xaper.max() + 1, yaper.max() + 1, len(objs))) * np.nan
    diff_er = np.zeros((len(objs[0]), xaper.max() + 1, yaper.max() + 1, len(objs))) * np.nan
    difflags = np.zeros(len(objs[0]))
    log.debug('Arrays sized {}'.format(ar.shape))

    mastcoord = SkyCoord(mast.RA, mast.Dec, unit=(u.deg, u.deg))
    for file in tqdm(np.arange(len(mast)), desc='Inflating Files\t'):
        campaign = mast.campaign[file]
        channel = mast.channel[file]
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

        for idx, obj in enumerate(objs):
            tablecoord = SkyCoord(obj.ra, obj.dec, unit=(u.deg, u.deg))
            # THIS IS DANGEROUS
            # IT SHOULD BE MORE INTELLIGENT
            ok = mastcoord[file].separation(tablecoord) < PIXEL_TOL * 4*u.arcsec
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
                if can_difference:
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

def run(name, campaign=None, search_radius=4, aperture_radius=3,
        nlagged=6, xoffset=0, yoffset=0):

        objs = get_radec(name, campaign, nlagged=nlagged, aperture_radius=aperture_radius)
        # Trim to the times that the asteroid is in the campaign
        mast = get_mast(objs, search_radius=search_radius)

        xaper, yaper = build_aperture(aperture_radius, xoffset=xoffset, yoffset=yoffset)

        make_arrays(objs, mast, xaper, yaper)
