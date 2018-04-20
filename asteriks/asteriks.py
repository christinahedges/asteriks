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
from astropy.utils.data import download_file, clear_download_cache
import astropy.units as u

from scipy.interpolate import interp1d

import fitsio

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

# asteriks is ONLY designed to work with the following quality flag.
# change it at your own risk.
quality_bitmask=(32768|65536)

def get_meta(campaign, cadence='long'):
    '''Load the time axis from the package meta data
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

def get_radec(name, campaign=None, nlagged=None, aperture_radius=3, plot=False,
              img_dir='', cadence='long', minvel_cap=0.5*u.pix/u.hour):
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
    if not hasattr(minvel_cap, 'value'):
        minvel_cap *= u.pix/u.hour

    if campaign is None:
        campaigns = []
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
                log.info('\n\tMoving object found in campaign {}'.format(c))
                break
        if len(campaigns) == 0:
            raise ValueError('{} never on Silicon'.format(name))
        campaign = campaigns[0]

    with silence():
        df = K2ephem.get_ephemeris_dataframe(name, campaign, campaign, step_size=1./(8))

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

    log.debug('Finding channels')
    x = np.asarray([k.getChannelColRow(r, d) for r, d in zip(df.ra, df.dec)])
    df['channel'] = x[:,0]

    log.debug('Creating lagged apertures')
    # Find the lagged apertures
    if nlagged is None:
        nlagged = 0
    if nlagged % 2 is 1:
        log.warning('\n\tOdd value of nlagged set ({}). '
                    'Setting to nearest even value. ({})'.format(nlagged, nlagged + 1))
        nlagged+=1
    else:
        #Find lagged apertures based on the maximum velocity of the asteroid
        ok = df.onsil == True
        dr = (np.asarray(df[ok].ra[1:]) - np.asarray(df[ok].ra[0:-1])) * u.deg
        dd = (np.asarray(df[ok].dec[1:]) - np.asarray(df[ok].dec[0:-1])) * u.deg
        t = np.asarray(df[ok].jd[1:]) * u.day
        dt = (np.asarray(df[ok].jd[1:]) - np.asarray(df[ok].jd[0:-1])) * u.day
        dr, dd, t = dr[dt == np.median(dt)], dd[dt == np.median(dt)], t[dt == np.median(dt)]
        dt = np.median(dt)
        velocity = np.asarray(((dr**2 + dd**2)**0.5).to(u.arcsec).value/4)*u.pixel/dt.to(u.hour)
        minvel = np.min(velocity)
        log.info('\n\tMinimum velocity of {} found'.format(np.round(minvel, 2)))
        df['CONTAMINATEDAPERTUREFLAG'] =  np.interp(np.asarray(df.jd), t.value, velocity.value) < minvel_cap.value
        if minvel < minvel_cap:
            log.warning('\n\tMinimum velocity ({}) less than '
                        'minimum velocity cap ({})! Setting to '
                        'minimum velocity cap.'.format(np.round(minvel, 2), np.round(minvel_cap, 2)))
            minvel = minvel_cap
        lagspacing = np.arange(-nlagged - 2, nlagged + 4, 2)
        lagspacing = lagspacing[np.abs(lagspacing) != 2]
        lagspacing = lagspacing[np.argsort(lagspacing**2)]
        lag = (aperture_radius * u.pixel * lagspacing/minvel).to(u.day).value
        log.info('\n\tLag found \n {} (days)'.format(np.atleast_2d(lag).T))
    dfs = []
    for l in lag:
        df1 = df.copy()
        f = interp1d(df1.jd, df1.ra, fill_value='extrapolate')
        ra = f(df1.jd + l) * u.deg
        f = interp1d(df1.jd, df1.dec, fill_value='extrapolate')
        dec = f(df1.jd + l) * u.deg
        df1['ra'] = ra
        df1['dec'] = dec
        dfs.append(df1)
    if plot:
        log.info('Creating an mp4 of apertures, this will take a few minutes. '
                 'To turn this feature off set plot to False.')
        make_aperture_movie(dfs, name=name, campaign=campaign, lagspacing=lagspacing,
                           aperture_radius=aperture_radius, dir=img_dir)
        log.debug('Saved mp4 to {}{}_aperture.mp4'.format(img_dir, name.replace(' ','')))
    return dfs

def get_mast(obj, search_radius=4.):
    '''Queries MAST for all files near a moving object.

    Parameters
    ----------
    obj : list of pandas.DataFrame's
        Result from `get_radec` function
    search_radius : float
        MAST API search radius in arcmin. Default is 4. Increase this to be more
        robust if TPFs in the channel are larger than 50 pixels.

    Returns
    -------
    mast : pandas.DataFrame
        A dataframe with all the target pixel files near the object at any time.
    timetable : list of pandas.DataFrames

        IF there is a lookup table of times, we can move timetable into get_radec
        and call it get timetable! Much less confusing.

        BUT this is currently robust against SC data?

    '''
    ra, dec, channel = np.asarray(obj.ra), np.asarray(obj.dec), np.asarray(obj.channel)
    ra_chunk = list(chunk(ra, int(np.ceil(len(ra)/200))))
    dec_chunk = list(chunk(dec, int(np.ceil(len(ra)/200))))
    channel_chunk = list(chunk(channel, int(np.ceil(len(ra)/200))))

    MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
    extra = 'outputformat=CSV&action=Search'
    columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel'
    mast = pd.DataFrame(columns=['RA','Dec','EPIC', 'channel'])
    campaign = np.unique(obj.campaign)[0]
    for r, d, ch in zip(ra_chunk, dec_chunk, channel_chunk):
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
    c = np.asarray(['{:02}'.format(campaign) in c for c in campaign_strb])
    m = pd.DataFrame(columns = mast.columns)
    times = []
    cadences = []
    for a, b in zip(campaign_stra[c], campaign_strb[c]):
        m1 = mast.copy()
        urls = ['http://archive.stsci.edu/missions/k2/target_pixel_files/c{}/'.format(a)+i[0:4] +
                '00000/'+i[4:6]+'000/ktwo' + i +
                '-c{}_lpd-targ.fits.gz'.format(b) for i in ids]
        m1['url'] = urls
        m1['campaign'] = b
        with silence():
            tpf_filename = download_file(urls[0], cache=True)
        tpf = KeplerTargetPixelFile(tpf_filename, quality_bitmask=quality_bitmask)
        times.append(tpf.timeobj.jd)
        cadences.append(tpf.hdu[1].data['CADENCENO'][tpf.quality_mask])
        m1['starttime'] = tpf.hdu[1].data['CADENCENO'][tpf.quality_mask][0]
        m1['endtime'] = tpf.hdu[1].data['CADENCENO'][tpf.quality_mask][-1]
        m = m.append(m1)
    coord = SkyCoord(m.RA, m.Dec, unit=(u.hourangle, u.deg))
    m['RA'] = coord.ra.deg
    m['Dec'] = coord.dec.deg
    times = np.sort(np.unique(np.asarray([item
                                           for sublist in times
                                             for item in sublist])))
    cadences = np.sort(np.unique(np.asarray([item
                                                for sublist in cadences
                                                    for item in sublist], dtype=int)))
    RA = np.interp(times, obj.jd, obj.ra)
    Dec = np.interp(times, obj.jd, obj.dec)
    timetable = pd.DataFrame(np.asarray([RA, Dec, cadences, times]).T,
                             columns=['RA', 'Dec', 'cadenceno', 'jd'])
    timetable = timetable[(timetable.jd > obj.jd.min()) & (timetable.jd < obj.jd.max())]
    for b in campaign_strb[c]:
        for ch in np.unique(m.channel):
            wcs = pickle.load(open('{}c{}_{:02}.p'.format(WCS_DIR, b, int(ch)), 'rb'))
            X, Y = wcs.wcs_world2pix([[r, d] for r, d in zip(timetable.RA, timetable.Dec)], 1).T
            timetable['Row_{}_{}'.format(b, ch)] = Y.astype(int)
            timetable['Column_{}_{}'.format(b, ch)] = X.astype(int)
    m = m.reset_index(drop=True)
    timetable = timetable.reset_index(drop=True)
    timetable['order'] = (timetable.cadenceno - timetable.cadenceno[0]).astype(int)
    return m, timetable

def open_tpf(tpf_filename):
    '''Opens a TPF

    Parameters
    ----------
    tpf_filename : str
        Name of the file to open. Can be a URL
    quality_bitmask : bitmask
        bitmask to apply to data
    '''
    if tpf_filename.startswith("http"):
        try:
            with silence():
                tpf_filename = download_file(tpf_filename, cache=True)
        except:
            log.warning('Can not find file {}'.format(tpf_filename))
    tpf = fitsio.FITS(tpf_filename)
    hdr_list = tpf[0].read_header_list()
    hdr = {elem['name']:elem['value'] for elem in hdr_list}
    keplerid = int(hdr['KEPLERID'])
    try:
        aperture = tpf[2].read()
    except:
        log.warning('No aperture found for TPF {}'.format(tpf_filename))
    aperture_shape = aperture.shape
    # Get the pixel coordinates of the corner of the aperture
    hdr_list = tpf[1].read_header_list()
    hdr = {elem['name']:elem['value'] for elem in hdr_list}
    col, row = int(hdr['1CRV5P']), int(hdr['2CRV5P'])
    height, width = aperture_shape[0], aperture_shape[1]
    y, x = np.meshgrid(np.arange(col, col + width), np.arange(row, row + height))
    qmask = tpf[1].read()['QUALITY'] & quality_bitmask == 0
    flux = (tpf[1].read()['FLUX'])[qmask]
    cadence = (tpf[1].read()['CADENCENO'])[qmask]
    error = (tpf[1].read()['FLUX_ERR'])[qmask]
    tpf.close()

    return cadence, flux, error, y, x

def make_arrays(name, campaign=None, search_radius=1, aperture_radius=3,
                lag=[0, 0.2, 0.4, 0.6, -0.2, -0.4 -0.6], xoffset=0, yoffset=0):
    '''Make moving TPFs
    '''

    if not hasattr(lag, '__iter__'):
        lag = [lag]

    objs = [get_radec(name, campaign, lag = l) for l in lag]
    timetables = []
    for obj in objs:
        mast, t = get_mast(obj)
        timetables.append(t)
    n = aperture_radius
    x, y = np.meshgrid(np.arange(np.ceil(n).astype(int) * 2 + 1, dtype=float), np.arange(np.ceil(n).astype(int) * 2 + 1, dtype=float))
    aper = ((x - n + 1 + xoffset)**2 + (y - n + 1 + yoffset)**2) < (n**2)
    x[~aper] = np.nan
    y[~aper] = np.nan
    xaper, yaper = np.asarray(x), np.asarray(y)
    xaper, yaper = np.asarray(xaper[np.isfinite(xaper)], dtype=int), np.asarray(yaper[np.isfinite(yaper)], dtype=int)

    ar = np.zeros((len(timetables[0]), np.ceil(n).astype(int) * 2, np.ceil(n).astype(int) * 2, len(timetables))) * np.nan
    er = np.zeros((len(timetables[0]), np.ceil(n).astype(int) * 2, np.ceil(n).astype(int) * 2, len(timetables))) * np.nan


    mastcoord = SkyCoord(mast.RA, mast.Dec, unit=(u.deg, u.deg))
    for file in tqdm(np.arange(len(mast))):
        campaign = mast.campaign[file]
        channel = mast.channel[file]
        timetable = timetables[0]
        tablecoord = SkyCoord(timetable.RA, timetable.Dec, unit=(u.deg, u.deg))
        ok = mastcoord[file].separation(tablecoord) < 50 * 4*u.arcsec
        tab = timetable[['cadenceno', 'order','Column_{}_{}'.format(campaign, channel),'Row_{}_{}'.format(campaign, channel)]][ok]
        end, start = int(mast.endtime[file]), int(mast.starttime[file])
        ok = []
        for t in tab.iterrows():
            ok.append((int(t[1][0]) > start) & (int(t[1][0]) < end))
        ok = np.asarray(ok)
        if not np.any(ok):
            continue

        url = mast.url[file]
        cadence, flux, error, column, row = open_tpf(url)
        pixel_coordinates = np.asarray(['{}, {}'.format(i, j) for i, j in zip(column.ravel(), row.ravel())])

        for idx, timetable in enumerate(timetables):
            tablecoord = SkyCoord(timetable.RA, timetable.Dec, unit=(u.deg, u.deg))
            ok = mastcoord[file].separation(tablecoord) < 50 * 4*u.arcsec
            tab = timetable[['cadenceno', 'order','Column_{}_{}'.format(campaign, channel),'Row_{}_{}'.format(campaign, channel)]][ok]
            for t in tab.iterrows():
                inaperture = np.asarray(['{}, {}'.format(int(i), int(j)) for i, j in zip(xaper - n + t[1][2], yaper - n  + t[1][3])])
                mask_1 = np.asarray([i in pixel_coordinates for i in inaperture])
                if not np.any(mask_1):
                    continue
                mask_2 = np.asarray([i in inaperture for i in pixel_coordinates])
                c = np.where(cadence == int(t[1][0]))[0]
                if len(c) == 0:
                    continue
                ar[int(t[1][1]), xaper[mask_1], yaper[mask_1], idx] = (flux[c[0]].ravel()[mask_2])
                er[int(t[1][1]), xaper[mask_1], yaper[mask_1], idx] = (error[c[0]].ravel()[mask_2])

    return ar, er
