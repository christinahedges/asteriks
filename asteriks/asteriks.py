"Makes light curves of moving objects in K2 data"
PACKAGEDIR = '/Users/ch/K2/projects/asteriks/python/' #For now

import logging
import numpy as np
import pandas as pd
import K2ephem
import K2fov
from tqdm import tqdm
import pickle

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.data import download_file, clear_download_cache
import astropy.units as u

import fitsio

from lightkurve import KeplerTargetPixelFile

from . import utils

campaign_stra = np.asarray(['1', '2', '3','4', '5', '6', '7' ,'8', '91', '92',
                            '101', '102', '111', '112', '12', '13', '14', '15',
                            '16', '17', '18'])
campaign_strb = np.asarray(['01', '02', '03','04', '05', '06', '07' ,'08', '91',
                            '92', '101', '102', '111', '112', '12', '13', '14',
                            '15', '16', '17', '18'])

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')


def verify_lag():
    pass

def get_radec(name, campaign=None, lag=[0]):
    '''Finds RA and Dec of moving object using K2 ephem.

    When lag is specified, will interpolate the RA and Dec and find the solution
    the specified number of days ahead or behind the object.

    e.g. A lag of 0.25 will find the position of the point a quarter of a day ahead
    of the moving object. A lag of -0.25 will find the position of the point a
    quarter of a day behind the moving object.

    This is used to create a moving aperture before and behind the target aperture.

    Parameters
    ----------
    name : str
        Name of object, resolved by JPL small bodies
    campaign : int or None
        Campaign number in K2. If None, campaigns will be stepped through until
        a campaign containing the object is reached.
    lag : list of floats
        List of lags at which to return the dataframe. Must specify at least one.

    Returns
    -------
    dfs : list of pandas.DataFrame
        List of dataframes containing Julian Date, RA, Dec, Campaign, and channel.
    '''

    if campaign is None:
        campaigns = []
        for c in tqdm(np.arange(1,18)):
            with utils.silence():
                df = K2ephem.get_ephemeris_dataframe(name, c, c, step_size=1./(8))
            k = K2fov.getKeplerFov(c)
            onsil = np.zeros(len(df), dtype=bool)
            for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
                try:
                    onsil[i] = k.isOnSilicon(r, d, 1)
                except:
                    continue
            if np.any(onsil):
                campaigns.append(c)
                df = df[onsil]
                break
        if len(campaigns) == 0:
            logging.exception('{} never on Silicon'.format(name))
        campaign = campaigns[0]
    else:
        with utils.silence():
            df = K2ephem.get_ephemeris_dataframe(name, campaign, campaign, step_size=1./(8))
            k = K2fov.getKeplerFov(campaign)
            onsil = np.zeros(len(df), dtype=bool)
            for i, r, d in zip(range(len(df)), list(df.ra), list(df.dec)):
                try:
                    onsil[i] = k.isOnSilicon(r, d, 1)
                except:
                    continue
            df = df[onsil]

    x = np.asarray([k.getChannelColRow(r, d) for r, d in zip(df.ra, df.dec)])
    df['channel'] = x[:,0]
    times = [t[0:23] for t in np.asarray(df.index, dtype=str)]
    df['jd'] = Time(times,format='isot').jd
    df['campaign'] = campaign
    df = df[['jd', 'ra', 'dec', 'campaign', 'channel']]

    dfs = []
    for l in lag:
        df1 = df.copy()
        ra, dec = np.interp(df1.jd + lag, df1.jd, df1.ra) * u.deg, np.interp(df1.jd + lag, df1.jd, df1.dec) * u.deg
        df1['ra'] = ra
        df1['dec'] = dec
        dfs.append(df1)
    return dfs

def get_mast(obj, search_radius=1.):
    '''Queries mast for object.
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
        tpf = KeplerTargetPixelFile(tpf_filename, quality_bitmask=(32768|65536))
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

def open_tpf(tpf_filename, quality_bitmask=(32768|65536)):
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
            logging.warning('Can not find file {}'.format(tpf_filename))
    tpf = fitsio.FITS(tpf_filename)
    hdr_list = tpf[0].read_header_list()
    hdr = {elem['name']:elem['value'] for elem in hdr_list}
    keplerid = int(hdr['KEPLERID'])
    try:
        aperture = tpf[2].read()
    except:
        logging.warning('No aperture found for TPF {}'.format(tpf_filename))
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
