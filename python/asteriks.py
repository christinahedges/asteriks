"Makes light curves of moving objects in K2 data"
PACKAGEDIR = '/Users/ch/K2/projects/asteriks/python/' #For now
import os
from contextlib import contextmanager
import warnings
import sys
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

campaign_stra = np.asarray(['1', '2', '3','4', '5', '6', '7' ,'8', '91', '92',
                            '101', '102', '111', '112', '12', '13', '14', '15',
                            '16', '17', '18'])
campaign_strb = np.asarray(['01', '02', '03','04', '05', '06', '07' ,'08', '91',
                            '92', '101', '102', '111', '112', '12', '13', '14',
                            '15', '16', '17', '18'])
WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')

@contextmanager
def silence():
    '''Suppreses all output'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

def chunk(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def check_cache(cache_lim=2):
    cache_size=get_dir_size(get_cache_dir())/1E9
    if cache_size>=cache_lim:
        logging.warning('Cache hit limit of {} gb. Clearing.'.format(cachelim))
        clear_download_cache()

def get_radec(name, campaign=None, lag=0):
    '''Finds RA and Dec of object using K2 ephem
    '''
    if campaign is None:
        campaigns = []
#        with silence():
        for c in tqdm(np.arange(1,18)):
            with silence():
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
        with silence():
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
    ra, dec = np.interp(df.jd + lag, df.jd, df.ra) * u.deg, np.interp(df.jd + lag, df.jd, df.dec) * u.deg
    df['ra'] = ra
    df['dec'] = dec
    return df

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
            wcs = pickle.load(open('{}/c{}_{}.p'.format(WCS_DIR, b, ch), 'rb'))
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

def make_lcs(name, campaign=None, search_radius=1, aperture_radius=5, lag=0.25):
    '''Make light curves of moving objects at various aperture sizes
    '''
    x, y = np.meshgrid(np.arange(aperture_radius * 2 + 1) - aperture_radius,
                       np.arange(aperture_radius * 2 + 1)- aperture_radius)
    aper = ((x)**2 + (y)**2) < (n**2)
    x[~aper] = np.nan
    y[~aper] = np.nan
    xaper, yaper = np.asarray(x), np.asarray(y)
    xaper, yaper = xaper[np.isfinite(xaper)], yaper[np.isfinite(yaper)]

    obj = get_radec(name=name, campaign=campaign)
    mast, timetable = get_mast(obj, search_radius=search_radius)
    for idx in range(mast.index.max()):
        for u in mast.loc[idx,'url']:
            open_tpf(u)
            return
