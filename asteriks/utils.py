'''Common functions for asteriks
'''

import os
from contextlib import contextmanager
import warnings
import sys
import numpy as np
import astropy.units as u
import logging
import K2fov
from astropy.time import Time
from astropy.utils.data import download_file, clear_download_cache
import pandas as pd
import pickle
from . import PACKAGEDIR

from lightkurve import KeplerTargetPixelFile
from lightkurve.mast import ArchiveError

import fitsio


log = logging.getLogger('\tASTERIKS ')
loggingLevels = {'CRITICAL' : 50,
                 'ERROR': 40,
                 'WARNING': 30,
                 'INFO': 20,
                 'DEBUG': 10}

LC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'lc_meta.p')
SC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'sc_meta.p')

# asteriks is ONLY designed to work with the following quality flag.
# change it at your own risk.
quality_bitmask=(32768|65536)

def setLevel(level):
    '''Set the level of logging for asteriks.

    Parameters
    ----------
    level : str
        Choose from CRITICAL, ERROR, WARNING, INFO and DEBUG
    '''
    if level not in loggingLevels:
        log.setLevel(20)
        log.error(" No logging level of '{}'. Setting to INFO".format(level))
    else:
        log.setLevel(loggingLevels[level])

setLevel('DEBUG')


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
    '''Turns array 'a' in to a list of n arrays with equal length
    '''
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def check_cache(cache_lim=2):
    '''Checks the astropy cache. If above cachelim, clears the cache.
    '''
    cache_size=get_dir_size(get_cache_dir())/1E9
    if cache_size>=cache_lim:
        logging.warning('Cache hit limit of {} gb. Clearing.'.format(cachelim))
        clear_download_cache()


def return_first_mast_hit(campaigns, cadence='LC'):
    '''Returns a list of ids which are the first hits for the
    specified campaigns at MAST
    '''
    MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
    extra = 'outputformat=CSV&action=Search'
    columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel'
    ids = []
    log.info('Finding {} ids'.format(cadence))
    for c in campaigns:
        query = 'sci_campaign={}&ktc_target_type={}&max_records=1&'.format(c, cadence)
        df = pd.read_csv(MAST_API + query + extra + columns,
                                       error_bad_lines=False,
                                       names=['RA','Dec','EPIC','Investigation ID', 'channel'])
        df = df.dropna(subset=['EPIC']).reset_index(drop=True)
        df = df.loc[df.RA != 'RA (J2000)']
        df = df.loc[df.RA != 'ra']
        if len(df) != 0:
            ids.append(np.asarray(df.EPIC)[0])
    return ids

def create_meta_data(campaigns = np.asarray(['1', '2', '3','4', '5', '6', '7' ,'8', '91', '92',
                            '101', '102', '111', '112', '12', '13', '14', '15',
                            '16', '17', '18'], dtype=int)):
    '''Creates the meta data for K2. If there has been a new release on MAST, run this.
    '''
    log.warning('Finding meta data. This requires an online connection and files to be downloaded from MAST. '
                'This may take several hours.')

    log.info('Finding short cadence meta data.')
    ids = return_first_mast_hit(campaigns, 'SC')
    sc_meta = {}
    for idx, i, c in zip(range(len(ids)), ids, campaigns):
        log.info('\tCampaign {}'.format(c))
        try:
            with silence():
                tpf = KeplerTargetPixelFile.from_archive(i, quality_flag=(32768|65536),
                                                         cadence='short')
        except ArchiveError:
            log.warning('Could not download {} in Campaign {}'.format(i, c))
            continue
        sc_meta['{}'.format(c)] = {'cadenceno':tpf.cadenceno, 'time':tpf.timeobj.jd}
    pickle.dump(sc_meta, open(SC_TIME_FILE,'wb'))
    log.info('Meta data saved to {}.'.format(SC_TIME_FILE))

    log.info('Finding long cadence meta data.')
    ids = return_first_mast_hit(campaigns, 'LC')
    lc_meta = {}
    for idx, i, c in zip(range(len(ids)), ids, campaigns):
        log.info('\tCampaign {} (ID:{})'.format(c, i))
        try:
            with silence():
                tpf = KeplerTargetPixelFile.from_archive(i, quality_flag=(32768|65536), campaign=c)
        except ArchiveError:
            log.warning('Could not download {} in Campaign {}'.format(i, c))
            continue

        lc_meta['{}'.format(c)] = {'cadenceno':tpf.cadenceno, 'time':tpf.timeobj.jd}
    pickle.dump(lc_meta, open(LC_TIME_FILE,'wb'))
    log.info('Meta data saved to {}.'.format(LC_TIME_FILE))

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
    poscorr1 = (tpf[1].read()['POS_CORR1'])[qmask]
    poscorr2 = (tpf[1].read()['POS_CORR2'])[qmask]
    tpf.close()

    return cadence, flux, error, y, x, poscorr1, poscorr2
