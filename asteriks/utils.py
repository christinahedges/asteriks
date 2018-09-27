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
from astropy.io import fits
from astropy.utils.data import download_file, clear_download_cache
import pandas as pd
import pickle
from . import PACKAGEDIR
import fitsio
from lightkurve.targetpixelfile import KeplerTargetPixelFileFactory, KeplerTargetPixelFile

log = logging.getLogger('\tASTERIKS ')

campaign_stra = np.asarray(['1', '2', '3', '4', '5', '6', '7', '8', '91', '92',
                            '101', '102', '111', '112', '12', '13', '14', '15',
                            '16', '17', '18'])
campaign_strb = np.asarray(['01', '02', '03', '04', '05', '06', '07', '08', '91',
                            '92', '101', '102', '111', '112', '12', '13', '14',
                            '15', '16', '17', '18'])

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
LC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'lc_meta.p')
SC_TIME_FILE = os.path.join(PACKAGEDIR, 'data', 'sc_meta.p')


# asteriks is ONLY designed to work with the following quality flag.
# change it at your own risk.
quality_bitmask = (32768 | 65536)


@contextmanager
def silence():
    '''Suppreses all output'''
    logger = logging.getLogger()
    logger.disabled = True
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
    cache_size = get_dir_size(get_cache_dir())/1E9
    if cache_size >= cache_lim:
        log.warning('Cache hit limit of {} gb. Clearing.'.format(cachelim))
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
                         names=['RA', 'Dec', 'EPIC', 'Investigation ID', 'channel'])
        df = df.dropna(subset=['EPIC']).reset_index(drop=True)
        df = df.loc[df.RA != 'RA (J2000)']
        df = df.loc[df.RA != 'ra']
        if len(df) != 0:
            ids.append(np.asarray(df.EPIC)[0])
    return ids


def create_meta_data(campaigns=np.asarray(['1', '2', '3', '4', '5', '6', '7', '8', '91', '92',
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
                tpf = KeplerTargetPixelFile.from_archive(i, quality_flag=(32768 | 65536),
                                                         cadence='short')
        except ArchiveError:
            log.warning('Could not download {} in Campaign {}'.format(i, c))
            continue
        sc_meta['{}'.format(c)] = {'cadenceno': tpf.cadenceno, 'time': tpf.timeobj.jd}
    pickle.dump(sc_meta, open(SC_TIME_FILE, 'wb'))
    log.info('Meta data saved to {}.'.format(SC_TIME_FILE))

    log.info('Finding long cadence meta data.')
    ids = return_first_mast_hit(campaigns, 'LC')
    lc_meta = {}
    for idx, i, c in zip(range(len(ids)), ids, campaigns):
        log.info('\tCampaign {} (ID:{})'.format(c, i))
        try:
            with silence():
                tpf = KeplerTargetPixelFile.from_archive(
                    i, quality_flag=(32768 | 65536), campaign=c)
        except ArchiveError:
            log.warning('Could not download {} in Campaign {}'.format(i, c))
            continue

        lc_meta['{}'.format(c)] = {'cadenceno': tpf.cadenceno, 'time': tpf.timeobj.jd}
    pickle.dump(lc_meta, open(LC_TIME_FILE, 'wb'))
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
            log.warning('Cannot find file {}'.format(tpf_filename))
    tpf = fitsio.FITS(tpf_filename)

    hdr_list = tpf[0].read_header_list()
    hdr = {elem['name']: elem['value'] for elem in hdr_list}
    keplerid = int(hdr['KEPLERID'])
    try:
        aperture = tpf[2].read()
    except:
        log.warning('No aperture found for TPF {}'.format(tpf_filename))
    aperture_shape = aperture.shape
    # Get the pixel coordinates of the corner of the aperture
    hdr_list = tpf[1].read_header_list()
    hdr = {elem['name']: elem['value'] for elem in hdr_list}
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


def build_aperture(n=5, shape='square', xoffset=0, yoffset=0):
    '''Create the X and Y locations for a circular aperture.

    Parameters
    ----------
    n : float
        Aperture radius
    '''
    log.debug('Building apertures')
    if isinstance(n, int):
        n1, n2 = n, n
    if hasattr(n, '__iter__'):
        if (len(np.asarray(n).shape) == 1):
            n1, n2 = n[0], n[1]
        elif not isinstance(n[0], bool):
            log.error('Please pass a radius, two length dimensions or '
                      'a boolean array')

    if 'n1' in locals():
        x, y = np.meshgrid(np.arange(np.ceil(n1).astype(int) * 2 + 1, dtype=float),
                           np.arange(np.ceil(n2).astype(int) * 2 + 1, dtype=float))
        if isinstance(shape, str):
            if shape == 'circular':
                aper = ((x - n1 + xoffset)**2 + (y - n2 + yoffset)**2) < (np.max([n1, n2])**2)
            if shape == 'square':
                aper = np.ones(x.shape, dtype=bool)
    else:
        x, y = np.meshgrid(np.arange(aper.shape[0], dtype=float),
                           np.arange(aper.shape[1], dtype=float))
        aper = n
    x[~aper] = np.nan
    y[~aper] = np.nan
    xaper, yaper = np.asarray(x), np.asarray(y)
    xaper, yaper = np.asarray(xaper[np.isfinite(xaper)], dtype=int), np.asarray(
        yaper[np.isfinite(yaper)], dtype=int)
    return xaper, yaper, aper


def fix_aperture(aper):
    '''Make sure the aperture is one continuous blob.
    '''
    idx = 0
    jdx = 0
    okaper = np.copy(aper*False)
    for idx in np.arange(1, aper.shape[0] - 1):
        for jdx in np.arange(1, aper.shape[1] - 1):
            okaper[idx, jdx] |= np.any(
                np.any([(aper[idx][jdx + 1]), (aper[idx][jdx - 1]), (aper[idx - 1][jdx]), (aper[idx + 1][jdx])]))
    aper *= okaper
    x, y = np.where(aper == True)
    n = np.shape(aper)[0]/2
    ok = ((x - n)**2 + (y - n)**2)**0.5 < 4
    x = x[ok]
    y = y[ok]
    aper *= np.zeros(aper.shape, dtype=bool)
    aper[x, y] = True



def _make_aperture_extension(self, aperture):
    """Create the aperture mask extension (i.e. extension #2)."""
    hdu = fits.ImageHDU(aperture*3)

    # Set the header from the template TPF again
    template = self._header_template(2)
    for kw in template:
        if kw not in ['XTENSION', 'NAXIS1', 'NAXIS2', 'CHECKSUM', 'BITPIX']:
            try:
                hdu.header[kw] = (self.keywords[kw],
                                  self.keywords.comments[kw])
            except KeyError:
                hdu.header[kw] = (template[kw],
                                  template.comments[kw])

    # Override the defaults where necessary
    for keyword in ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CUNIT1',
                    'CUNIT2', 'CDELT1', 'CDELT2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
            hdu.header[keyword] = ""  #override wcs keywords
    hdu.header['EXTNAME'] = 'APERTURE'
    return hdu


def build_tpf(r, time, name, aper):
    '''Build a tpf using lightkurve factory
    '''
    ok = np.where(np.nansum(r['ar'][:, :, :, 0] - r['diff'][:, :, :, 0], axis=(1,2)) != 0)[0]
    ar = r['ar'][ok, :, :, 0] - r['diff'][ok, :, :, 0]
    er = (r['er'][ok, :, :, 0]**2 + r['ediff'][ok, :, :, 0]**2)**0.5


    fac = KeplerTargetPixelFileFactory(ar.shape[0],ar.shape[1],ar.shape[2], name)
    fac.time = time[ok]

    for i, a, e in zip(range(len(ar)), ar, er):
        fac.add_cadence(i, flux=a, flux_err=e, raw_cnts=r['ar'][ok[i], :, :, 0], flux_bkg=r['diff'][ok[i], :, :, 0], flux_bkg_err=r['ediff'][ok[i], :, :, 0])
    target = fac._make_target_extension()
    target.header['1CRV5P'] = 0
    target.header['2CRV5P'] = 0
    tpf = KeplerTargetPixelFile(fits.HDUList([fac._make_primary_hdu(), target, _make_aperture_extension(fac, aper)]))
    return tpf

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
