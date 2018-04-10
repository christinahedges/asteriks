'''Creates HDF5 files from Kepler/K2 target pixel files.
'''
PACKAGEDIR = '/Users/ch/K2/projects/asteriks/python/' #For now
import glob
import pickle
import re
import os
from contextlib import contextmanager
import warnings
import sys
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import fitsio

from lightkurve import KeplerTargetPixelFile
from astropy.utils.data import download_file,clear_download_cache
from astropy.config.paths import get_cache_dir
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from k2mosaic import mast
import K2ephem
import K2fov

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

WCS_DIR = os.path.join(PACKAGEDIR, 'data', 'wcs/')
DATA_DIR = os.path.join(PACKAGEDIR, 'data', 'database/')

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def open_tpf(tpf_filename, wcs, quality_bitmask=(32768|65536)):
    '''Opens a TPF and returns the flux and error in each pixel with meta data

    Returns a n+6 by m array where n is the number of time stamps and m is the
    number of pixels. Array has meta data in the first 6 columns of RA, Dec, X
    pixel, Y pixel, aperture flag and kepler id.

    Parameters
    ----------
    tpf_filename : str
        Name of the file to open. Can be a URL
    wcs : astropy.coordinates.wcs.WCS object
        astropy wcs object to great the RA/Dec solution
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
    # Fill the data
    mask = aperture > 0
    y, x = np.meshgrid(np.arange(col, col + width), np.arange(row, row + height))
    x, y = x[mask], y[mask]
    qmask = tpf[1].read()['QUALITY'] & quality_bitmask == 0
    flux=(tpf[1].read()['FLUX'])[qmask]
    error=(tpf[1].read()['FLUX_ERR'])[qmask]
    tpf.close()
    f = np.asarray([f[mask].ravel() for f in flux])
    ap = aperture[mask].ravel()
    e = np.asarray([e[mask].ravel() for e in error])
    RA, Dec = wcs.wcs_pix2world(y.ravel(), x.ravel(),1)
    f=np.asarray(np.reshape(np.append([np.asarray(RA,dtype=float),
                                        np.asarray(Dec,dtype=float),
                                        np.asarray(x.ravel(),dtype=float),
                                        np.asarray(y.ravel(),dtype=float),
                                        ap,
                                        np.zeros(ap.shape[0], dtype=int)+keplerid],
                                        f),
                                        (np.shape(f)[0]+6,np.shape(f)[1])),dtype=float)

    e=np.asarray(np.reshape(np.append([np.asarray(RA,dtype=float),
                                        np.asarray(Dec,dtype=float),
                                        np.asarray(x.ravel(),dtype=float),
                                        np.asarray(y.ravel(),dtype=float),
                                        ap,
                                        np.zeros(ap.shape[0],dtype=int)+keplerid],
                                        e),
                                        (np.shape(e)[0]+6,np.shape(e)[1])),dtype=float)
    return f, e


def build_tables(tpf_filenames, campaign=None, channel=None,
    output_prefix='', fname='asteriks', verbose=False, memory_lim=1, overwrite=True,
    dtype=np.float32, return_table=True, quality_bitmask = (32768|65536)):
    '''Mosaic a set of TPFS into a dataframe. NOTE: by default this will be a float32
    dataframe which saves a little on storage space, these can become very large. Use
    these at your own risk.

    Paramters
    ---------
    tpf_filenames : list of str
        List of tpf filenames to create a HDF5 file for
    campaign : int or None
        Camapaign of the data. If None will find from the first tpf file.
    channel : int or None
        Channel of the data. If None will find from the first tpf file.
    Returns
    -------
    '''
    FILEPATH = '{}{}.h5'.format(output_prefix, fname)
    ERROR_FILEPATH = '{}{}_ERR.h5'.format(output_prefix, fname)
    if overwrite:
        if os.path.isfile(FILEPATH):
            if verbose:
                print('Clearing old file')
            os.remove(FILEPATH)
    else:
        print('File already exists. Run with overwrite.')
    if overwrite:
        if os.path.isfile(ERROR_FILEPATH):
            if verbose:
                print('Clearing old error file')
            os.remove(ERROR_FILEPATH)
    else:
        print('File already exists. Run with overwrite.')

    if tpf_filenames[0].startswith("http"):
        if verbose:
            tpf_filename = download_file(tpf_filenames[0], cache=True)
        else:
            with silence():
                tpf_filename = download_file(tpf_filenames[0], cache=True)
    else:
        tpf_filename = tpf_filenames[0]

    #Get basic data from the first tpf. Get rid of NoData and NoFinePoint flags
    tpf = KeplerTargetPixelFile(tpf_filename, quality_bitmask=quality_bitmask)
    if campaign is None:
        try:
            campaign = tpf.campaign
        except:
            print('No campaign number found. Currently only supports K2 data.')
            return
    if channel is None:
        channel = tpf.channel
#    import pdb; pdb.set_trace()
    timejd = np.asarray(tpf.timeobj.jd - 2454833, dtype=np.float32)
    cadenceno = tpf.hdu[1].data['CADENCENO'][tpf.quality_mask]

    #Load the WCS from the package directory
    wcs_file=WCS_DIR+'c{0:02}_'.format(campaign)+'{0:02}.p'.format(channel)
    if verbose:
        print('Loading WCS')
    with silence():
        r = pickle.load(open(wcs_file,'rb'))

    #Set up the dataframe
    cols = np.asarray(cadenceno, dtype=int)
    cols = np.append(['RA','Dec','Row','Column','APERFLAG','ID'], cols)
    df = pd.DataFrame(columns=cols, dtype=dtype)
    edf = pd.DataFrame(columns=cols, dtype=dtype)

    #Make the first row the time stamps
    df = df.append(pd.DataFrame(np.atleast_2d(np.append(np.zeros(6), timejd)),
                   columns=cols, dtype=dtype))
    df.index = [-1]
    edf = edf.append(pd.DataFrame(np.atleast_2d(np.append(np.zeros(6), timejd)),
                    columns=cols, dtype=dtype))
    edf.index = [-1]
    df.to_hdf(FILEPATH, 'table', append=True)
    df=pd.DataFrame(columns=cols, dtype=dtype)
    edf.to_hdf(ERROR_FILEPATH, 'table', append=True)
    edf=pd.DataFrame(columns=cols, dtype=dtype)


    for i,tpf_filename in enumerate(tqdm(tpf_filenames)):
        f, e = open_tpf(tpf_filename, r, quality_bitmask)
        df=df.append(pd.DataFrame(f.T, columns=cols, dtype=dtype))
        edf = edf.append(pd.DataFrame(e.T, columns=cols, dtype=dtype))
#        import pdb;pdb.set_trace()
        mem=np.nansum(df.memory_usage())/1E9
        if mem>=memory_lim/2:
            if verbose:
                print('\n{} gb memory limit reached after {} TPFs. Appending to file.'.format(memory_lim,i))
            df.to_hdf(FILEPATH, 'table', append=True)
            df=pd.DataFrame(columns=cols, dtype=dtype)
            edf.to_hdf(ERROR_FILEPATH, 'table', append=True)
            edf=pd.DataFrame(columns=cols, dtype=dtype)
            if return_table:
                print('Memory limit reached. Cannot return table.')
                return_table=False
        df.to_hdf(FILEPATH, 'table', append=True)
        edf.to_hdf(ERROR_FILEPATH, 'table', append=True)
    return


def build_database(dir=DATA_DIR, input_directory=None, verbose=True, cachelim=30,
    overwrite=False, campaigns=None, channels=None, memory_lim=4):
    '''Creates a database of HDF5 files from a directory of tpfs.'''

    print ('-------------------------------')
    print ('Building K2 TPF HDF5 database.')
    if not verbose:
        print('Trying running with verbose enabled for more information.')
    if (os.path.isdir(WCS_DIR) == False):
        print ('No WCS Files Found')
    if not os.path.isdir(dir):
        if verbose:
            print('Creating Directory')
        os.makedirs(dir)
    if input_directory is None:
        print('No input directory. Building URLS using k2mosaic.')
    else:
        print('Input directory: {}'.format(input_directory))
        print('Assuming MAST-like structure.')

    if verbose:
        if overwrite:
            print('Overwrite enabled.')
    if campaigns is None:
        campaigns=range(14)
    if channels is None:
        channels=range(1,85)

    for campaign in campaigns:
        cdir='{}'.format(dir)+'c{0:02}/'.format(campaign)
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        for ext in channels:
            edir='{}'.format(cdir)+'{0:02}/'.format(ext)
            if not os.path.isdir(edir):
                os.makedirs(edir)
            if (os.path.isfile('{}'.format(edir)+'0.h5')):
                if overwrite==False:
                    if verbose:
                        print('File Exists. Skipping. Set overwrite to True to overwrite.')
                    continue
            try:
                urls = mast.get_tpf_urls('c{}'.format(campaign), ext)
            except:
                if verbose:
                    print('Channel {} : No URLS found?'.format(ext))
                continue
            cache_size=get_dir_size(get_cache_dir())/1E9
            if verbose:
                print ('-------------------------------')
                print ('Campaign:\t {}'.format(campaign))
                print ('Channel:\t {}'.format(ext))
                print ('-------------------------------')
                print ('{} Files'.format(len(urls)))
                print ('{0:.2g} gb in astropy cache'.format(cache_size))
            if cache_size>=cachelim:
                print ('Cache hit limit of {} gb. Clearing.'.format(cachelim))
                clear_download_cache()

            if (input_directory is None)==False:
                if verbose:
                    print('Building from input')
                tpf_filenames=np.asarray(['{}{}'.format(input_directory,u.split('https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]) for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    tpf_filenames=np.asarray(['{}{}'.format(input_directory,(u.split('https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1])).split('.gz')[0] for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    if verbose:
                        print ('No MAST structure...trying again.')
                    tpf_filenames=np.asarray(['{}{}'.format(input_directory,(u.split('https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]).split('/')[-1]) for u in urls])
                if os.path.isfile(tpf_filenames[0]) is False:
                    tpf_filenames=np.asarray(['{}{}'.format(input_directory,((u.split('https://archive.stsci.edu/missions/k2/target_pixel_files/')[-1]).split('/')[-1])).split('.gz')[0] for u in urls])
            else:
                if verbose:
                    print ('Downloading/Caching')
                tpf_filenames=[None]*len(urls)
                if verbose:
                    for i,u in enumerate(urls):
                        tpf_filenames[i] = download_file(u,cache=True)
                else:

                    with click.progressbar(length=len(urls)) as bar:
                        for i,u in enumerate(urls):
                            with silence():
                                tpf_filenames[i] = download_file(u,cache=True)
                            bar.update(1)
                tpf_filenames=np.asarray(tpf_filenames)
            if verbose:
                [print(t) for t in tpf_filenames[0:10]]
                print('...')
            print('Building Campaign {} Channel {}'.format(campaign,ext))
            build_tables(tpf_filenames, campaign, ext,
                        output_prefix='{}'.format(edir), verbose=verbose,
                        memory_lim=memory_lim)
            if verbose:
                print ('Campaign {} Complete'.format(campaign))
                print ('-------------------------------')
    print ('ALL DONE')
    print ('-------------------------------')
