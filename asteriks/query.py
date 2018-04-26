'''Queries MAST for files relating to an asteroid
'''

from astropy.utils.data import download_file
import pandas as pd
import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import logging
import warnings
from contextlib import contextmanager
import sys, os
import K2ephem
import matplotlib.pyplot as plt
import pickle
import json
import ast
from copy import deepcopy

from .utils import *
from . import PACKAGEDIR
from . import plotting

class CAFFailure(Exception):
    # There are no CAF files at MAST to use for this object
    pass


MOV_FILE = os.path.join(PACKAGEDIR, 'data', 'moving_objects.csv')
mov = pd.read_csv(MOV_FILE)
mov.NAMES = [ast.literal_eval(m.replace("""' '""","""','""")) for m in mov.NAMES]


def find_mast_files_using_CAF(name):
    '''Find all the custom aperture files for a specific moving object.

    Parameters
    ----------
    name : str
        Name of moving object to query
    Returns
    -------
    mast : pandas.DataFrame
        DataFrame containing the EPIC ids, campaigns and channels of all files in
        the custom aperture for the object
    '''
    mask = np.zeros(len(mov), dtype=bool)
    for idx, m in enumerate(mov.NAMES):
        try:
            mask[idx] = np.any(np.asarray([name.split('.')[-1].lower() in i.lower() for i in m]))
        except:
            continue
    mast = pd.DataFrame(columns=['RA','Dec','EPIC', 'channel'])
    if np.any(mask):
        string = np.asarray(mov.STRING[mask])[0]
        log.debug('{} had a custom mask in K2'.format(string))
        string = np.asarray(mov.STRING[mask])[0]
        MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
        extra = 'outputformat=CSV&action=Search'
        columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel,sci_campaign'
        query = 'objtype={}&'.format(string.split('.')[-1].replace(' ','%20'))
        chunk_df = pd.read_csv(MAST_API + query + extra + columns,
                               error_bad_lines=False,
                               names=['RA','Dec','EPIC','Investigation ID', 'channel', 'campaign'])
        chunk_df = chunk_df.dropna(subset=['EPIC']).reset_index(drop=True)
        chunk_df = chunk_df.loc[chunk_df.RA != 'RA (J2000)']
        chunk_df = chunk_df.loc[chunk_df.RA != 'ra']
        mast = mast.append(chunk_df.drop_duplicates(['EPIC']).reset_index(drop=True))
        log.debug('\t{} Files found'.format(len(mast)))
    else:
        log.error('Could not find a CAF file for this moving body.')
        raise CAFFailure()
    return mast


def find_moving_objects_in_campaign(campaign=2):
    '''Finds the names of all moving objects in a given campaign with a custom
    aperture file.

    Parameters
    ----------
    campaign : int
        A campaign number to search
    Returns
    -------
    obj : list
        List of names of asteroids
    '''
    mask = np.zeros(len(mov), dtype=bool)
    campaigns = [campaign]
    if campaign > 8:
        campaigns = [campaign, campaign*10+1, campaign*10+2]
    for c in campaigns:
        mask |= np.asarray([np.any(np.asarray([(i == str(c)) for i in str(m).split('|')]))
                          for m in mov.campaign], dtype=bool)
    return deepcopy((mov[mask][['NAMES', 'campaign']])).reset_index(drop=True)



def find_all_nearby_files(RA, Dec, Channels, campaign, search_radius=(100*4.)/60.):
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

    ra, dec, channel = np.asarray(RA), np.asarray(Dec), np.asarray(Channels)
    ra_chunk = list(chunk(ra, int(np.ceil(len(ra)/100))))
    dec_chunk = list(chunk(dec, int(np.ceil(len(ra)/100))))
    channel_chunk = list(chunk(channel, int(np.ceil(len(ra)/100))))

    MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
    extra = 'outputformat=CSV&action=Search'
    columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel'
    mast = pd.DataFrame(columns=['RA','Dec','EPIC', 'channel'])
    campaign = np.unique(campaign)[0]
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
    mast['campaign'] = campaign
    return mast


def clean_mast_file(mast, campaign):
    '''Add URLs and cut down to the right campaign.
    '''
    ids = np.asarray(mast.EPIC, dtype=str)
    c = np.asarray(['{:02}'.format(campaign) in c[0:2] for c in campaign_strb])
    # Only want data from the correct campaign
    ok = np.zeros(len(mast)).astype(bool)
    for c1 in campaign_stra[c]:
        ok |= np.asarray([m[1]['campaign'] == c1 for m in mast.iterrows()])
    ok |= np.asarray([m[1]['campaign'] == campaign for m in mast.iterrows()])
    if not np.any(ok):
        raise CAFFailure
    mast = mast[ok].reset_index(drop=True)
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

def get_mast(name, campaign, timetables=None):
    try:
        mast = find_mast_files_using_CAF(name)
        mast = clean_mast_file(mast, campaign)
    except CAFFailure:
        if timetables is None:
            log.exception('There is no CAF file and no timetable for this'
                          ' object. Run again with a timetable')
        log.info('There was no CAF file for {}. Querying nearby.'.format(name))
        mast = find_all_nearby_files(np.asarray(timetables[0].ra),
                                          np.asarray(timetables[0].dec),
                                          np.asarray(timetables[0].channel),
                                          np.asarray(timetables[0].campaign))
        mast = clean_mast_file(mast, campaign)
    return mast
