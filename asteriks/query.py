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
import sys
import os
import K2ephem
import matplotlib.pyplot as plt
import pickle
import json
import ast
import re
from copy import deepcopy
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

from .utils import *
from . import PACKAGEDIR
from .plotting import *

class CAFFailure(Exception):
    # There are no CAF files at MAST to use for this object
    pass


class CampaignFailure(Exception):
    # There are no CAF files at MAST to use for this object
    pass


MOV_FILE = os.path.join(PACKAGEDIR, 'data', 'CAF_with_proposers.csv')
MovingBodyMetaData = pd.read_csv(MOV_FILE)
# print(MovingBodyMetaData)
MovingBodyMetaData.loc[~(np.asarray([m == m for m in MovingBodyMetaData.alternate_names])), 'alternate_names'] = ''
MovingBodyMetaData.alternate_names = [m.split('|') for m in MovingBodyMetaData.alternate_names]


def get_bibtex(ID):
    '''Get a bibtex entry for a give K2 GO Proposal
    '''
    if ID == '':
        return ''

    url = 'https://keplerscience.arc.nasa.gov/data/k2-programs/{}.txt'.format(ID)
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data, 'html.parser')
    results = {'ID': ID}
    for item in ['Title:', 'PI:', 'CoIs:']:
        results[item[:-1]] = (str(soup).split(item)[1].split('\n')[0]).split('(')[0].strip()
    abstract = ''
    for s in str(soup).split('\n\n'):
        if np.asarray([s.strip().replace('\n','').startswith(item) for item in ['#', 'Title:', 'CoIs:', 'PI:', 'References:', '[']]).any():
            continue
        abstract = '{}{}'.format(abstract, s.split('References:')[0])

    abstract = (re.sub("(.{64})", "\\1\n\t\t", abstract.replace('\r\n',''), 0, re.DOTALL))
    results['Abstract'] = abstract
    c = int(ID[2:-3])
    if c in [0, 1]:
        date = ['February 2014']
    if c in [2, 3, 4]:
        date = ['June 2014']
    if c in [5, 6, 7]:
        date = ['October 2014']
    if c in [8, 9, 10]:
        date = ['June 2015']
    if c in [11, 12, 13]:
        date = ['Februrary 2016']
    if c in [14, 15, 16]:
        date = ['November 2016']
    if c in [17, 18, 19]:
        date = ['October 2017']
    results['Date'] = date[0]
    results['URL'] = url

    authors = np.append(results['PI'].split(';'), results['CoIs'].split(';'))
    authors = authors[authors != '']
    authors = ' and '.join(['{{{}}}, {}.'.format(auth.strip().split(' ')[0][:-1], auth.strip().split(' ')[0][0]) for auth in authors])
    abstract = (re.sub("(.{92})", "\\1\n\t\t", authors, 0, re.DOTALL))
    bib = ("@MISC{{{0}ktwo.prop{1},\n\tauthor = {{{2}}},"
           "\n\ttitle = {{{3}}},\n\tabstract = {{{6}}}"
           "\n\thowpublished = {{K2 Proposal}},"
           "\n\tyear = {{{0}}},\n\tmonth = {{{4}}},\n\turl = {{{5}}},\n\t"
           "notes = {{K2 Proposal {1}}}\n}}"
            "".format(results['Date'].split(' ')[1], results['ID'], authors, results['Title'], results['Date'].split(' ')[0], results['URL'], results['Abstract']))
    return bib

def find_alternate_names_using_CAF(name):
    mask = np.zeros(len(MovingBodyMetaData), dtype=bool)
#    if len(np.where(MovingBodyMetaData.clean_name == name)[0]) != 0:
#        return name
    for idx, m, n in zip(range(len(MovingBodyMetaData)), MovingBodyMetaData.alternate_names, MovingBodyMetaData.clean_name):
        mask[idx] = np.any(np.asarray([name.split('.')[-1].lower().replace(' ','') == i.lower().replace(' ','') for i in m]))
        mask[idx] |= name.split('.')[-1].lower().replace(' ','') == n.lower().replace(' ','')
    names = [m for m in MovingBodyMetaData[mask].clean_name]
    for m in MovingBodyMetaData[mask].alternate_names.reset_index(drop=True)[0]:
        if len(m) > 1:
            names.append(m)
    return(names)

def find_GO_proposal(name):
    mask = np.zeros(len(MovingBodyMetaData), dtype=bool)
    for idx, m, n in zip(range(len(MovingBodyMetaData)), MovingBodyMetaData.alternate_names, MovingBodyMetaData.clean_name):
        mask[idx] = np.any(np.asarray([name.split('.')[-1].lower().replace(' ','') == i.lower().replace(' ','') for i in m]))
        mask[idx] |= name.split('.')[-1].lower().replace(' ','') == n.lower().replace(' ','')
    IDs = np.asarray([m for m in MovingBodyMetaData[mask].IDS])
    IDs = IDs[[not isinstance(i, float) for i in IDs]]
    return('|'.join(IDs))

def find_PIs(name):
    mask = np.zeros(len(MovingBodyMetaData), dtype=bool)
    for idx, m, n in zip(range(len(MovingBodyMetaData)), MovingBodyMetaData.alternate_names, MovingBodyMetaData.clean_name):
        mask[idx] = np.any(np.asarray([name.split('.')[-1].lower().replace(' ','') == i.lower().replace(' ','') for i in m]))
        mask[idx] |= name.split('.')[-1].lower().replace(' ','') == n.lower().replace(' ','')
    PIs = [m for m in MovingBodyMetaData[mask].PROPOSERS]
    if len(PIs) == 0:
        return ''
    if np.asarray(isinstance(PIs[0], float)):
        return ''
    return('|'.join(PIs))


def _mast_fail(chunk_df):
    fail = (np.asarray(chunk_df)[0][0] == 'no rows found')
    fail |= (np.asarray(chunk_df)[0][0] == '''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"''')
    return fail

def find_mast_files_using_CAF(name, desired_campaign=None):
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
    campaign : int
        Campaign number
    '''
    mask = np.zeros(len(MovingBodyMetaData), dtype=bool)
    for idx, m, n in zip(range(len(MovingBodyMetaData)), MovingBodyMetaData.alternate_names, MovingBodyMetaData.clean_name):
        mask[idx] = np.any(np.asarray([name.split('.')[-1].lower() == i.lower() for i in m]))
        mask[idx] |= name.split('.')[-1].lower() == n.lower()
    mast = pd.DataFrame(columns=['RA', 'Dec', 'EPIC', 'channel'])
    if np.any(mask):
        string = np.asarray(MovingBodyMetaData.obj_name[mask])[0]
        log.debug('{} had a custom mask in K2'.format(string))
        string = np.asarray(MovingBodyMetaData.obj_name[mask])[0]
        MAST_API = 'https://archive.stsci.edu/k2/data_search/search.php?'
        extra = 'outputformat=CSV&action=Search'
        columns = '&selectedColumnsCsv=sci_ra,sci_dec,ktc_k2_id,ktc_investigation_id,sci_channel,sci_campaign'
        query = 'objtype={}&'.format(string.split('.')[-1].replace(' ', '%20'))
        chunk_df = pd.read_csv(MAST_API + query + extra + columns,
                               error_bad_lines=False,
                               names=['RA', 'Dec', 'EPIC', 'Investigation ID', 'channel', 'campaign'])

        # MAST queries are hard and specific...so we have to cycle through a few options.
        if _mast_fail(chunk_df):
            objtype = ['TNO', 'TROJAN', 'ASTEROID', 'COMET', 'MOON', 'PLANET']
            subtype = ['tnotype', 'trojantype', 'asteroidtype', 'comettype', 'moontype', 'planettype']
            for obj, sub in zip(objtype, subtype):
                for j in range(len(string.split('.')[-1].split(' '))):
                    query = 'objtype={}&{}={}&'.format(obj, sub,  '%20'.join(string.split('.')[-1].split(' ')[j:]))
                    chunk_df = pd.read_csv(MAST_API + query + extra + columns,
                               error_bad_lines=False,
                               names=['RA', 'Dec', 'EPIC', 'Investigation ID', 'channel', 'campaign'])
                    if not _mast_fail(chunk_df):
                        break
                if not _mast_fail(chunk_df):
                    break
        if _mast_fail(chunk_df):
            raise CAFFailure('Can not parse name {} into a MAST query?'.format(name))

        chunk_df = chunk_df.dropna(subset=['EPIC']).reset_index(drop=True)
        chunk_df = chunk_df.loc[chunk_df.RA != 'RA (J2000)']
        chunk_df = chunk_df.loc[chunk_df.RA != 'ra']
        mast = mast.append(chunk_df.drop_duplicates(['EPIC']).reset_index(drop=True))
        log.debug('\t{} Files found'.format(len(mast)))
    else:
        raise CAFFailure('Could not find a CAF file for this moving body.')
    if len(mast) == 0:
        raise CAFFailure('Could not find a CAF file for this moving body.')

    campaign = np.asarray(mast.campaign, dtype=float)
    for idx in range(len(campaign)):
        if campaign[idx] > 80:
            campaign[idx] = campaign[idx] // 10

    if desired_campaign is not None:
        if not (campaign == desired_campaign).any():
            raise CampaignFailure('This object is not available in Campaign {}.'
                                  ''.format(desired_campaign))
        mast = mast[campaign==desired_campaign].reset_index(drop=True)
        campaign = np.asarray(mast.campaign, dtype=float)
        for idx in range(len(campaign)):
            if campaign[idx] > 80:
                campaign[idx] = campaign[idx] // 10

    if len(np.unique(campaign)) > 1:
        raise CampaignFailure('This object is available in more than 1 campaign. Please specify.'
                              ' Campaigns: {}'.format(campaign))
    return mast, int(np.unique(campaign))


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
    mask = np.zeros(len(MovingBodyMetaData), dtype=bool)
    campaigns = [campaign]
    if campaign > 8:
        campaigns = [campaign, campaign*10+1, campaign*10+2]
    for c in campaigns:
        mask |= np.asarray([np.any(np.asarray([(i == str(c)) for i in str(m).split('|')]))
                            for m in MovingBodyMetaData.campaign], dtype=bool)
    return deepcopy((MovingBodyMetaData[mask][['NAMES', 'campaign']])).reset_index(drop=True)


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
    mast = pd.DataFrame(columns=['RA', 'Dec', 'EPIC', 'channel'])
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
                               names=['RA', 'Dec', 'EPIC', 'Investigation ID', 'channel'])
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

    c = np.asarray(['{:02}'.format(campaign) in c[0:2] for c in campaign_strb])
    # Only want data from the correct campaign
    ok = np.zeros(len(mast)).astype(bool)
    for c1 in campaign_stra[c]:
        ok |= np.asarray([m[1]['campaign'] == c1 for m in mast.iterrows()])
    ok |= np.asarray([m[1]['campaign'] == campaign for m in mast.iterrows()])
    if not np.any(ok):
        raise CAFFailure
    mast = mast[ok].reset_index(drop=True)
    m = pd.DataFrame(columns=mast.columns)
    for a, b in zip(campaign_stra[c], campaign_strb[c]):
        m1 = mast.copy()
        ids = np.asarray(m1.EPIC, dtype=str)
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


def get_mast(name, campaign=None, timetables=None):
    try:
        mast, campaign = find_mast_files_using_CAF(name, campaign)
        mast = clean_mast_file(mast, campaign)
    except CAFFailure:
        if campaign is None:
            log.exception('There is no campaign file for this'
                          ' object. Run again with a campaign')
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
