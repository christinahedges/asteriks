from astropy.utils.data import download_file
import pandas as pd
import numpy as np
from tqdm import tqdm
from astroquery.simbad import Simbad
import logging
import warnings
from contextlib import contextmanager
import sys, os
import K2ephem
import matplotlib.pyplot as plt
import pickle
import json

from .utils import *



def get_moving_objects(outfile='out.p', plot=True):
    obj=pd.DataFrame(columns=['InvestigationID', 'Name', 'Campaign', 'EPICs', 'dist', 'tpfs', 'minmag', 'maxmag'])
    k=0
    for campaign in tqdm(range(20), desc='Finding TILES'):
        targlisturl = 'https://keplerscience.arc.nasa.gov/data/campaigns/c{}/K2Campaign{}targets.csv'.format(campaign,campaign)
        with silence():
            targlistfname = download_file(targlisturl, cache=True)
        df = pd.read_csv(targlistfname)
        col = np.asarray(df.columns)[np.asarray(['RA' in d for d in df.columns])][0]
        if isinstance(df[col][0], str):
            mask = np.asarray([len(d.strip()) for d in df[col]])==0
        else:
            mask = np.isfinite(df[col])
        col = np.asarray(df.columns)[np.asarray(['Investigation' in d for d in df.columns])][0]
        ids1=np.asarray(df[col][mask])
        holdids=[]
        ids=[]

        for i in ids1:
            for j in np.unique(i.split('|')):
                ids.append(j)
                if len(np.unique(i.split('|')))==1:
                    holdids.append(j)
        holdids=np.unique(holdids)
        ids=np.asarray(ids)

        ids=np.unique(ids)
        mask=[('TILE' in i) for i in ids]
        mask=np.any([mask,np.in1d(ids,holdids)],axis=0)

        ids=ids[mask]
        for i in ids:
            i=i.strip(' ')
            i2=np.asarray(i.split('_'))
            pos=np.where((i2!='LC')&(i2!='SSO')&(i2!='TILE')&(i2!='TROJAN')&(i2!='SC'))[0]
            i2=' '.join(i2[pos])
            i2=np.asarray(i2.split('-'))
            pos=np.where((i2!='LC')&(i2!='SSO')&(i2!='TILE')&(i2!='TROJAN')&(i2!='SC'))[0]
            i2=' '.join(i2[pos])

            if (i2[0:4].isdigit()) and (~i2[4:6].isdigit()) and (i2[6:].isdigit()):
                i2=' '.join([i2[0:4],i2[4:]])
            obj.loc[k]=(np.transpose([i,i2,campaign,'',0,0,0,0]))
            k+=1


        for i in ids:
            mask=[i in d for d in df[col]]
            epic=np.asarray(df[df.columns[0]][mask])
            loc=np.where(obj.InvestigationID==i.strip())[0]
            obj.loc[loc[0],'EPICs']=list(epic)
            obj.loc[loc[0],'tpfs']=len(epic)
    obj=obj.reset_index(drop=True)
    with tqdm(total=len(obj), desc='Querying JPL') as pbar:
        for i,o in obj.iterrows():
            try:
                with silence():
                    df=K2ephem.get_ephemeris_dataframe(o.Name,o.Campaign,o.Campaign)
                obj.loc[i,['minmag']]=float(np.asarray(df.mag).min())
                obj.loc[i,['maxmag']]=float(np.asarray(df.mag).max())
                ra,dec=np.asarray(df.ra),np.asarray(df.dec)
                obj.loc[i,['dist']]=(np.nansum(((ra[1:]-ra[0:-1])**2+(dec[1:]-dec[0:-1])**2)**0.5))
                pbar.update()
            except:
                continue
                pbar.update()
    obj = obj[np.asarray(obj.dist, dtype=float) != 0].reset_index(drop=True)
    log.info('Saved file to {}'.format(outfile))
    pickle.dump(obj,open(outfile,'wb'))
    return obj

def plot_tracks(obj, img_dir=''):
    fpurl='https://raw.githubusercontent.com/KeplerGO/K2FootprintFiles/master/json/k2-footprint.json'
    fpfname=download_file(fpurl,cache=True)
    fp = json.load(open(fpfname))

    with tqdm(total=20) as pbar:
        for campaign in range(20):
            o=obj[np.asarray(obj.Campaign,dtype=float)==campaign]
            nepics = np.asarray([len(o) for o in np.asarray(obj.EPICs)])
            srcs=len(nepics[nepics>=3])
            if srcs==0:
                pbar.update()
                continue
            fig,ax=plt.subplots(figsize=(10,10))
            for module in range(100):
                try:
                    ch = fp["c{}".format(campaign)]["channels"]["{}".format(module)]
                    ax.plot(ch["corners_ra"] + ch["corners_ra"][:1],
                            ch["corners_dec"] + ch["corners_dec"][:1],c='C0')
                    ax.text(np.mean(ch["corners_ra"] + ch["corners_ra"][:1]),np.mean(ch["corners_dec"] + ch["corners_dec"][:1]),'{}'.format(module),fontsize=10,color='C0',va='center',ha='center',alpha=0.3)
                except:
                    continue
            xlim,ylim=ax.get_xlim(),ax.get_ylim()

            for i,o in obj[np.asarray(obj.Campaign,dtype=float)==campaign].iterrows():
                if nepics[i]<=3:
                    continue
                if o.Name.startswith('GO'):
                    continue
                try:
                    with silence():
                        df=K2ephem.get_ephemeris_dataframe(o.Name, campaign, campaign, step_size=2)
                except:
                    continue

                ra,dec=df.ra,df.dec
                if fp["c{}".format(campaign)]['ra']-np.mean(ra)>300:
                    ra+=360.
                ok=np.where((ra>xlim[0])&(ra<xlim[1])&(dec>ylim[0])&(dec<ylim[1]))[0]
                ra,dec=ra[ok],dec[ok]
                p=ax.plot(ra,dec,lw=(24.-float(o.maxmag))/2)
                c=p[0].get_color()
                ax.text(np.median(ra),np.median(dec)+0.2,o.Name,color=c,zorder=99,fontsize=10)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.title('Campaign {}'.format(campaign),fontsize=20)
            plt.xlabel('RA')
            plt.ylabel('Dec')

            fig.savefig(img_dir+'campaign{}.png'.format(campaign),bbox_inches='tight',dpi=200)
            pbar.update()
            plt.close()
