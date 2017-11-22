from contextlib import contextmanager
import warnings
import sys
import pandas as pd
import pickle
import os
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib import animation
import astropy.units as u
from tqdm import tqdm
from K2fov.K2onSilicon import onSiliconCheck,fields
@contextmanager
def silence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
TIME_FILE = '/Users/ch/K2/projects/k2movie/k2movie/data/campaign_times.txt'
WCS_DIR = '/Users/ch/K2/projects/asteroid/python/wcs/'


from scipy.ndimage.filters import gaussian_filter1d

import astropy.units as u
import K2ephem
import numpy as np
from astropy.time import Time
import dask.dataframe as dd
import dask.array as da
from photutils import CircularAperture, aperture_photometry

class Asteroid(object):
    '''asteroid object'''

    def __init__(self, name=None, campaign=None, channel=None, dir=None,lag=0.5):
        self.name = name
        self.campaign = campaign
        self.channel = channel
        self.dir = dir
        self.lag = lag
        self.times = pd.read_csv(TIME_FILE)
        self.LC = 29.4295*u.min
        campaign_str = 'c{}'.format(self.campaign)
        self.start_time = np.asarray(self.times.StartTime[self.times.Campaign==campaign_str])[0]+2454833.
        self.end_time = np.asarray(self.times.EndTime[self.times.Campaign==campaign_str])[0]+2454833.
        self.start_cad = np.asarray(self.times.StartCad[self.times.Campaign==campaign_str])[0]
        self.end_cad = np.asarray(self.times.EndCad[self.times.Campaign==campaign_str])[0]
        self.ncad = (self.end_cad-self.start_cad)+1
        self.time = (np.arange(self.ncad)*self.LC).to(u.day).value+self.start_time


        if name is None:
            print('Pass object name')
            return
        if self.campaign is None:
            print('Pass a campaign number')
            return
        self.find_ephem()
        self.find_silicon()

        self.ra,self.dec = np.interp(self.time,self.ephem.jd,self.ephem.ra)*u.deg,np.interp(self.time,self.ephem.jd,self.ephem.dec)*u.deg
        self.ra_lag,self.dec_lag = np.interp(self.time-lag,self.ephem.jd,self.ephem.ra)*u.deg,np.interp(self.time-lag,self.ephem.jd,self.ephem.dec)*u.deg

    def find_ephem(self):
        if self.name is None:
            print('Specify an asteroid name.')
        if self.campaign is None:
            print('Specify a campaign')
        with silence():
            df = K2ephem.get_ephemeris_dataframe(self.name,self.campaign,self.campaign,step_size=1./(4))
        times = [t[0:23] for t in np.asarray(df.index,dtype=str)]
        df['jd'] = Time(times,format='isot').jd
        df=df[['jd','ra','dec']]
        self.ephem = df

    def find_silicon(self):
        if not hasattr(self,'ephem'):
            self.find_ephem()
        ra,dec=np.asarray(self.ephem[['ra','dec']]).T
        k = fields.getKeplerFov(self.campaign)
        onsil=np.asarray(list(map(onSiliconCheck,list(ra),list(dec),np.repeat(k,1))))[0]
        if (onsil is False):
            print('Never on silicon?')
        else:
            channels = np.zeros(len(ra))
            for i,r,d in zip(range(len(ra)), ra, dec):
                channels[i] = k.getChannelColRow(r,d)[0]

            self.ephem['channels'] = channels.astype(int)
        if hasattr(self,'channel'):
            if (self.channel is None) is False:
                return
            else:
                self.channel = int(channels[0])
        else:
            self.channel = int(channels[0])


    def lightcurve(self, channel = 1, cadence = None, r=5, difference=True,plot=False,tol=30*u.arcsec,f_lag_lim=50):
        self.channelObj = Channel(self.campaign,channel,self.dir)

        print('Finding asteroid')
        self.channelObj.query_hdf5(self.ra,self.dec,tol)
        if cadence is None:
            cadence = np.arange(self.channelObj.ncad)
        track = self.channelObj.track(cadence)
        f = np.copy(self.channelObj.lightcurve(track,cadence=cadence,r=r,difference=difference,plot=plot))
        print('Finding background (lagged {} days)'.format(self.lag))
        self.channelObj.query_hdf5(self.ra_lag,self.dec_lag,tol)
        f_lag = np.copy(self.channelObj.lightcurve(track,cadence=cadence,r=r,difference=difference))
        ok = np.isfinite(f) & np.isfinite(f_lag) & (f!=0)
        x,y,z = cadence[ok], f[ok], f_lag[ok]

        ok = np.abs(z)<f_lag_lim
        ok = np.isclose(gaussian_filter1d(ok.astype(float),0.5),1,1E-6)
        lag = np.interp(self.lag,self.time-self.time[0],np.arange(len(time)))
        ok = np.interp(x+lag,x,ok).astype(bool)
        self.time = self.time[x[ok]]
        self.f = y[ok]

        return self.time, self.f



class Channel(object):
    '''Channel object. Holds all the data for a given channel. '''

    def find_data(self):
        if (self.dir is None) or (os.path.isdir(self.dir) is False):
            print('No data directory.')
            return
        if self.channel is None:
            print('No channel specified.')
            return

        fname=self.dir+'c{}/{}/0.h5'.format('{0:02}'.format(self.campaign),'{0:02}'.format(self.channel))
        if not os.path.isfile(fname):
            print('No such channel')
            return

        self.df = dd.read_hdf(fname,'table').reset_index(drop=True)
        self.cols = self.df.columns

    def update_tol(self,tol):
        if not hasattr(tol,'value'):
            self.tol = tol*float((4.*u.arcsec).to(u.deg).value)
            self.pixtol = int(tol)
        else:
            self.tol = float(tol.to(u.deg).value)
            self.pixtol = int(tol.to(u.arcsecond).value/4)
            #self.find_close()

    def __init__(self, campaign=None, channel=None, dir=None, tol=5):
        self.campaign = campaign
        self.channel = channel
        self.dir = dir
        self.find_data()
        self.times = pd.read_csv(TIME_FILE)
        self.LC = 29.4295*u.min
        campaign_str = 'c{}'.format(self.campaign)
        self.start_time = np.asarray(self.times.StartTime[self.times.Campaign==campaign_str])[0]+2454833.
        self.end_time = np.asarray(self.times.EndTime[self.times.Campaign==campaign_str])[0]+2454833.
        self.start_cad = np.asarray(self.times.StartCad[self.times.Campaign==campaign_str])[0]
        self.end_cad = np.asarray(self.times.EndCad[self.times.Campaign==campaign_str])[0]
        self.ncad = (self.end_cad-self.start_cad)+1
        self.time = (np.arange(self.ncad)*self.LC).to(u.day).value+self.start_time
        self.update_tol(tol)
        self.wcs_file = '{}{}'.format(WCS_DIR,'c{0:02}_'.format(self.campaign)+'{0:02}.p'.format(self.channel))
        if not os.path.isfile(self.wcs_file):
            print('No WCS found?')
        self.r = pickle.load(open(self.wcs_file,'rb'))

    def find_close(self, stol=0.04, ctol=5, npix=500):
        print('Finding overlaps for difference imaging.')
        '''Given a channel mosaic, find the cadences that are closest to overlapping with the given cadence. This makes the assumption that stars are on average, for the most part, constant.'''
        m = self.df[list(self.df.columns[4:])].astype(float).mean(axis=1, skipna=True).compute()
        #Ignore saturated stars
        m[m>5e4]=0
        self.top = list(np.argsort(m)[::-1][0:npix])
        df = self.df[list(self.df.columns[4:])]
        df_selected = df.map_partitions(lambda x: x[x.index.isin(self.top)])
        z = np.asarray(df_selected.astype(float).compute())
        z[z==0]=np.nan
        z[~np.isfinite(z)]=np.nan
        z /= np.atleast_2d(np.nanmedian(z,axis=1)).T
        z = z[(np.nansum(np.isfinite(z),axis=1)>0.9*np.shape(z)[1])]
        self.z = z
        nearby = {}
        for c in tqdm(np.arange(np.shape(z)[1])):
            with silence():
                ok = np.asarray(list(set(np.arange(np.shape(z)[1]))-set(np.arange(c-ctol,c+ctol))))
                s = np.nanstd(z[:,ok]-np.transpose(np.atleast_2d(z[:,c])),axis=0)
                sok = np.where(s < stol)[0]
                nearby[c] = sok
        self.nearby = nearby


    def query_hdf5(self, x=None, y=None, tol=None, difference=True):
        '''Go to the database and grab the data at a given point'''
        if (x is None) or (y is None):
            print('Specify a location.')
            return None
        if hasattr(self,'df') is False:
            print('No data.')
            return None
        if (tol is None) is False:
            self.update_tol(tol)

        typ = 'pixel'
        #Check for RA and Dec
        if hasattr(x,'value'):
            x = x.value
            y = y.value
            self.typ = 'deg'
            tol = self.tol
        else:
            self.typ = 'pix'
            tol = self.pixtol

        if not hasattr(self,'nearby'):
            self.find_close()

        if hasattr(x,'__iter__'):
            #Run through all cadences
            if len(x)!=self.ncad:
                print('Pass either single value or values for all cadences.')
                return
            if not hasattr(self,'nearby'):
                self.find_close()

            xar,yar = np.asarray(self.df['Y'].astype(float).compute()), np.asarray(self.df['X'].astype(float).compute())
            if self.typ == 'deg':
                xar,yar = self.r.wcs_pix2world(xar,yar,1)


            x1 = xar > np.min(x) - tol*2
            x2 = xar <= np.max(x) + tol*2
            y1 = yar > np.min(y) - tol*2
            y2 = yar <= np.max(y) + tol*2
            df = np.asarray(self.df.astype(float).compute())[x1 & x2 & y1 & y2,:]

            xar,yar = df[:,3], df[:,2]
            if self.typ == 'deg':
                xar,yar = self.r.wcs_pix2world(xar,yar,1)

            df = df[:,4:]
            ar = {}
            mod = {}
            x_store = {}
            y_store = {}
            for i, j, n, k in tqdm(zip(x, y, self.nearby.values(), range(len(x))),leave=True):
                with silence():
                    x1 = xar > i - tol
                    x2 = xar <= i + tol
                    y1 = yar > j - tol
                    y2 = yar <= j + tol
                    ok = x1 & x2 & y1 & y2
                    d = df[ok,:]
                    ar[k] = d[:,k]
                    x_store[k] = np.asarray(xar[ok])
                    y_store[k] = np.asarray(yar[ok])

                    if len(n)>3:
                        mod[k] = np.asarray(np.nanmedian(d[:,n],axis=1))
                    else:
                        mod[k]=None
            self.ar = ar
            self.mod = mod
            self.x = x_store
            self.y = y_store
            self.x_mid = x
            self.y_mid = y

        else:
            xar,yar = np.asarray(self.df['Y'].astype(float).compute()), np.asarray(self.df['X'].astype(float).compute())
            if self.typ == 'deg':
                xar,yar = self.r.wcs_pix2world(xar,yar,1)

            #Produce single cadence value
            x1 = xar > x - tol
            x2 = xar <= x + tol
            y1 = yar > y - tol
            y2 = yar <= y + tol
            d = np.asarray(self.df.astype(float).compute())[x1 & x2 & y1 & y2,:]
            d = d[:,4:]
            xar = xar[x1 & x2 & y1 & y2]
            yar = yar[x1 & x2 & y1 & y2]

            ar = {}
            mod = {}
            x_store = {}
            y_store = {}
            for n,k in tqdm(zip(self.nearby.values(), range(np.shape(d)[1])),leave=True):
                with silence():
                    ar[k] = d[:,k]
                    x_store[k] = np.asarray(xar)
                    y_store[k] = np.asarray(yar)
                    if len(n)>3:
                        mod[k] = np.asarray(np.nanmedian(d[:,n],axis=1))
                    else:
                        mod[k]=None

            self.ar = ar
            self.mod = mod
            self.x = x_store
            self.y = y_store
            self.x_mid = np.zeros(np.shape(d)[1])+x
            self.y_mid = np.zeros(np.shape(d)[1])+y

    def calc_color(self,data,scale='log'):
        '''Calculate a '''
        l = []
        for i in data.values():
            l.append(i)
        l = np.asarray([item for sublist in l for item in sublist])
        if scale=='log':
            y = np.log10(l)
        else:
            y = l
        y=y[np.isfinite(y)]
        if len(y)==0:
            print('No data')
            return None,None
        vmin=np.percentile(y,10)
        vmax=np.percentile(y,90)
        return vmin,vmax

    def reconstruct(self,data,t,scale='linear'):
        '''For a given time stamp, reconstruct a 2D image'''
        if (not hasattr(self,'rec')):
            if self.typ == 'deg':
                self.rec = np.zeros((self.pixtol*4, self.pixtol*4))*np.nan
            else:
                self.rec = np.zeros((self.pixtol*2+2, self.pixtol*2+2))*np.nan

        def recon(t):
            self.rec*=np.nan
            if self.typ == 'deg':
                try:
                    xcorr,ycorr = self.r.wcs_world2pix(self.x_mid[t],self.y_mid[t],1)
                    xvals,yvals = self.r.wcs_world2pix(self.x[t],self.y[t],1)
                except:
                    return self.rec
                xvals += self.pixtol*2 - xcorr
                yvals += self.pixtol*2 - ycorr
            else:
                xvals = (self.x[t]-self.x_mid[t]+self.pixtol)-1
                yvals = (self.y[t]-self.y_mid[t]+self.pixtol)-1
            if scale == 'log':
                self.rec[np.floor(xvals).astype(int),np.floor(yvals).astype(int)] = np.log10(data[t])
            else:
                self.rec[np.floor(xvals).astype(int),np.floor(yvals).astype(int)] = data[t]
            return self.rec

        return recon(t)

    def clean(self):
        if hasattr(self,'rec'):
            delattr(self,'rec')


    def track(self, cadence=None,fwhm=2.0, threshold=10, difference=True):
        '''Find where the source is in each stamp'''
        if (cadence is None):
            cadence = np.arange(self.ncad)
        xm, ym = np.zeros(len(cadence))*np.nan,np.zeros(len(cadence))*np.nan
        plotted = False
        for i,c in enumerate(cadence):
            if difference:
                im = np.copy(self.reconstruct(self.ar,c))-np.copy(self.reconstruct(self.mod,c))
            else:
                im = np.copy(self.reconstruct(self.ar,c))
            im[~np.isfinite(im)]=0
            if np.nansum(im)==0:
                continue
            try:
                xm[i],ym[i]=np.where(im==np.nanmax(im))
                continue
            except:
                continue
        loc = [np.nanmedian(xm),np.nanmedian(ym)]
        self.clean()
        return loc


    def lightcurve(self, track = None, cadence = None, r = 5, fwhm=2.0, threshold=10, difference=True, plot=False):
        '''Generate a light curve'''
        if (cadence is None):
            cadence = np.arange(self.ncad)
        if track is None:
            track = self.track(cadence,difference=difference,fwhm=fwhm, threshold=threshold)

        f = np.zeros(len(cadence))
        if plot:
            fig,ax=plt.subplots(1,5,figsize=(15,3),sharex=True,sharey=True)
            stops = [cadence[0],cadence[len(cadence)//4],cadence[len(cadence)//2],cadence[3*len(cadence)//4],cadence[-1]]

        for i,c in enumerate(cadence):
            if difference:
                im = np.copy(self.reconstruct(self.ar,c))-np.copy(self.reconstruct(self.mod,c))
            else:
                im = np.copy(self.reconstruct(self.ar,c))
            im[~np.isfinite(im)]=0
            if np.nansum(im)==0:
                continue

            apertures = CircularAperture(track, r=r)
            f[i] = aperture_photometry(im,apertures)[0]['aperture_sum']
            if plot:
                if c in stops:
                    s = np.where(stops == c)[0][0]
                    ax[s].imshow(im,vmin=0,vmax=threshold)
                    ax[s].set_xticks([])
                    ax[s].set_yticks([])
                    apertures.plot(ax=ax[s],color='C3')
        self.clean()
        return f


    def movie(self, plottype = 'data', outfile = 'out.mp4', cadence = None, cmap = 'viridis', colorbar=True, scale='log', vmin=None, vmax=None, text=True, title=None, frame_interval=30):
        '''Create a movie of whatever array is currently live'''
        data = self.ar

        if plottype == 'data':
            data = self.ar

        if plottype == 'model':
            data = self.mod

        if plottype == 'diff':
            data = np.copy(self.ar)
            for k in data.keys():
                data[k]-=self.mod[k]

        if plottype == 'baddiff':
            print("HELP ME")
            return
            data = self.ar-np.atleast_2d(np.nanmedian(self.ar,axis=1)).T

        if (cadence is None):
            cadence = np.arange(self.ncad)

        if hasattr(self,'ar') is False:
            print('No array specified. Query the databse first.')
            return None

        fig=plt.figure(figsize=(4,4))
        if colorbar==True:
            fig=plt.figure(figsize=(5,4))
        ax=fig.add_subplot(111)

        if vmin is None:
            vmin,vmax = self.calc_color(data,scale)

        cm = plt.get_cmap(cmap)
        cm.set_bad(cm(vmin),1)
        im=ax.imshow(self.reconstruct(data,cadence[0],scale),cmap=cm,vmin=vmin,vmax=vmax,origin='bottom',interpolation='none')
        if colorbar==True:
            cbar=fig.colorbar(im,ax=ax)
            cbar.ax.tick_params(labelsize=10)
            if scale=='log':
                cbar.set_label('log10(e$^-$s$^-1$)',fontsize=10)
            else:
                cbar.set_label('e$^-$s$^-1$',fontsize=10)

        if text:
            if (title is None)==False:
                text1=ax.text(0.1,0.9,title,fontsize=10,color='white',transform=ax.transAxes)
                text1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                       path_effects.Normal()])
            text2=ax.text(0.1,0.83,'Campaign {}'.format(self.campaign),fontsize=8,color='white',transform=ax.transAxes)
            text2.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
            text4=ax.text(0.1,0.78,'Channel {}'.format(self.channel),fontsize=8,color='white',transform=ax.transAxes)
            text4.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
            text3=ax.text(0.1,0.72,'Time (BJD): {}'.format(int(self.time[cadence[0]])),fontsize=8,color='white',transform=ax.transAxes)
            text3.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])


        def animate(i):
            im.set_array(self.reconstruct(data,cadence[i],scale))
            if text:
                text3.set_text('{}'.format(int(self.time[cadence[i]])))
                return im,text3,
            return im,

        anim = animation.FuncAnimation(fig,animate,frames=len(cadence), interval=frame_interval, blit=True)
        anim.save(outfile, dpi=150)
        self.clean()
