import asteriks as ak
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import os
from scipy.ndimage.filters import gaussian_filter1d

obj = pd.read_pickle('/Users/ch/K2/projects/moving_objects/out.p')
obj = obj[obj.Campaign=='8']

for i,o in obj.iterrows():
    if o.minmag=='0':
        continue

    a = ak.Asteroid(o.Name, campaign=int(o.Campaign), dir='/Volumes/cupertino/database/')
    channels = np.unique(a.ephem.channels)
    for c in channels:
        if os.path.isfile('/Users/ch/K2/projects/asteroid/output/lcs/{}_{}.csv'.format(a.name.replace(' ',''),c)):
            continue
        print('NAME: {}\n CHANNEL: {}\n'.format(o.Name,c))
        a.lightcurve(c,r=5, val_lim=100)

        t, f = a.t, a.f
        if t is None:
            pd.DataFrame(np.asarray([[0],[0]]).T,columns=["Time","Counts"]).to_csv('/Users/ch/K2/projects/asteroid/output/lcs/{}_{}.csv'.format(a.name.replace(' ',''),c),index=False)
        else:
            ok = a.ok
            fig,ax=plt.subplots(1,figsize=(10,3))
            ax.scatter(t[ok],f[ok],s=5,c='black',zorder=1)

            h=np.histogram(f[ok],np.arange(-300,300,3))
            g = models.Gaussian1D(amplitude=h[0].max(), mean=0, stddev=50)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g, np.arange(np.min(f[ok]),np.max(f[ok]),5)[0:-1]-2.5,h[0])
            lim = 6*g.stddev.value
            ylim = [g.mean.value-lim,g.mean.value+lim]
            ax.plot(t,f,zorder=0,alpha=0.5,ls='--',c='C3')
            ax.set_ylim(ylim)
            plt.xlabel('Time (BJD)',fontsize=15)
            plt.ylabel('Counts ($e^{-}s^{-1}$)',fontsize=15)
            plt.title('{} (Channel:{} Campaign:{})'.format(a.name,c,a.campaign),fontsize=15)
            fig.savefig('/Users/ch/K2/projects/asteroid/output/images/{}_{}.png'.format(a.name.replace(' ',''),c),dpi=200,bbox_inches='tight')
            pd.DataFrame(np.asarray([t[ok],f[ok]]).T,columns=["Time","Counts"]).to_csv('/Users/ch/K2/projects/asteroid/output/lcs/{}_{}.csv'.format(a.name.replace(' ',''),c),index=False)
