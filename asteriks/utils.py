'''Common functions for asteriks
'''

import os
from contextlib import contextmanager
import warnings
import sys
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
import astropy.units as u
import matplotlib.pyplot as plt
import logging
import K2fov

log = logging.getLogger('asteriks')
loggingLevels = {'CRITICAL' : 50,
                 'ERROR': 40,
                 'WARNING': 30,
                 'INFO': 20,
                 'DEBUG': 10}

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

setLevel('INFO')

def movie(dat, title='', out='out.mp4', scale='linear', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data = np.log10(np.copy(dat))
    else:
        data = dat
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    im=ax.imshow(data[0], origin='bottom', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    if scale == 'log':
        cbar.set_label('log10(e$^-$s$^-1$)',fontsize=10)
    else:
        cbar.set_label('e$^-$s$^-1$',fontsize=10)
    def animate(i):
        im.set_array(data[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)

def make_aperture_plot(obj, name, campaign, lagspacing, aperture_radius, dir=''):
    with plt.style.context(('ggplot')):
        norm = Normalize(vmin = 0, vmax = np.max(lagspacing))
        cmap = plt.get_cmap('RdYlGn_r')
        circ_radius = (4 * u.arcsec * aperture_radius).to(u.deg).value
        fig, ax = plt.subplots(figsize=(10,10))
        fov = K2fov.getKeplerFov(campaign)
        corners = fov.getCoordsOfChannelCorners()
        for ch in np.arange(1, 85, dtype=int):
            if ch in fov.brokenChannels:
                continue  # certain channel are no longer used
            idx = np.where(corners[::, 2] == ch)
            mdl = int(corners[idx, 0][0][0])
            out = int(corners[idx, 1][0][0])
            ra = corners[idx, 3][0]
            #if np.any(ra > 340):  # Engineering test field overlaps meridian
            #    ra[ra > 180] -= 360
            dec = corners[idx, 4][0]
            ax.fill(np.concatenate((ra, ra[:1])),
                         np.concatenate((dec, dec[:1])),
                         lw=1, edgecolor='#AAAAAA',
                         facecolor='#AAAAAA', zorder=0, alpha=0.5)

        plt.plot(obj[0].ra, obj[0].dec, zorder=-1, lw=1, color='grey')
        plt.xlim(obj[0].ra.min(), obj[0].ra.max())
        plt.ylim(obj[0].dec.min(), obj[0].dec.max())
        plt.gca().set_aspect(1)
        plt.xlabel('RA', fontsize=20)
        plt.ylabel('Declination', fontsize=20)
        plt.title('{} Aperture Masks'.format(name))

        ra = np.zeros((len(obj), len(obj[0][obj[0].onsil == True])))
        dec = np.zeros((len(obj), len(obj[0][obj[0].onsil == True])))
        for idx, o in enumerate(obj):
            ra[idx, :], dec[idx, :] = np.asarray(o[o.onsil == True]['ra']), np.asarray(o[o.onsil == True]['dec'])
        ra, dec = ra.T, dec.T
        naper = len(ra[0])
        patches = [plt.Circle((ra[0][j], dec[0][j]), circ_radius, fc=cmap(norm(np.abs(lagspacing[j])))) for j in range(naper)]
        def init():
            for j, patch in enumerate(patches):
                patch.center = (ra[0][j], dec[0][j])
            return patches
        def animate(i):
            for j, r, d, patch in zip(range(len(ra[i])), ra[i], dec[i], patches):
                patch.center = (r, d)
                patch.fc = cmap(norm(np.abs(lagspacing[j])))
                ax.add_artist(patch)
            return patches
        log.info('Generating animation')
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=np.arange(len(ra)),
                                       interval=20,
                                       blit=True)
        log.info('Saving animation')
        anim.save('{}{}_aperture.mp4'.format(dir, name.replace(' ','')), dpi=150)

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
