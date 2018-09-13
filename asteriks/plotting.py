import numpy as np

from matplotlib import animation
from matplotlib.colors import Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import K2ephem
from astropy.coordinates import SkyCoord
import astropy.units as u

from .utils import *
from . import PACKAGEDIR
from . import query

import matplotlib.pyplot as plt
import K2fov


def campaign_base_plot(ax=None, campaigns=[2], alpha=[0.3]):
    if ax is None:
        _, ax = plt.subplots(1, figsize=(6, 6))
        print(ax)
    for jdx, campaign in enumerate(campaigns):
        fov = K2fov.getKeplerFov(campaign)
        corners = fov.getCoordsOfChannelCorners()
        for ch in np.arange(1, 85, dtype=int):
            if ch in fov.brokenChannels:
                continue  # certain channel are no longer used
            idx = np.where(corners[::, 2] == ch)
            mdl = int(corners[idx, 0][0][0])
            out = int(corners[idx, 1][0][0])
            ra = corners[idx, 3][0]
            # if np.any(ra > 340):  # Engineering test field overlaps meridian
            #    ra[ra > 180] -= 360
            dec = corners[idx, 4][0]
            ax.fill(np.concatenate((ra, ra[:1])),
                    np.concatenate((dec, dec[:1])),
                    lw=1, edgecolor='#AAAAAA',
                    facecolor='#AAAAAA', zorder=0, alpha=np.asarray(alpha)[jdx])

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    return xlims, ylims

def stack_array(dat, n=3):
    '''Bin an array down.
    '''
    ar = np.copy(dat)
    ar[~np.isfinite(ar)] = 0
    length = len(ar)
    while not ((length / n) == length // n):
        length -= 1
    stacked = ar[0:length][0::n, :, :]
    for i in np.arange(1, n):
        stacked += ar[0:length][i::n, :, :]
    ar[ar == 0] = np.nan
    return stacked


def two_panel_movie(dat, dif, title='', out='out.mp4', scale='linear', stack=1, **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data = np.log10(stack_array(np.copy(dat), stack))
        diff = np.log10(stack_array(np.copy(dif), stack))
    else:
        data = stack_array(np.copy(dat), stack)
        diff = stack_array(np.copy(dif), stack)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax in axs:
        ax.set_facecolor('#ecf0f1')
    im1 = axs[0].imshow(data[0], origin='bottom', **kwargs)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('Asteroid Centered Flux', fontsize=10)
#    cbar1 = fig.colorbar(im1, ax=axs[0])
#    cbar1.ax.tick_params(labelsize=10)
    im2 = axs[1].imshow(diff[0], origin='bottom', ** kwargs)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('Motion Differenced Flux', fontsize=10)
#    cbar2 = fig.colorbar(im2, ax=axs[1])
#    cbar2.ax.tick_params(labelsize=10)
#    if scale == 'log':
#        cbar1.set_label('log10(e$^-$s$^-1$)', fontsize=10)
#        cbar2.set_label('log10(e$^-$s$^-1$)', fontsize=10)
#    else:
#        cbar1.set_label('Flux [e$^-$/s]', fontsize=12)
#        cbar2.set_label('Flux [e$^-$/s]', fontsize=12)

    def animate(i):
        im1.set_array(data[i])
        im2.set_array(diff[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30 * stack**0.5)
    anim.save(out, dpi=150)


def movie(dat, title='', out='out.mp4', scale='linear', facecolor='red', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data = np.log10(np.copy(dat))
    else:
        data = dat
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.set_facecolor(facecolor)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanpercentile(data, 75)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(data, 5)
    im1 = ax.imshow(data[0], origin='bottom', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15)
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.ax.tick_params(labelsize=10)
    if scale == 'log':
        cbar1.set_label('log10(e$^-$s$^-1$)', fontsize=10)
    else:
        cbar1.set_label('e$^-$s$^-1$', fontsize=10)

    def animate(i):
        im1.set_array(data[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)


def plot_aperture_movie(obj, name, campaign, lagspacing, aperture_radius, dir='', frameskip=5):
    with plt.style.context(('ggplot')):
        norm = Normalize(vmin=0, vmax=np.max(lagspacing))
        cmap = plt.get_cmap('RdYlGn_r')
        circ_radius = (4 * u.arcsec * aperture_radius).to(u.deg).value
        ra = obj[0][obj[0].incampaign == True].ra
        dec = obj[0][obj[0].incampaign == True].dec
        aspect_ratio = (np.nanmax(dec)-np.nanmin(dec))/(np.nanmax(ra)-np.nanmin(ra))
        if aspect_ratio < 1:
            fig, ax = plt.subplots(1, figsize=(10, 10*aspect_ratio))
        if aspect_ratio > 1:
            fig, ax = plt.subplots(1, figsize=(10*aspect_ratio, 10))
        plt.subplots_adjust(wspace=1, hspace=1)
        xlims, ylims = campaign_base_plot(ax=ax, campaigns=[campaign])

        plt.plot(obj[0][obj[0].incampaign == True].ra, obj[0]
                 [obj[0].incampaign == True].dec, zorder=-1, lw=1, color='grey')
        plt.xlim(obj[0][obj[0].incampaign == True].ra.min(),
                 obj[0][obj[0].incampaign == True].ra.max())
        plt.ylim(obj[0][obj[0].incampaign == True].dec.min(),
                 obj[0][obj[0].incampaign == True].dec.max())
        plt.gca().set_aspect(1)
        plt.xlabel('RA', fontsize=20)
        plt.ylabel('Declination', fontsize=20)
        plt.title('{} Aperture Masks'.format(name))

        ra = np.zeros((len(obj), len(obj[0][obj[0].onsil == True])))
        dec = np.zeros((len(obj), len(obj[0][obj[0].onsil == True])))
        for idx, o in enumerate(obj):
            ra[idx, :], dec[idx, :] = np.asarray(
                o[o.onsil == True]['ra']), np.asarray(o[o.onsil == True]['dec'])
        ra, dec = ra.T, dec.T
        naper = len(ra[0])
        patches = [plt.Circle((ra[0][j], dec[0][j]), circ_radius, fc=cmap(
            norm(np.abs(lagspacing[j])))) for j in range(naper)]

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
        log.debug('Generating animation')
        frames = np.arange(0,
                           len(obj[0][obj[0].onsil == True]),
                           frameskip)
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=frames,
                                       interval=20,
                                       blit=True)
        log.debug('Saving animation')
        anim.save('{}{}_aperture.mp4'.format(dir, name.replace(' ', '')), dpi=150)


def plot_all_asteroid_tracks(img_dir='', campaigns=None):
    '''Plots all the asteroid tracks that asteriks knows about.
    '''
    if campaigns is None:
        campaigns = np.arange(20)
    if not hasattr(campaigns, '__iter__'):
        campaigns = [campaigns]

    with plt.style.context(('ggplot')):
        for campaign in tqdm(campaigns):
            o = query.find_moving_objects_in_campaign(campaign)
            fig, ax = plt.subplots(figsize=(10, 10))
            xlim, ylim = campaign_base_plot(ax=ax, campaigns=[campaign], alpha=[0.3])
            plt.grid(False)
            for name in o.NAMES:
                with silence():
                    try:
                        df = K2ephem.get_ephemeris_dataframe(
                            name[0], campaign, campaign, step_size=1/8)
                    except K2ephem.EphemFailure:
                        continue
                ra, dec = df.ra, df.dec
                ok = np.where((ra > xlim[0]) & (ra < xlim[1]) &
                              (dec > ylim[0]) & (dec < ylim[1]))[0]
                ra, dec = ra[ok], dec[ok]
                p = ax.plot(ra, dec)
                c = p[0].get_color()
                if np.isfinite(np.nanmedian(ra) * np.nanmedian(dec)):
                    ax.text(np.nanmedian(ra), np.nanmedian(dec)+0.2,
                            name[0], color=c, zorder=99, fontsize=10)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.title('Campaign {}'.format(campaign), fontsize=20)
            plt.xlabel('RA')
            plt.ylabel('Dec')

            fig.savefig(img_dir+'campaign{}.png'.format(campaign), bbox_inches='tight', dpi=200)
            plt.close()


def plot_all_CAF_files(img_dir='', campaigns=None):
    '''Plots all the asteroid CAF files that asteriks knows about.
    '''
    if campaigns is None:
        campaigns = np.arange(20)
    if not hasattr(campaigns, '__iter__'):
        campaigns = [campaigns]

    with plt.style.context(('ggplot')):
        for campaign in tqdm(campaigns):
            o = query.find_moving_objects_in_campaign(campaign)
            fig, ax = plt.subplots(figsize=(10, 10))
            xlim, ylim = campaign_base_plot(ax=ax, campaigns=[campaign], alpha=[0.3])
            plt.grid(False)
            for name in o.NAMES:
                df = query.find_mast_files_using_CAF(name[0])
                coord = SkyCoord(df.RA, df.Dec, unit=(u.hourangle, u.deg))
                ra, dec = coord.ra.deg, coord.dec.deg
                p = ax.plot(ra, dec, ls='', markersize=3, marker='.')
                c = p[0].get_color()
                if (np.nanmedian(ra) > xlim[0]) & (np.nanmedian(ra) < xlim[1]) &\
                        (np.nanmedian(dec) > ylim[0]) & (np.nanmedian(dec) < ylim[1]):
                    ax.text(np.nanmedian(ra), np.nanmedian(dec)+0.2,
                            name[0], color=c, zorder=99, fontsize=10)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.title('Campaign {}'.format(campaign), fontsize=20)
            plt.xlabel('RA')
            plt.ylabel('Dec')

            fig.savefig(img_dir+'campaign{}_CAF.png'.format(campaign), bbox_inches='tight', dpi=200)
            plt.close()
