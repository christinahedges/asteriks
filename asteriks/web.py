#!/usr/bin/env python
from flask import Flask, render_template
from . import PACKAGEDIR
import os
import shutil
from jinja2 import Environment, FileSystemLoader
from glob import glob
from .query import *
import pickle
from astropy.time import Time
from datetime import datetime
from .version import __version__


TEMPLATE_DIR = os.path.join(PACKAGEDIR, 'data/')
OUTPUT_DIR = os.path.join('/'.join(PACKAGEDIR.split('/')[0:-1]), 'docs/', 'pages/')
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(TEMPLATE_DIR, 'templates')),
    trim_blocks=False)

index_link = '../../index.html'
search_link = '../../search.html'


citation = """<code><pre>@ARTICLE{asteriks,
               author = {{Hedges}, C. and Co},
                title = "{}",
              journal = {},
            archivePrefix = "arXiv",
               eprint = {},
             primaryClass = "",
             keywords = {},
                 year = ,
                month = ,
               volume = ,
                pages = {},
                  doi = {},
               adsurl = {},
            }</pre></code>
        """

acknowledgement = '''<code><pre>
                    Acknowledgement:
                    This work uses...</pre></code>'''


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_asteroid_page_html(name, campaign, start_date=0, end_date=0):

    fname = "{0}{1}/{1}.html".format(OUTPUT_DIR, name.replace(' ', ''))
    header = '{}'.format(name)
    other_names = find_alternate_names_using_CAF(name)
    ids = find_GO_proposal(name)
    PIs = find_PIs(name)
    extra_citations = '<code><pre>' +\
                      '\n\n'.join([get_bibtex(id) for id in ids.split('|')]) +\
                      '</pre></code>'

    if isinstance(other_names, str):
        other_names = [other_names]
    if isinstance(other_names, (pd.Series, np.ndarray, list)):
        other_names = list(set(other_names) - set([name]))
    aka = ''
    if len(other_names) != 0:
        aka = ' (a.k.a. {})'.format(', '.join(other_names))
    jpllink = "https://ssd.jpl.nasa.gov/sbdb.cgi?sstr={};old=0;orb=1;cov=0;log=0;cad=0#orb".format(
        name.replace(' ', '%20'))
    fitsfile = glob('{0}{1}/*lightcurve*v{2}.fits'.format(OUTPUT_DIR, name.replace(' ', ''), __version__))[0]
    tpffitsfile = glob('{0}{1}/*tpf*v{2}.fits'.format(OUTPUT_DIR, name.replace(' ', ''), __version__))[0]
    img = '{}_lc.png'.format(name.replace(' ', ''))
    mp4 = '{}.mp4'.format(name.replace(' ', ''))

    size1 = '{:.2}'.format(os.path.getsize(fitsfile)/1e6)
    size2 = '{}'.format(int(np.round(os.path.getsize(tpffitsfile)/1e6)))
    intro_string = ("<br>{0}{4} is a moving object from K2 campaign {1}. "
                    "You can read more information about this object at the <b>JPL Small-Body Database Browser</b> <a href={5}>here</a>. "
                    " Data was taken from {2} to {3}. "

                    "".format(name, campaign, start_date, end_date, aka, jpllink))

    if len(PIs) != 0:
        intro_string += ("<br></br>{0} was proposed for by <b>{1}</b> in {2}. "
                         "If you use this data, please cite their proposal. "
                         "You can find the bibtex citation by clicking the button below."
                         "".format(name, ', '.join(PIs.split('|')), ', '.join(ids.split('|'))))
    else:
        intro_string += "<br></br> You can find the bibtex citation for this object by clicking the button below."

    context = {
        'header': header,
        'size1': size1,
        'size2': size2,
        'download1': fitsfile.split('/')[-1],
        'download2': tpffitsfile.split('/')[-1],
        'mp4': mp4,
        'intro_string': intro_string,
        'img': img,
        'citation': citation + extra_citations + acknowledgement
    }
    with open(fname, 'w') as f:
        html = render_template('template.html', context)
        f.write(html)


def create_search_page_html():
    pagenames = glob(OUTPUT_DIR+"*/*.html")
    names = np.asarray([p.split('/')[-2] for p in pagenames])
    for idx, name in enumerate(names):
        if (str(name[0:4]).isdigit()) and (not str(name[4:]).isdigit()):
            names[idx] = '{} {}'.format(name[0:4], name[4:])

    PIs = np.asarray(['(PI: {})'.format(', '.join(find_PIs(name).split('|'))) for name in names])
    PIs[np.where(PIs == '(PI: )')[0]] = ''
    links = ['pages/{}'.format('/'.join(name.split('/')[-2:])) for name in pagenames]
    context = {
        'index_link': index_link,
        'names_links': zip(names, links, PIs)
    }
    html = render_template('search.html', context)
    with open('{}/docs/search.html'.format('/'.join(PACKAGEDIR.split('/')[:-1])), 'w') as f:
        f.write(html)
