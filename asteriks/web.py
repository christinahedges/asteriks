#!/usr/bin/env python
from flask import Flask, render_template
from . import PACKAGEDIR
import os
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = os.path.join(PACKAGEDIR, 'data/')
OUTPUT_DIR = os.path.join('/'.join(PACKAGEDIR.split('/')[0:-1]), 'website', 'pages/')
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(TEMPLATE_DIR, 'templates')),
    trim_blocks=False)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_index_html(name, dir):
    fname = "{}{}.html".format(OUTPUT_DIR, name.replace(' ', ''))
    header = '{}'.format(name)
    other_names = ['Christina']
    aka = ''
    if len(other_names) != 0:
        aka = ' (a.k.a. {})'.format(', '.join(other_names))
    jpllink = "https://ssd.jpl.nasa.gov/sbdb.cgi?sstr={};old=0;orb=1;cov=0;log=0;cad=0#orb".format(
        name.replace(' ', '%20'))
    img = '{0}{1}/{1}_lc.png'.format(dir, name.replace(' ', ''))
    mp4 = '{0}{1}/{1}.mp4'.format(dir, name.replace(' ', ''))
    size1 = 2
    size2 = 10
    size3 = 20
    campaign = 6
    start_date = 'February 2018'
    end_date = 'May 2018'
    citation = """<code><pre>@ARTICLE{2018MNRAS.476.2968H,
                   author = {{Hedges}, C. and {Hodgkin}, S. and {Kennedy}, G.},
                    title = "{Discovery of new dipper stars with K2: a window into the inner disc region of T Tauri stars}",
                  journal = {\mnras},
                archivePrefix = "arXiv",
                   eprint = {1802.00409},
                 primaryClass = "astro-ph.SR",
                 keywords = {methods: data analysis, techniques: photometric, stars: variables: T Tauri, Herbig Ae/Be},
                     year = 2018,
                    month = may,
                   volume = 476,
                    pages = {2968-2998},
                      doi = {10.1093/mnras/sty328},
                   adsurl = {http://adsabs.harvard.edu/abs/2018MNRAS.476.2968H},
                  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
                }</pre></code>
            """
    intro_string = ("<br>{0}{4} is a moving object from K2 campaign {1}. "
                    "You can read more information about this object at the <b>JPL Small-Body Database Browser</b> <a href={5}>here</a>. "
                    " Data was taken from {2} to {3}. "
                    "<br></br>You can download the light curve, target pixel file and vizualisations "
                    " of {0} using the links below."
                    "".format(name, campaign, start_date, end_date, aka, jpllink))
    urls = ['http://example.com/1', 'http://example.com/2', 'http://example.com/3']
    context = {
        'header': header,
        'urls': urls,
        'size1': size1,
        'size2': size2,
        'size3': size3,
        'mp4': mp4,
        'intro_string': intro_string,
        'img': img,
        'citation': citation
    }
    #
    with open(fname, 'w') as f:
        html = render_template('template.html', context)
        f.write(html)


def main(name, dir='/Users/ch/K2/projects/hlsp-asteriks/output/'):
    create_index_html(name, dir)

########################################


if __name__ == "__main__":
    main()
