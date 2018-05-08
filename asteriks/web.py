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


def create_index_html(name):
    fname = "{}{}.html".format(OUTPUT_DIR, name)
    header = '{}'.format(name)
    size1 = 2
    size2 = 10
    campaign = 6
    start_date = 'February 2018'
    end_date = 'May 2018'
    intro_string = ("Asteroid {0} is a moving object from campaign {1} of K2. "
                    " Data was taken of {0} from {2} to {3}"
                    "".format(name, campaign, start_date, end_date))
    urls = ['http://example.com/1', 'http://example.com/2', 'http://example.com/3']
    context = {
        'header': header,
        'urls': urls,
        'size1': size1,
        'size2': size2,
        'intro_string': intro_string
    }
    #
    with open(fname, 'w') as f:
        html = render_template('template.html', context)
        f.write(html)


def main():
    create_index_html('Christina')

########################################


if __name__ == "__main__":
    main()
