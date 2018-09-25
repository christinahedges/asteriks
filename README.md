# asteriks 0.5
[![Build Status](https://travis-ci.org/christinahedges/asteriks.svg?branch=master)](https://travis-ci.org/christinahedges/asteriks)

`asteriks` is an open source python to work with Kepler/K2 data and generate light curves of solar system objects.

## Installation

You can in install the most up to the minute version of `asteriks` by executing the following commands in a terminal...

```
  git clone https://github.com/christinahedges/asteriks.git
  cd asteriks
  python setup.py
```

Our most recent stable release is available on PyPI which you can install with

```
  pip install asteriks
```

Note if you want to run this and generate movies with this code you will need to install ffmpeg:

```
    sudo apt-get install ffmpeg
```


## File Formats

`asteriks` provides two final products; a fits file with a light curve and a Target Pixel File of the object moving across the Kepler focal plane. Below is a description of the fits file.

### .fits files

There are several extensions in the fits files, where each extension is a different **aperture**. The first extension is the "optimum" aperture, as decided by the `asteriks` pipeline. Other extensions have apertures of varying sizes, enabling the user to use visual inspection to determine the best aperture for their science.

##### Header

The primary header of the file contains the origin (asteriks) and object name (use `OBJECT` to find the name. The exposure start and end time are also in the header. The keyword `LDLGCORR` is short for "Lead Lag Correction". If True, then the Lead Lag Correction has been applied to the object, providing more quality flags. (See further documentation for a full description of the Lead Lag Correction.) Finally, the asteriks version number is included in the primary header.

#### First Extension

The first extension contains the optimum aperture, as decided by the pipeline. The following columns are included:

* `TIME`: The time in JD of each observation
* `FLUX`: The flux of the object as observed by Kepler
* `FLUX_ERR`: The error in the flux of the object as observed by Kepler
* `RA_OBJ`: The Right Ascention of the object at each exposure, as determined by JPL Small Bodies Database
* `DEC_OBJ`: The Declination of the object at each exposure, as determined by JPL Small Bodies Database
* `LEAD_QUAL`: Short for 'Lead Quality'. First quality flag for the flux of the object. This is the quality of the Lead Lag Correction. If False, the Lead Lag Correction could not be successfully applied. In such cases, the user may choose to mask out bad quality data.
* `NPIX_QUAL`: Short for 'Number of Pixels Quality'. Second quality flag for flux of the object. This flag is True if all pixels of the aperture have real values (i.e. are not NaNs). Where e.g. an asteroid moves off the edge of the detector NPIX_QUAL will be False.
* `BKG_QUAL`: Short for 'Background Quality'. Third quality flag for flux of the object. This flag is True if the pipeline determines the background to be static and corrected. The flag is False if the pipeline determines that there was variable background that was poorly corrected, such as from an object moving over a saturated star or through a halo of a bright star.
* `NPIX_APER`: Number of pixels inside the aperture at each point in time
* `EXTNAME`: Name of the extension. If extension one, is `BESTAPER`
* `PERC` : Percentile cut used to create the aperture
* `NPIX` : Number of pixels in the aperture.
* `LEADFLAG` : Whether Lead Lag Correction was used.
