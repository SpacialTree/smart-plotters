import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
from cmd_plot import Plotter
import regions
from regions import Regions
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC

basepath = '/orange/adamginsburg/jwst/cloudc/'

class DecapsCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.ra = catalog['ra']*u.deg
        self.dec = catalog['dec']*u.deg
        self.coords = SkyCoord(self.ra, self.dec)
        self.filters = ['u', 'g', 'r', 'i', 'z', 'y']

        # ugrizY

    def arg(self, band):
        return self.filters.index(band)
    
    def band_col(self, band, colname):
        return np.array(self.catalog[colname])[:, self.arg(band)]
    
    def flux(self, band):
        arg = self.filters.index(band)
        return np.array(self.catalog['mean'])[:, self.arg(band)]*3631*u.Jy

    def qf(self, band):
        arg = self.filters.index(band)
        return np.array(self.catalog['qf_avg'])[:, self.arg(band)]

    def band(self, band):
        flux = self.flux(band)/(3631*u.Jy)
        mag = -2.5*np.log10(flux)  # AB magnitude
        return mag

    def get_band_names(self):
        return self.filters

    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)

    def qf_mask(self, band, min=0.1):
        return self.qf(band) > min

    def qf_mask_full(self, mini=0.1):
        qf = np.array(self.catalog['qf_avg'])
        return np.any(qf > mini, axis=1)

def make_decaps_use(basepath='/orange/adamginsburg/jwst/cloudc/'):
    # Open catalog file
    cat_fn = f'{basepath}/catalogs/decam_flux_l0.5b0.5.fits'
    cat = Table.read(cat_fn)
    dcat = DecapsCatalog(cat)

    mask_qf = dcat.qf_mask_full()

    cat = cat[mask_qf]

    return DecapsCatalog(cat)


# decaps distances
# /orange/adamginsburg/cmz/decaps/decaps_galcen.fits