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

class GaiaCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.ra = self.catalog['RA_ICRS']
        self.dec = self.catalog['DE_ICRS']

    def band(self, band): # G or BP or RP
        return self.catalog[f'{band.upper()}mag']

    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)