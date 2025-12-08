from cmd_plot import Plotter
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

class Flexible(Plotter):
    def __init__(self, catalog):
        self.catalog = catalog
        pass

    def set_coords(self, coords):
        self.coords = coords
        self.ra = self.coords.ra
        self.dec = self.coords.dec

    def set_band_format(self, band_prefix='', band_suffix='mag'):
        self.band_prefix = band_prefix
        self.band_suffix = band_suffix
        self.band_format = f'{self.band_prefix}{{}}_{self.band_suffix}'

    def band(self, band):
        return self.catalog[self.band_format.format(band)]

    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)

    def get_Av(self, band1, band2, wave1, wave2, ext):
        return (self.color(band1, band2)) / (ext(wave1) - ext(wave2))