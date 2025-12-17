import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
#from cmd_plot import Plotter
from smart_plotters.cmd_plot import Plotter
import regions
from regions import Regions
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC
from astroquery.vizier import Vizier


basepath = '/orange/adamginsburg/jwst/cloudc/'

class VVVCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.coords = SkyCoord(self.catalog['RAJ2000'], self.catalog['DEJ2000'], unit=(u.deg, u.deg))
        self.ra = self.coords.ra
        self.dec = self.coords.dec

        suffix = '1ap1'
        boo = check_suffix(suffix)
        if not boo:
            suffix = 'mag'
            boo = check_suffix(suffix)
        if not boo:
            suffix = 'mag3'
            boo = check_suffix(suffix)
        if not boo:
            print(f'No valid suffix among -1ap1, -mag, -mag3 found in catalog columns: {self.catalog.colnames}')
            suffix = ''
            
        self.suffix = suffix

    def band(self, band): # J, H, Ks, Y, Z
        return self.catalog[f'{band}{self.suffix}']

    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)

def check_suffix(suffix):
    boo = False
    for colname in self.catalog.colnames:
        if colname.endswith(suffix): 
            boo = True
            break
    return boo

def make_vvv_cat(pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)), l=113.8*u.arcsec, w=3.3*u.arcmin):
    reg = regions.RectangleSkyRegion(pos, width=l, height=w)
    Vizier.ROW_LIMIT = 5e5
    cat_VVV = Vizier.query_region(coordinates=pos, width=l, height=w, catalog=['II/387/virac2'])[0]
    vvv_cat = VVVCatalog(cat_VVV)
    return vvv_cat