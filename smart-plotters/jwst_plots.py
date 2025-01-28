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

class JWSTCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.coords = self.catalog['skycoord_ref']
        self.ra = self.coords.ra
        self.dec = self.coords.dec

    def band(self, band):
        return self.catalog[f'mag_ab_{band.lower()}']

    def get_band_names(self):
        return [colname[-5:] for colname in self.catalog.colnames if colname.startswith('qfit_')]

    def color(self, band1, band2):
        return self.catalog[f'mag_ab_{band1.lower()}'] - self.catalog[f'mag_ab_{band2.lower()}']

    def flux(self, band):
        return self.catalog[f'flux_jy_{band.lower()}']

    def eflux(self, band):
        return self.catalog[f'eflux_jy_{band.lower()}']

    def plot_band_histogram(self, band, min=None, max=None, num=50, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if min is None:
            min = np.min(self.band(band))
        if max is None:
            max = np.max(self.band(band))
        h, bins = np.histogram(self.band(band), range=(min, max), bins=num)
        ax.step(bins[:-1], h, where='mid', **kwargs)
        #ax.hist(self.band(band), **kwargs)
        ax.set_xlabel(f'{band.upper()} Magnitude')
        ax.set_ylabel('Star Count')
        return ax

    def get_Av_182410(self, ext=CT06_MWGC()):
        av182410 = (self.color('f182m', 'f410m')) / (ext(1.82*u.um) - ext(4.10*u.um))
        return av182410

    def get_Av_212410(self, ext=CT06_MWGC()):
        av212410 = (self.color('f212n', 'f410m')) / (ext(2.12*u.um) - ext(4.10*u.um))
        return av212410
    
    def get_Av(self, band1, band2, ext=CT06_MWGC()):
        return (self.color(band1, band2)) / (ext(int(band1[1:-1])/100*u.um) - ext(int(band2[1:-1])/100*u.um))

    def get_qf_mask(self, qf=0.4):
        #mas_405 = np.logical_or(np.array(self.catalog['qfit_f405n'])<qf, np.isnan(np.array(self.catalog['mag_ab_f405n'])))
        #mas_410 = np.logical_or(np.array(self.catalog['qfit_f410m'])<qf, np.isnan(np.array(self.catalog['mag_ab_f410m'])))
        #mask = np.logical_and(mas_405, mas_410)
        #mas_466 = np.logical_or(np.array(self.catalog['qfit_f466n'])<qf, np.isnan(np.array(self.catalog['mag_ab_f466n'])))
        #mask = np.logical_and(mask, mas_466)
        #mas_187 = np.logical_or(np.array(self.catalog['qfit_f187n'])<qf, np.isnan(np.array(self.catalog['mag_ab_f187n'])))
        #mask = np.logical_and(mask, mas_187)
        #mas_182 = np.logical_or(np.array(self.catalog['qfit_f182m'])<qf, np.isnan(np.array(self.catalog['mag_ab_f182m'])))
        #mask = np.logical_and(mask, mas_182)
        #mas_212 = np.logical_or(np.array(self.catalog['qfit_f212n'])<qf, np.isnan(np.array(self.catalog['mag_ab_f212n'])))
        #mask = np.logical_and(mask, mas_212)
        bands = self.get_band_names()
        #[colname[-5:] for colname in self.catalog.colnames if colname.startswith(f'qfit_')]
        #mask = np.array([np.logical_or(np.array(self.catalog[f'qfit_{band}']) < qf, np.isnan(np.array(self.catalog[f'mag_ab_{band}']))) for band in bands])
        mask = np.logical_or.reduce([np.logical_or(np.array(self.catalog[f'qfit_{band}']) < qf, np.isnan(np.array(self.catalog[f'mag_ab_{band}']))) for band in bands])

        return mask
    
    def table_qf_mask(self):
        mask = self.get_qf_mask()
        return self.catalog[mask]

    def get_brights_mask(self):
        mask_187 = np.logical_and(self.catalog['mag_ab_f187n'] < 15, self.color('F187N', 'F182M') < -0.3)
        mask = np.logical_or(~mask_187, np.isnan(np.array(self.catalog['mag_ab_f187n'])))
        mask_405 = np.logical_and(self.catalog['mag_ab_f405n'] < 13.5, self.color('F405N', 'F410M') < -0.3)
        mask &= np.logical_or(~mask_405, np.isnan(np.array(self.catalog['mag_ab_f405n'])))
        return mask

    def get_count_mask(self):
        mask = np.array([self.catalog[colname] < 0.1 for colname in self.catalog.colnames if colname.startswith('emag')])
        mask = mask.max(axis=0)
        return mask

    def apply_mask(self, mask):
        return self.catalog[mask]

    def get_multi_detection_mask(self):
        # Mask for detection in more than one filter
        combine_mask = np.zeros(len(self.catalog), dtype=int)
        for band in self.get_band_names():
            combine_mask += ~np.isnan(self.catalog[f'mag_ab_{band}'])

        return combine_mask > 1


def make_cat_use(basepath = '/orange/adamginsburg/jwst/cloudc/'):
    # Open catalog file
    cat_fn = f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits'
    basetable = Table.read(cat_fn)

    # Create JWSTCatalog object
    base_jwstcatalog = JWSTCatalog(basetable)

    # Mask for quality factor
    mask_qf = base_jwstcatalog.get_qf_mask(0.4)

    # Mask for count
    mask_count = base_jwstcatalog.get_count_mask()

    # Mask for bad bright stars
    mask_brights = base_jwstcatalog.get_brights_mask()

    # Mask for detections in more than one band
    mask_multi = base_jwstcatalog.get_multi_detection_mask()

    # Combine Masks
    mask = np.logical_and(mask_qf, mask_count)
    mask = np.logical_and(mask, mask_brights)
    mask = np.logical_and(mask, mask_multi)

    # Return catalog with quality factor mask
    cat_use = JWSTCatalog(basetable[mask])
    return cat_use

def make_cat_raw(basepath='/orange/adamginsburg/jwst/cloudc/'):
    cat_fn = f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits'
    basetable = Table.read(cat_fn)
    base_jwstcatalog = JWSTCatalog(basetable)
    return base_jwstcatalog

def make_brick_cat():
    #basepath = '/orange/adamginsburg/jwst/brick/'
    cat_fn = '/orange/adamginsburg/jwst/brick/catalogs/basic_merged_indivexp_photometry_tables_merged_qualcuts_oksep2221.fits'
    print(f"Reading {cat_fn}")
    basetable = Table.read(cat_fn)

    # Create JWSTCatalog object
    base_jwstcatalog = JWSTCatalog(basetable)

    # Mask for quality factor
    mask_qf = base_jwstcatalog.get_qf_mask(0.4)

    # Mask for count
    mask_count = base_jwstcatalog.get_count_mask()

    # Mask for bad bright stars
    mask_brights = base_jwstcatalog.get_brights_mask()

    # Mask for detections in more than one band
    mask_multi = base_jwstcatalog.get_multi_detection_mask()

    # Combine Masks
    mask = np.logical_and(mask_qf, mask_count)
    mask = np.logical_and(mask, mask_brights)
    mask = np.logical_and(mask, mask_multi)

    # Return catalog with quality factor mask
    cat_use = JWSTCatalog(basetable[mask])
    return cat_use