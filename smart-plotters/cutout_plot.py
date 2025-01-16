import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from spectral_cube import SpectralCube
import regions
from regions import Regions
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

class Cutout:
    def __init__(self, position, l, w):
        self.position = position
        self.l = l
        self.w = w

    def get_cutout(self, filename, format='fits'):
        if format == 'fits':
            try: 
                hdu = fits.open(filename, ext='SCI')[0]
            except: 
                hdu = fits.open(filename)[0]
        elif format == 'casa':
            hdu = SpectralCube.read(filename, format='casa').hdu
        data = np.squeeze(hdu.data)
        head = hdu.header

        ww = WCS(head).celestial
        size = (self.l, self.w)
        cutout = Cutout2D(data, position=self.position, size=size, wcs=ww)
        return cutout

    def blind_cutout(self, filename, position):
        try:
            cutout = self.get_cutout(filename, position)
        except: 
            cutout = self.get_cutout(filename, position, format='casa')
        return cutout

    def get_cutout_region(self, frame='icrs'):
        if frame == 'galactic':
            return regions.RectangleSkyRegion(center=self.position.galactic, width=self.l, height=self.w)
        elif frame == 'icrs':
            return regions.RectangleSkyRegion(center=self.position.icrs, width=self.l, height=self.w)
        else:
            raise ValueError('frame must be either "icrs" or "galactic"')

    def get_cutout_rgb(self, red_fn, green_fn, blue_fn, format='fits', rmax=90, gmax=210, bmax=120):
        red_cutout = self.blind_cutout(red_fn, position, l, w)
        green_cutout = self.blind_cutout(green_fn, position, l, w)
        blue_cutout = self.blind_cutout(blue_fn, position, l, w)

        ww = red_cutout.wcs

        if red_cutout.data.shape != green_cutout.data.shape or red_cutout.data.shape != blue_cutout.data.shape:
            hdu_red = fits.PrimaryHDU(data=red_cutout.data, header=red_cutout.wcs.to_header())
            hdu_green = fits.PrimaryHDU(data=green_cutout.data, header=green_cutout.wcs.to_header())
            hdu_blue = fits.PrimaryHDU(data=blue_cutout.data, header=blue_cutout.wcs.to_header())

            ww, shape = find_optimal_celestial_wcs([hdu_red, hdu_green, hdu_blue])
            red_reproj, _ = reproject_interp(hdu_red, ww, shape)
            red_cutout = fits.PrimaryHDU(data=red_reproj, header=ww.to_header())
            green_reproj, _ = reproject_interp(hdu_green, ww, shape)
            green_cutout = fits.PrimaryHDU(data=green_reproj, header=ww.to_header())
            blue_reproj, _ = reproject_interp(hdu_blue, ww, shape)
            blue_cutout = fits.PrimaryHDU(data=blue_reproj, header=ww.to_header())
        
        rgb = np.array([
            red_cutout.data,
            green_cutout.data,
            blue_cutout.data
        ]).swapaxes(0,2).swapaxes(0,1)

        rgb_scaled = np.array([
                simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=rmax)(rgb[:,:,0]),
                simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=gmax)(rgb[:,:,1]),
                simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=bmax)(rgb[:,:,2]),
            ]).swapaxes(0,2)

        return rgb_scaled, ww

def get_cutout_405(position, l, w, filename='/orange/adamginsburg/jwst/cloudc/images/F405_reproj_merged-fortricolor.fits'):
    cutout = Cutout(position, l, w)
    return cutout.get_cutout(filename)

def get_cutout_jwst(position, l, w, band, basepath='/orange/adamginsburg/jwst/cloudc/images/'):
    cutout = Cutout(position, l, w)
    if band[0] == 'f':
        band = band[1:]
    if band[-1] == 'm' or band[-1] == 'n':
        band = band[:-1]
    filename = f'{basepath}/F{band}_reproj_merged-fortricolor.fits'
    return cutout.get_cutout(filename)

def get_cutout_spitzer(position, l, w, band, basepath='/orange/adamginsburg/cmz/glimpse_data/'):
    cutout = Cutout(position, l, w)
    filename = f'{basepath}/GLM_00000+0000_mosaic_{band}.fits'
    return cutout.get_cutout(filename)

def get_cutout_jwst_ice(position, l, w, band, basepath='/orange/adamginsburg/jwst/cloudc/images/'):
    cutout_R = get_cutout_jwst(position, l, w, 'f466n', basepath)
    cutout_B = get_cutout_jwst(position, l, w, 'f405n', basepath)
    cutout_G = cutout_R.data + cutout_B.data

    rgb = np.array(
        [
            cutout_R.data,
            cutout_G,
            cutout_B.data
        ]
    ).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=210)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs

def get_cutout_jwst_rgb(position, l, w, basepath='/orange/adamginsburg/jwst/cloudc/images/'):
    fn_red = f'{basepath}/F410_reproj_merged-fortricolor.fits'
    fn_green = f'{basepath}/F212_reproj_merged-fortricolor.fits'
    fn_blue = f'{basepath}/F182_reproj_merged-fortricolor.fits'

    cutout = Cutout(position, l, w)
    return cutout.get_cutout_rgb(fn_red, fn_green, fn_blue)