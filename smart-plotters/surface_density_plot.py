import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import regions
from regions import Regions
from astropy.coordinates import search_around_sky
from dust_extinction.averages import CT06_MWGC, I05_MWAvg, G21_MWAvg, CT06_MWLoc, RL85_MWGC, RRP89_MWGC, F11_MWGC
from scipy.spatial import KDTree
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm

def make_grid(shape):
    grid = np.empty(shape)
    grid.fill(np.nan)
    return grid

def get_pixcoords(cat, wcs):
    return np.array(wcs.all_world2pix(cat.coords.ra, cat.coords.dec, 0)).T

def make_kdtree(data, k=5):
    kdtr = KDTree(data)
    seps, inds = kdtr.query(data, k=[k])
    return seps, inds

def get_seps_arcsec(seps, pixel_scale):
    return seps * pixel_scale

def fill_grid(grid, data, value):
    data_rounded = np.floor(data).astype(int)
    for i in range(len(data_rounded)):
        x, y = data_rounded[i]
        grid[y, x] = value[i]
    return grid

def get_grid_mask(cat, shape, wcs, k=5):
    grid = make_grid(shape)

    data = get_pixcoords(cat, wcs)
    seps, inds = make_kdtree(data, k=k)

    grid = fill_grid(grid, data, np.ones(len(cat.catalog)))
    grid_bool = ~np.isnan(grid)
    return grid_bool

def interpolate_grid(grid, stdev):
    kernel = Gaussian2DKernel(stdev, x_size=12*stdev+1, y_size=12*stdev+1)
    grid = convolve_fft(grid, kernel, nan_treatment='interpolate')
    return grid

def stellar_separation(cat, shape, wcs, k=5):
    grid = make_grid(shape)
    pixel_scale = wcs.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data = np.array(wcs.all_world2pix(cat.ra, cat.dec, 0)).T
    kdtr = KDTree(data)

    grid_coords = np.indices(grid.T.shape)
    grid_coords = grid_coords.reshape(2, -1).T

    seps, inds = kdtr.query(grid_coords + 0.5, k=[k])

    grid[grid_coords[:, 1], grid_coords[:, 0]] = (seps[:, 0] * pixel_scale).value

    return grid

def stellar_density(cat, shape, wcs, k=5):
    grid = stellar_separation(cat, shape, wcs, k=5)
    return k/grid**2

def gutermuth_stellar_density(cat, shape, wcs, k=5):
    grid = stellar_separation(cat, shape, wcs, k=5)
    return (k-1)/(np.pi*grid**2)

def extinction_map(cat, Av, shape, wcs, fwhm=30, k=5, mask=None):
    grid = make_grid(shape)
    pixel_scale = wcs.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data = get_pixcoords(cat, wcs)
    seps, inds = make_kdtree(data, k=k)

    grid = fill_grid(grid, data, Av)

    if mask is not None:
        grid[mask] = np.nan

    grid = interpolate_grid(grid, fwhm)

    grid[grid < 0] = np.nan

    return grid