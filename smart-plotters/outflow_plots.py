import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import regions
from regions import Regions
from spectral_cube import SpectralCube

default_fn = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw27.cube.I.iter1.image.pbcor.fits'
default_restfreq = 86.84696*u.GHz # ACES SiO 2-1

class OutflowPlot:
    """ 
    Class for quickly plotting outflows from astronomical data cubes.
    """
    def __init__(self, position=None, l=5*u.arcsec, w=5*u.arcsec, restfreq=None, cube_fn=default_fn, reg=None):
        """ 
        Parameters
        ----------
        position : astropy.coordinates.SkyCoord
            Center of the region to extract from the cube.
        l : astropy.units.Quantity, default=5*u.arcsec
            Length of the region to extract from the cube.
        w : astropy.units.Quantity, default=5*u.arcsec
            Width of the region to extract from the cube.
        restfreq : astropy.units.Quantity, optional
            Rest frequency of the spectral cube. If not provided, the rest frequency will be read from the FITS header.
        cube_fn : str, optional
            File path to the FITS file containing the spectral cube.
        """

        self.position = position
        self.l = l
        self.w = w

        if position is None and reg is None:
            raise ValueError("Either position or reg must be provided.")

        self.cube_fn = cube_fn

        if restfreq is not None:
            self.restfreq = restfreq
        elif cube_fn == default_fn:
            self.restfreq = default_restfreq
        else:
            header = fits.getheader(self.cube_fn)
            try:
                self.restfreq = header['RESTFRQ'] * u.Hz
            except KeyError:
                raise ValueError("RESTFRQ keyword not found in FITS header and no restfreq provided.")
                
        if reg is not None:
            self.reg = reg
        else:
            self.reg = regions.RectangleSkyRegion(position, self.l, self.w)

    def open_cube(self):
        """ 
        Open the spectral cube file and return the cube object.
        """
        cube = SpectralCube.read(self.cube_fn).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=self.restfreq)
        return cube

    def get_subcube(self):
        """
        Extract the subcube defined by the region.
        """
        cube = self.open_cube()
        if isinstance(self.reg, list) or isinstance(self.reg, Regions):
            subcube = cube.subcube_from_regions(self.reg)
        else:
            subcube = cube.subcube_from_regions([self.reg])
        return subcube

    def get_spectral_slab(self, vmin, vmax):
        """ 
        Extract a spectral slab from the cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity
            Maximum velocity of the slab.
        """
        subcube = self.get_subcube()
        slab = subcube.spectral_slab(vmin, vmax)
        return slab

    def get_moment0(self, vmin=None, vmax=None):
        """ 
        Calculate the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        """
        if vmin is not None and vmax is not None:
            slab = self.get_spectral_slab(vmin, vmax)
            return slab.moment0()
        elif vmin is not None or vmax is not None:
            raise ValueError("Both vmin and vmax must be provided.")
        else:
            subcube = self.get_subcube()
            return subcube.moment0()

    def plot_moment0(self, vmin=None, vmax=None, ax=None, **kwargs):
        """ 
        Plot the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        """
        moment0 = self.get_moment0(vmin, vmax)
        if ax is None:
            ax = plt.subplot(projection=moment0.wcs)
        ax.imshow(moment0.value, **kwargs)

    def make_levels(self, data, level_type='percentages', nlevels=5):
        """ 
        Generate contour levels for the moment 0 map.

        Parameters
        ----------
        data : numpy.ndarray
            Data array of the moment 0 map.
        level_type : str
            Type of contour levels to generate. Options are 'start-step-multiplier', 'min-max-scaling', 'percentages', 'mean-sigma-list'.
        nlevels : int, optional
            Number of contour levels to generate.
        """
        return make_levels(data, level_type=level_type, nlevels=nlevels)
        
    def plot_moment0_contours(self, vmin=None, vmax=None, levels=None, ax=None, nlevels=5, **kwargs):
        """ 
        Plot contours of the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        levels : list, str, optional
            List of contour levels to plot or type of levels to generate.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        """
        moment0 = self.get_moment0(vmin, vmax)
        plot_proj_contours(moment0, vmin, vmax, levels, ax, nlevels, **kwargs)

    def plot_outflows(self, vcen=0*u.km/u.s, vmin=-10*u.km/u.s, vmax=10*u.km/u.s, levels=None, 
                      ax=None, blue_color='blue', red_color='red', nlevels=5, **kwargs):
        """
        Plot blue and red outflows on the moment 0 map.

        Parameters
        ----------
        vcen : astropy.units.Quantity, default=0*u.km/u.s
            Center velocity of the outflow.
        vmin : astropy.units.Quantity, default=-10*u.km/u.s
            Minimum velocity of the blueshifted outflow.
        vmax : astropy.units.Quantity, default=10*u.km/u.s
            Maximum velocity of the redshifted outflow.
        levels : list, optional
            List of contour levels to plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        blue_color : str, default='blue'
            Color of the blueshifted outflow contours.
        red_color : str, default='red'
            Color of the redshifted outflow contours.
        """
        if ax is None:
            ax = plt.subplot(projection=self.get_moment0().wcs)
        # Plot redshifted outflow
        self.plot_moment0_contours(vmin=vcen, vmax=vmax, levels=levels, ax=ax, colors=red_color, nlevels=nlevels, **kwargs)
        # Plot blueshifted outflow
        self.plot_moment0_contours(vmin=vmin, vmax=vcen, levels=levels, ax=ax, colors=blue_color, nlevels=nlevels, **kwargs)

def make_levels(data, level_type='percentages', nlevels=5):
    """ 
    Generate contour levels for the moment 0 map.

    Parameters
    ----------
    data : numpy.ndarray
        Data array of the moment 0 map.
    level_type : str
        Type of contour levels to generate. Options are 'start-step-multiplier', 'min-max-scaling', 'percentages', 'mean-sigma-list'.
    nlevels : int, optional
        Number of contour levels to generate.
    """
    if level_type == 'start-step-multiplier' or level_type == 'start-step':
        levels = start_step_multiplier(data, nlevels=nlevels)
    elif level_type == 'min-max-scaling' or level_type == 'min-max':
        levels = min_max_scaling(data, nlevels=nlevels)
    elif level_type == 'percentages' or level_type == 'percentile':
        levels = percentages(data, nlevels=nlevels)
    elif level_type == 'mean-sigma-list' or level_type == 'mean-sigma':
        levels = mean_sigma_list(data, nlevels=nlevels)
    else:
        raise ValueError("Invalid level_type. Options are 'start-step-multiplier', 'min-max-scaling', 'percentages', 'mean-sigma-list'.")
    return levels

def plot_proj_contours(mom0, vmin=None, vmax=None, levels=None, ax=None, nlevels=5, **kwargs):
    """ 
    Plot contours of the moment 0 map of the spectral cube.

    Parameters
    ----------
    mom0 : Projection
        Moment 0 map of the spectral cube.
    vmin : astropy.units.Quantity, optional
        Minimum velocity of the slab.
    vmax : astropy.units.Quantity, optional
        Maximum velocity of the slab.
    levels : list, str, optional
        List of contour levels to plot or type of levels to generate.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot the moment 0 map on.
    """
    if ax is None:
        ax = plt.subplot()
    if levels is None:
        levels = make_levels(mom0.data, level_type='percentages', nlevels=nlevels)
        ax.contour(mom0.data, levels=levels, transform=ax.get_transform(mom0.wcs), **kwargs)
    elif isinstance(levels, list):
        ax.contour(mom0.data, levels=levels, transform=ax.get_transform(mom0.wcs), **kwargs)
    elif isinstance(levels, str):
        levels = make_levels(mom0.data, level_type=levels, nlevels=nlevels)
        ax.contour(mom0.data, levels=levels, transform=ax.get_transform(mom0.wcs), **kwargs)
    else:
        ax.contour(mom0.data, levels=levels, transform=ax.get_transform(mom0.wcs), **kwargs)

### Level generation functions, default from Carta ###

def start_step_multiplier(data, nlevels=5, start=None, step=None, multiplier=None):
    if start is None:
        start = np.nanmean(data) + 1.7*np.nanstd(data)
    if step is None:
        step = 1.5*np.nanstd(data)
    if multiplier is None:
        multiplier = 1
    levels = [start]
    for i in range(nlevels-1):
        levels.append(levels[-1] + step)
        step *= multiplier
    return levels

def min_max_scaling(data, nlevels=5, min=None, max=None, scaling='linear'):
    if min is None:
        min = np.nanpercentile(data, 90)
    if max is None:
        max = np.nanpercentile(data, 99.9)
    if scaling == 'linear':
        levels = np.linspace(min, max, nlevels)
    elif scaling == 'log':
        levels = np.logspace(np.log10(min), np.log10(max), nlevels)
    return levels

def percentages(data, nlevels=5, reference=None, lower=30, upper=100):
    if reference is None:
        reference = np.nanpercentile(data, 99.9)
    levels = np.linspace(lower, upper, nlevels)/100 * reference
    return levels

def mean_sigma_list(data, nlevels=5, mean=None, sigma=None, sigma_list=[3, 5, 7]):
    if mean is None:
        mean = np.nanmean(data)
    if sigma is None:
        sigma = np.nanstd(data)
    levels = [mean + sig*sigma for sig in sigma_list]
    return levels


def quickplot_SiO(position, l=5*u.arcsec, w=5*u.arcsec, reg=None):
    """ 
    Quickly plot outflows from the SiO 2-1 line in the ACES data cube.
    """
    cube_fn = default_fn
    restfreq = default_restfreq
    op = OutflowPlot(position, l=l, w=w, reg=reg, cube_fn=cube_fn, restfreq=restfreq)
    return op

def get_ACES_info(line, basepath='/orange/adamginsburg/jwst/cloudc/alma/ACES/', 
                  table_fn='/orange/adamginsburg/jwst/cloudc/analysis/linelist.csv'):
    spec_tab = Table.read(spec_tab)
    mol = spec_tab[spec_tab['Line']==line]
    restfreq = mol['Rest (GHz)'].data[0]*u.GHz
    spw = mol['12m SPW'].data[0]
    cube_fn = f'{basepath}/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw{spw}.cube.I.iter1.image.pbcor.fits'
    return restfreq, cube_fn

def quickplot_ACES(line, position, l=5*u.arcsec, w=5*u.arcsec, reg=None):
    """ 
    Quickly plot outflows from a line in the ACES data cube.
    """
    try: 
        restfreq, cube_fn = get_ACES_info(line)
    except:
        raise ValueError(f"Line '{line}' not found in ACES linelist.csv.")

    op = OutflowPlot(position, l=l, w=w, reg=reg, cube_fn=cube_fn, restfreq=restfreq)
    return op
