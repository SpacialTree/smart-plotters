import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.coordinates import SkyCoord
import regions
from astropy.table import Table
from regions import Regions
from scipy.spatial import KDTree
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.nddata import Cutout2D

from spectral_cube import SpectralCube
import importlib as imp

from dust_extinction.averages import CT06_MWLoc, I05_MWAvg, CT06_MWGC, G21_MWAvg, RL85_MWGC, RRP89_MWGC, F11_MWGC

import icemodels
imp.reload(icemodels)
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.gaussian_model_components import co_ice_wls_icm, co_ice_wls, co_ice_widths, co_ice_bandstrength
from icemodels.core import optical_constants_cache_dir, read_ocdb_file, download_all_ocdb, composition_to_molweight, read_lida_file

from astroquery.svo_fps import SvoFps
from astropy import table

import smart_plotters.surface_density_plot as utils

from brick2221.analysis.analysis_setup import basepath, molscomps

def compute_molecular_column(unextincted_466m410, dmag_tbl, icemol='CO', ref_band='f405n'):
    dmags466 = dmag_tbl['F466N']
    dmags410 = dmag_tbl[ref_band.upper()]

    comp = "H2O:CO:CO2 (10:1:1)"#np.unique(dmag_tbl['composition'])[0]
    molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)

    cols = dmag_tbl['column'] * mol_frac #molwt * mol_massfrac / (mol_wt_tgtmol)

    dmag_466m410 = np.array(dmags466) - np.array(dmags410) 
    inferred_molecular_column = np.interp(unextincted_466m410, dmag_466m410[cols<1e21], cols[cols<1e21])

    return inferred_molecular_column

def get_dmag_tbl():
    from brick2221.analysis.analysis_setup import basepath
    dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
    dmag_tbl.add_index('composition')
    dmag_tbl.add_index('mol_id')
    
    return dmag_tbl

def unextinct(cat, ext, band1, band2, Av):
    EV_band2_band1 = (ext(int(band2[1:-1])/100*u.um) - ext(int(band1[1:-1])/100*u.um))
    return cat.color(band1, band2) + EV_band2_band1 * Av

def get_co_ice_column(cat, av, ext=CT06_MWLoc(), ref_band='f405n'):
    unextincted_466mref = unextinct(cat, ext=ext, band1='f466n', band2=ref_band, Av=av)
    dmag_tbl = get_dmag_tbl()
    dmag_tbl_sel = dmag_tbl.loc['H2O:CO:CO2 (10:1:1)']

    return compute_molecular_column(unextincted_466mref, dmag_tbl_sel, icemol='CO', ref_band=ref_band)

def co_ice_modeling(ref_band='f405n', consts_file='1_CO_(1)_12.5K_Baratta.txt'):
    
    filter_data = SvoFps.get_filter_list('JWST', instrument="NIRCam")
    filter_data.add_index('filterID')
    flxd = filter_data['filterID']
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    filterid = 'JWST/NIRCam.F466N'
    xarr = np.linspace(filter_data.loc[filterid]['WavelengthMin'] * u.AA,
                    filter_data.loc[filterid]['WavelengthMax'] * u.AA,
                    1000)
    xarr = np.linspace(3.95*u.um, 4.8*u.um, 5000)
    phx4000 = atmo_model(4000, xarr=xarr)

    trans = SvoFps.get_transmission_data(filterid)

    # loading CO molecule
    # molecule = 'co'
    # CO molecule constants 
        #load_molecule_ocdb(molecule) # OCDB = optical constants database
    try: 
        consts = baratta_co = read_ocdb_file(f'{optical_constants_cache_dir}/{consts_file}')
    except: 
        try: 
            consts = read_lida_file(f'{optical_constants_cache_dir}/{consts_file}')
        except:
            consts = Table.read(f'{optical_constants_cache_dir}/{consts_file}', format='ascii')
    # Get the mean molecular weight of the ice compound
    molwt = composition_to_molweight(consts.meta['composition'])#*u.Da
    # phx4000 = stellar atmosphere model spectrum at 4000K
    xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
    # column densities of CO ice
    cols = np.geomspace(1e15, 1e22, 25)
    dmags410 = []
    dmags466 = []

    #print(f"  column,   mag410,  mag410*,  mag466n, mag466n*, dmag410, dmag466")
    for col in cols:
        # for each column density of CO (ice?), make a spectrum of it 
        # absorbed_spectrum takes spectrum and puts the effects of an absorption feature in front of it 
        spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molwt,
                                spectrum=phx4000['fnu'].quantity, # flux array
                                xarr=xarr, # wavelength array
                                )
        cmd_x = (f'JWST/NIRCam.{ref_band.upper()}', 'JWST/NIRCam.F466N')
        flxd_ref = fluxes_in_filters(xarr, phx4000['fnu'].quantity)
        flxd = fluxes_in_filters(xarr, spec)
        # the star's magnitude
        mags_x_star = (-2.5*np.log10(flxd_ref[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                    -2.5*np.log10(flxd_ref[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)))
        # the magnitude of the star with the CO ice
        #mags_x = flxd[cmd_x[0]].to(u.ABmag).value, flxd[cmd_x[1]].to(u.ABmag).value
        mags_x = (-2.5*np.log10(flxd[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                -2.5*np.log10(flxd[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)))
        # the difference in magnitudes in F410M and F466N
        dmags466.append(mags_x[1]-mags_x_star[1])
        dmags410.append(mags_x[0]-mags_x_star[0])
        # why would f410m change at all?

    dmag_466m410 = np.array(dmags466) - np.array(dmags410) 
    return dmag_466m410, cols


def get_co_column_old(cat, Av, ext=CT06_MWLoc(), ref_band='f405n', consts_file='1_CO_(1)_12.5K_Baratta.txt'):
    #if ref_band not in ['f410m', 'f405n']:
    #    raise ValueError(f"ref_band must be either 'f410m' or 'f405n', not {ref_band}")

    dmag_466m410, cols = co_ice_modeling(ref_band, consts_file)
    unextincted_color = unextinct(cat, ext=ext, band1='f466n', band2=ref_band, Av=Av) #  testing out revering the ref band and f446n
    #unextincted_color[unextincted_color>0] = np.nan

    co_col = np.interp(unextincted_color, dmag_466m410[cols<1e21], cols[cols<1e21])
    return co_col

def list_consts_files(wild='', verbose=True):
    import os
    from glob import glob
    if verbose:
        for file in glob(f'{optical_constants_cache_dir}/*{wild}*.txt'):
            print(file.split('/')[-1])
    return glob(f'{optical_constants_cache_dir}/*{wild}*.txt')

def make_co_column_map(cat, co_col, wcs, shape, fwhm=30, k=5):
    grid = utils.make_grid(shape)
    pixel_scale = wcs.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data = utils.get_pixcoords(cat, wcs)
    seps, inds = utils.make_kdtree(data, k=k)

    grid = utils.fill_grid(grid, data, co_col)

    grid = utils.interpolate_grid(grid, fwhm)

    return grid

def plot_Av_COice(Av, co_col, extras=False, ax=None, Av_offset=0, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.scatter(Av, co_col, **kwargs)
    ax.set_xlim(-5, 95)
    ax.set_ylim(2e15, 5e20)
    ax.set_yscale('log')
    ax.set_xlabel(f"A$_V$")
    ax.set_ylabel("N(CO)")
    l1 = plt.legend()

    if extras:
        NCOofAV = 1.1e21 * np.linspace(0.1, 100, 1000) * 1e-4

        # by-eye fit Filament
        #x1,y1 = 22,1e17
        #x2,y2 = 50,7e18
        #m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
        #b = np.log10(y1 / 10**(m * x1))
        #ax.plot([x1, x2], 10**np.array([x1*m+b, x2*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Filament]')
        plt.ylim(1e17, 4e19)
        pt1 = (20, 3e17)
        pt2 = (37, 4e18)
        m = (np.log10(pt2[1]) - np.log10(pt1[1])) / (pt2[0] - pt1[0])
        b = np.log10(pt1[1] / 10**(m * pt1[0]))
        ax.plot([pt1[0], pt2[0]], 10**np.array([pt1[0]*m+b, pt2[0]*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Filament]')
        #plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], label='Filament', color='k', linestyle='--')

        # by-eye fit Brick
        x1,y1 = 10,2e17
        x2,y2 = 43,1e19
        x1,y1 = 33,8e16
        x2,y2 = 80,3e19
        m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
        b = np.log10(y1 / 10**(m * x1))
        ax.plot([x1, x2], 10**np.array([x1*m+b, x2*m+b]), 'b--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Ginsburg 2023 Brick]')

        # BGW 2015
        ax.plot([7, 23], [0.5e17, 7e17], 'g', label='log N = 0.07 A$_V$ + 16.2 [BGW 2015]')

        # 100% of CO in ice if N(H2)=2.2e21 A_V
        ax.plot(np.linspace(0.1, 100, 1000)+Av_offset, NCOofAV,
            label='100% of CO in ice if N(H$_2$)=1.1$\\times10^{21}$ A$_V$', color='k', linestyle=':')
        ax.legend()

def plot_ice_CCD(cat, ax=None, bins=100, threshold=5, cmap='autumn_r', color='k', s=1):
    import mpl_plot_templates as template
    x = np.array(cat.color('f182m', 'f212n'))
    x[x>5] = np.nan
    y = np.array(cat.color('f410m', 'f466n'))
    y[y>5] = np.nan
    if ax is None:
        ax = plt.subplot()
    
    template.adaptive_param_plot(x, y, threshold=threshold, bins=bins, cmap=cmap, marker_color=color, markersize=s, axis=ax)
    plt.xlabel('F182M - F212N')
    plt.ylabel('F410M - F466N');