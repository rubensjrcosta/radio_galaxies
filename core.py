import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

from astropy.table import Table, Column
from astropy.units import  Quantity
from astropy.visualization import quantity_support

from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.datasets import Datasets
from gammapy.estimators import FluxPoints
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.modeling.models import Models

from feupy.utils.units import Jy_to_erg_by_cm2_s

def load_source_model(datasets_path = None):
    """Load the `SkyModel` already prepared for the CTA simulation"""
    return Models.read(f'{datasets_path}/models.yaml').select(name_substring="model-fit")[0].copy()

def irfs_label_txt(irfs_opt):
    site = irfs_opt[0].replace("South", "S").replace("North", "N").replace("-SSTSubArray", "SST").replace("-MSTSubArray", "-MST").replace("-LSTSubArray", "-LST")
    average = irfs_opt[1].replace('AverageAz', 'Az').replace("NorthAz", "Nz").replace("SouthAz", "Sz")
    irfs_label = f"{site}-{average}-{irfs_opt[2]}-{irfs_opt[3]}"
    return  irfs_label

def set_sens_label(label, livetime_out=None, zenith_out=None):
    label = label.rsplit('(')[0]
    label = label.replace("sens", "CTAO")
    label = label.replace("N-Az", "North").replace("S-Az", "South")
    label = label.replace('SSST-Az', 'South-SSTs').replace('S-MST-Az', 'South-MSTs') 
    label = label.replace('N-MST-Az', 'North-MSTs').replace('N-LST-Az', 'North-LSTs') 
    if livetime_out is not None:
        label = label.replace(f"-{livetime_out}", "")
    if zenith_out is not None:
        label = label.replace(f"-{zenith_out}", "")  
    return label

def plot_sensitivity_table(table, wich, ax=None, **kwargs):
    
    ax = plt.gca() if ax is None else ax
    
    e_cta, ef_cta = table['e_ref'], table[wich]
    e_bin_min, e_bin_max = table['e_min'], table['e_max']
    xerr = u.Quantity([e_cta-e_bin_min, e_bin_max-e_cta], "TeV")
    with quantity_support():
        ax.errorbar(e_cta, ef_cta, xerr=xerr, **kwargs)

    ax.legend(loc= 'upper right', handlelength=4)
    ax.set_xlabel(f"Energy [{e_cta.unit.to_string(UNIT_STRING_FORMAT)}]", size=12)
    
    if wich == 'excess':
        ax.set_ylabel(f"Excess counts", size=12)
    elif wich == 'background':
        ax.set_ylabel(f"Background counts", size=12)
    elif wich == 'on_radii':
        ax.set_ylabel(f"On region radius [{ef_cta.unit.to_string(UNIT_STRING_FORMAT)}]", size=12)
    elif wich == 'e2dnde':
        ax.set_ylabel(f"Flux Sensitivity [{ef_cta.unit.to_string(UNIT_STRING_FORMAT)}]", size=12)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    return ax

def get_data_VizieR_byname(table=None, target=None):

    myArr = np.empty(len(table['Units']), dtype='U20')

    table.meta['catalog'] = 'http://ned.ipac.caltech.edu/byname'
    table['label'] = myArr
    table['label'].description = 'label of the legend'
    data_freq = table['Frequency']
    
    data = Quantity(data_freq, 'Hz').to('TeV', equivalencies=u.spectral())
    
    table['e_ref'] = Column(
        data=data,
        description='Energy of reference', unit=data[0].unit, format='.3e',
    )
    data = Jy_to_erg_by_cm2_s(table['Frequency'], table['Flux Density'])
    table['e2dnde'] = Column(
        data=data,
        description='Differential flux', unit=data[0].unit, format='.3e',
    )

    table['Frequency'].unit = 'Hz'
    table['Flux Density'].unit = 'Jy'
    table['Flux density uncertainty']  = table['Flux Density']

    table_copy = table[['Observed Passband', 'e_ref', 'e2dnde', 'Refcode']].copy()
    df = table_copy.to_pandas()
    df = df.drop_duplicates(subset=['e_ref'], keep='first')
    df_ordenado = df.sort_values(by='e_ref')
    df_ = dict(tuple(df_ordenado.groupby('Refcode')))
    datasets = Datasets()
    for name in list(df['Refcode'].unique()):
        tab = Table.from_pandas(df_[name])
        tab['e_ref'].unit = "TeV"
        tab['e2dnde'].unit = "erg s-1 cm-2"
        fp = FluxPoints.from_table(tab[['e_ref', 'e2dnde']], sed_type="e2dnde")
        ds_fp = FluxPointsDataset(data=fp, name=name)
        datasets.append(ds_fp)
    return datasets