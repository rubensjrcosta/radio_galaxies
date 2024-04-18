

    
from feupy.catalog import CATALOG_REGISTRY

from feupy.utils.string_handling import name_to_txt


from feupy.analysis.config_sml import SimulationConfig

import numpy as np
import astropy.units as u

from feupy.utils.stats import StatisticalUtilityFunctions as stats
from feupy.plotters.config import *

import matplotlib.pyplot as plt

from gammapy.datasets import Datasets, FluxPointsDataset

from gammapy.estimators import FluxPoints

from astropy.units import  Quantity
from astropy.table import Table
from gammapy.maps.axes import UNIT_STRING_FORMAT
from astropy.table import Column

path_blazars_tables = f"/home/born-again/Documents/GitHub/gamma-ray_from_blazars/data"


def get_datasets_blazars(table=None):

    myArr = np.empty(len(table['Units']), dtype='U20')

    table.meta['catalog_name'] = catalog_NASA_IPAC_name
    table.meta['reference'] = catalog_NASA_IPAC_reference
    table['label'] = myArr
    table['label'].description = 'label of the legend'
    data = Hz_to_TeV(table['Frequency'])
    table['e_ref'] = Column(
        data=data,
        description='energy of reference  (Frequency * 4.1356655385381E-15 eV)', unit=data[0].unit, format='.3e',
    )
    data = Jy_to_erg_by_cm2_s(table['Frequency'], table['Flux Density'])
    table['e2dnde'] = Column(
        data=data,
        description='energy of reference  (Frequency * 4.1356655385381E-15 eV)', unit=data[0].unit, format='.3e',
    )

    table['Frequency'].unit = 'Hz'
    table['Frequency'] = table['Frequency'].to('MHz')
    table['Flux Density'].unit = 'Jy'
    table['Flux Density'] = table['Flux Density'].to('mJy')
    table['Flux density uncertainty']  = table['Flux Density']

    table_copy = table[['Observed Passband', 'e_ref', 'e2dnde', 'Refcode']].copy()
    df = table_copy.to_pandas()
    df = df.drop_duplicates(subset=['e_ref'], keep='first')
    df_ordenado = df.sort_values(by='e_ref')
    df_ = dict(tuple(df_ordenado.groupby('Refcode')))
    datasets_blazars = Datasets()
    for name in list(df['Refcode'].unique()):
        tab = Table.from_pandas(df_[name])
        tab['e_ref'].unit = "TeV"
        tab['e2dnde'].unit = "erg s-1 cm-2"
        fp = FluxPoints.from_table(tab[['e_ref', 'e2dnde']], sed_type="e2dnde")
        ds_fp = FluxPointsDataset(data=fp, name=name)
        datasets_blazars.append(ds_fp)
    return datasets_blazars
        
def set_size_leg_style(obj):

    colors=[] 
    markers=[] 
    linestyles=[]
    
    if not isinstance(obj, list):
        obj = [obj]
        
    while len(colors) < len(obj) +1:
        colors.extend(COLORS)

    while len(markers) < len(obj)+1:
        markers.extend(MARKERS)

    while len(linestyles) < len(obj) +1:
        linestyles.extend(LINESTYLES)
    return colors, markers, linestyles

catalog_NASA_IPAC_name = 'NASA_IPACExtragalacticDatabase'
catalog_NASA_IPAC_reference = 'https://ned.ipac.caltech.edu/'

def Hz_to_TeV(data_freq=None):
    return Quantity(data_freq* 4.1356655385381E-15, 'eV').to('TeV')

def Jy_to_erg_by_cm2_s(data_freq=None, data_dens=None):
    return Quantity(data_dens*data_freq,'Jy Hz').to('erg cm-2 s-1')


CTA_SOURCE_NAME = ['CTA II', 'CTA I', 'CTA III', 'CTA IV']
PULSARS = ['PSR J1826-1334', 'PSR J1826-1256', 'PSR J1837-0604', 'PSR J1838-0537']
PEVATRONS = ['LHAASO J1825-1326', 'LHAASO J1839-0545']
PWNES = ["HESS J1825-137", 'HESS J1826-130', 'HESS J1837-069', 'HESS J1841-055']
on_region_radius_pwn ={PWNES[0]: 0.461*u.deg/2, PWNES[1]: 0.152*u.deg/2, PWNES[2]: 0.355*u.deg/2, PWNES[3]:0.408*u.deg/2}

LINESTYLES = ['solid','dotted','dashed','dashdot']


catalog_lhaaso = CATALOG_REGISTRY.get_cls('publish-nature-lhaaso')()
catalog_pulsar = CATALOG_REGISTRY.get_cls('ATNF')()
catalog_gammacat = CATALOG_REGISTRY.get_cls('gamma-cat')()
catalog_hees = CATALOG_REGISTRY.get_cls('hgps')()

COLOR_LHAASO = "red"
MARKER_LHAASO = "o"

COLOR_CTA = "blue"
MARKER_CTA = "s"

dict_analysis = {
    'LHAASO J1825-1326': {
        'HESS J1825-137': {
            'datasets': [1, 2, 6, 9, 11, 14, 17, 18],
            'pulsar': 'PSR J1826-1334'},
        'HESS J1826-130': {
            'datasets': [0, 3, 5, 7, 8, 12, 18],
            'pulsar': 'PSR J1826-1256'}},
    'LHAASO J1839-0545': {
        'HESS J1837-069': {
            'datasets': [0, 2, 22, 23],
            'pulsar': 'PSR J1837-0604'},
        'HESS J1841-055': {
            'datasets': [1, 3, 17, 23], 
            'pulsar': 'PSR J1838-0537'}}
}


import sys, os

def write_tables_fits(table, path_file, file_name):
    # Writes the flux points table in the fits format
    path_os = os.path.abspath(
        os.path.join(
            f"{path_file}/{file_name}.fits"
        )
    )      

    if path_os not in sys.path:
        sys.path.append(path_os)

    table.write(
        f"{path_os}",
        format = 'fits', 
        overwrite = True
    )   
    return

def write_tables_csv(table, path_file, file_name):
# Writes the flux points table in the csv format
    path_os = os.path.abspath(
        os.path.join(
            f"{path_file}/{file_name}.csv"
        )
    )

    if path_os not in sys.path:
        sys.path.append(path_os)

    table.write(
        f"{path_os}",
        format = 'ascii.ecsv', 
        overwrite = True
    )   
    return

def compute_wstat(dataset_onoff, alpha):
#     log.info("computing wstatistics.")
    wstat = stats.compute_wstat(dataset_onoff=dataset_onoff, alpha=alpha)
    wstat_dict = wstat.info_dict()
    wstat_dict["n_on"] = float(wstat_dict["n_on"])
    wstat_dict["n_off"] = float(wstat_dict["n_off"])
    wstat_dict["background"] = float(wstat_dict["background"])
    wstat_dict["excess"] = float(wstat_dict["excess"])
    wstat_dict["significance"] = float(wstat_dict["significance"])
    wstat_dict["p_value"] = float(wstat_dict["p_value"])
    wstat_dict["alpha"] = float(wstat_dict["alpha"])
    wstat_dict["mu_sig"] =float(wstat_dict["mu_sig"])

    wstat_dict['error'] = float(wstat.error)
    wstat_dict['stat_null'] = float(wstat.stat_null)
    wstat_dict['stat_max'] = float(wstat.stat_max)
    wstat_dict['ts'] = float(wstat.ts)
    print(f"Number of excess counts: {wstat.n_sig}")
    print(f"TS: {wstat.ts}")
    print(f"Significance: {wstat.sqrt_ts}")
    return wstat, wstat_dict



def plot_table_sigma(table):
    table_ = table
    
    N = len(table_)
    dtype = [('color', str)]
    color = np.full(N, "color")
    color_1 = np.full(N, "red")
    color_2 = np.full(N, "blue")
    color_3 = np.full(N, "black")

    mask_z20 = [(_['zenith-angle']) == 20 for _ in table_]
    table_['color_zenith-angle'][mask_z20] =  color_1[mask_z20]
    mask_z40 = [(_['zenith-angle']) == 40 for _ in table_]
    table_['color_zenith-angle'][mask_z40] = color_2[mask_z40]
    mask_z60 = [(_['zenith-angle']) == 60 for _ in table_]
    table_['color_zenith-angle'][mask_z60] = color_3[mask_z60]


    site_z20 = [(_['CTAO'].replace("SubArray", "")) for _ in table_[mask_z20]]
    sigma_z20 = [stats.ts_to_sigma(_['ts']) for _ in table_[mask_z20]]

    site_z40 = [(_['CTAO'].replace("SubArray", "")) for _ in table_[mask_z40]]
    sigma_z40 = [stats.ts_to_sigma(_['ts']) for _ in table_[mask_z40]]

    site_z60 = [(_['CTAO'].replace("SubArray", "")) for _ in table_[mask_z60]]
    sigma_z60 = [stats.ts_to_sigma(_['ts']) for _ in table_[mask_z60]]

    plt.figure(figsize=(10, 10));
    fig, ax = plt.subplots();
    ax.set_title(rf"Livetime: {table_['livetime'][0]:.1f}h")

    ax.plot(site_z20, sigma_z20, markersize= 3, marker="o", label = f"z {table_['zenith-angle'][mask_z20][0]}", color=table_['color_zenith-angle'][mask_z20][0], ls="none");
    ax.plot(site_z40, sigma_z40,markersize= 5, marker="o",  label = f"z {table_['zenith-angle'][mask_z40][0]}", color=table_['color_zenith-angle'][mask_z40][0], ls="none");
    ax.plot(site_z60, sigma_z60, markersize= 7, marker="o",  label = f"z {table_['zenith-angle'][mask_z60][0]}", color=table_['color_zenith-angle'][mask_z60][0], ls="none");

    ax.set_ylabel("Sigma")

    ax.legend()
    plt.axhline(y=15, color='r', linestyle='-')

    plt.show()
def set_leg_style_JCAP(leg_style):
    for name in list(leg_style.keys()):
        if  name.find('LHAASO ') != -1:
            color = COLOR_LHAASO
            marker = MARKER_LHAASO
            leg_style[name] = (color, 'solid' ,marker)
            
        if  name.find('CTAO ') != -1:
            color = COLOR_CTA
            marker = MARKER_CTA
            leg_style[name] = (color, 'solid', marker)
        
    return leg_style

def get_obs_label(irfs_opt=None, offset=None, on_region_radius=None, livetime=None, obs_settings = None):
    if obs_settings:
        irfs_opt = obs_settings.irfs_config
        offset = obs_settings.parameters.offset
        on_region_radius = obs_settings.on_region_radius
        livetime = obs_settings.parameters.livetime

    site = irfs_opt[0].replace("South", "S").replace("North", "N").replace("-SSTSubArray", "SST").replace("-MSTSubArray", "-MST").replace("-LSTSubArray", "-LST")
    average = irfs_opt[1].replace('AverageAz', 'Az').replace("NorthAz", "Nz").replace("SouthAz", "Sz")
    irfs_label = f"{site}-{average}-{irfs_opt[2]}-{irfs_opt[3]}"
#     return  f"{irfs_label} - ON{on_region_radius} ({offset},{livetime})".replace(" ", "").replace("(", " (").replace(".0", "")
    return  f"{irfs_label} ({on_region_radius},{offset},{livetime})".replace(" ", "").replace("(", " (").replace(".0", "")


def create_path_analysis(pulsar_name):
    path_analysis = make_path(f"../analysis/{name_to_txt(pulsar_name)}")
    path_analysis.mkdir(parents=True, exist_ok=True)
    return path_analysis

import pandas as pd 
def create_df_latex(sky_model, pulsar_name, dict_statistics= None):
    
    table = sky_model.parameters.to_table()
    df = table.to_pandas()

    columns_name = []
    df_column = []
    columns_name.append("Pulsar name")
    df_column.append(pulsar_name)
    
    for i, name in enumerate(df["name"]):
        if i<=3 and i != 2:
            value = df["value"][i]
            error = df["error"][i]
            frozen = df["frozen"][i]
            unit = df["unit"][i]
            if i== 3:
                b = df["value"][3]
                R = 1/b
                b_err = df["error"][3]
                value = 1/b
                error = R*(b_err/b)
                unit = "TeV"
                name = f"E_cut"
            if frozen == True:
                value = "{:.0f}".format(value)
            else: 
                if i == 0 or i ==3:
                    value = "{:.3f} ({:.3f})".format(value, error)
                else:
                    value = "{:.2e} ({:.2e})".format(value, error)

            if unit:
                name = f"{name} ({unit})"
            
            columns_name.append(name)
            df_column.append(value)
    if dict_statistics:
        name = 'significance'
        columns_name.append(name)
        value = dict_statistics[name]
        value = value = "{:.2f}".format(value)
        df_column.append(value)

        name = 'total_stat'
        columns_name.append(name)
        value = dict_statistics[name]
        value = value = "{:.2f}".format(value)
        df_column.append(value)
            
            
    df = pd.DataFrame([df_column], columns = columns_name)
    df_tex  = df.to_latex(index=False)
    return print(df_tex)


import re
def set_label_datasets(dataset_name):
    if dataset_name.find('HESS') != -1:
        test_string = dataset_name
        spl_word = ':'
        match = re.search(spl_word, test_string)
        if match:
            source_name = test_string[:match.end()-1]
            cat_tag = (test_string[match.end():]).replace(" ", "")
            if cat_tag == 'hgps':
                year = "2018"
            else:
                if source_name == "HESS J1825-137":
                    year = "2006"
                if source_name == "HESS J1826-130":
                    year = "2017"
                if source_name == "HESS J1837-069":
                    year = "2006"
                if source_name == "HESS J1841-055":
                    year = "2018b"
            return f'{source_name} ({year})' 
    else:
        return dataset_name

    
def show_SED(
    datasets = None,  
    models = None,
    leg_style = None, 
    sed_type = "e2dnde", 
    plot_axis =  dict(
        label =  (r'$\rm{E\ [TeV] }$', r'$\rm{E^2\ J(E)\ [TeV\ cm^{-2}\ s^{-1}] }$'),
        units =  (          'TeV',                       'TeV  cm-2     s-1')
    ),
    plot_limits = dict(
        energy_bounds = [1e-5, 3e2] * u.TeV,
        ylim = [1e-23, 1e-7]
    ),
    leg_place = dict(
        bbox_to_anchor = (0, -0.45), # Set legend outside plot
        ncol=3, 
        loc='lower left', 
    ),
    file_path=None,
    PeVatron = None,
    cta_source_name=None,
    ):    

    ax = plt.subplot()

    ax.xaxis.set_units(u.Unit(plot_axis['units'][0]))
    ax.yaxis.set_units(u.Unit(plot_axis['units'][1]))

    kwargs = {
        "ax": ax, 
        "sed_type": sed_type,
    #         "uplims": True
    }

    for index, dataset in enumerate(datasets):
        color = leg_style[dataset.name][0]
        marker = leg_style[dataset.name][2]

        if isinstance(cta_source_name, list):
            label = set_label_datasets(dataset.name).replace('CTAO', f'{cta_source_name[index]}')
        else: label = set_label_datasets(dataset.name).replace('CTAO', f'{cta_source_name}')
        
        dataset.data.plot(
            label = label, 
            marker = marker, 
            color=color,
            **kwargs
        )

    if models: 
        for index, model in enumerate(models):
            linestyle = leg_style[model.name][1]
            color = leg_style[model.name][0]
            spectral_model = model.spectral_model
            if  PeVatron == 1:
                energy_bounds = [7e-2, 2e3] * u.TeV
            elif PeVatron == 2: energy_bounds = [7e-2, 3e2] * u.TeV


            spectral_model.plot(label = f"{model.name}".replace('CTAO', f'{cta_source_name}'), energy_bounds=energy_bounds,  linestyle = linestyle,  marker = ',', color="black", **kwargs)
            spectral_model.plot_error(energy_bounds=energy_bounds,**kwargs)

    ax.set_ylim(plot_limits['ylim'])
    ax.set_xlim(plot_limits['energy_bounds'])

    ax.legend(**leg_place)

    plt.xlabel(plot_axis['label'][0])   
    plt.ylabel(plot_axis['label'][1])

    if file_path:
        plt.savefig(file_path, bbox_inches='tight')
        
    #    plt.grid(which="both")
    plt.show()

    return


from astropy import units as u
from astropy.table import Table
import numpy as np
def plot_SED_CR(
    name = "region_of_interest", 
    datasets = None,  
    models = None,
    leg_style = None, 
    sed_type = "e2dnde", 
    dict_plot_axis =  dict(
    label =  (r'$\rm{E\ [MeV] }$', r'$\rm{E^2\ J(E)\ [MeV\ cm^{-2}\ s^{-1}] }$'),
    units =  (          'MeV',                       'MeV  cm-2     s-1')
),
    dict_plot_limits = dict(
        energy_bounds = [1e-5, 3e2] * u.TeV,
        ylim = [1e-23, 1e-7]
    ),
    dict_leg_place = dict(
#         bbox_to_anchor = (0, -0.45), # Set legend outside plot
        ncol=2, 
        loc='lower left', 
    ),
    pulsar_name = None,
        model_name= None,
    table= None,
    PeVatron=1,
    file_path=None,
    cta_source_name=None,

):    
    
    table_g = Table.read(f"./data/galprop_{name_to_txt(pulsar_name)}.csv",format='ascii', delimiter=',', comment='#')
    table_h = Table.read(f"./data/huang_{name_to_txt(pulsar_name)}.csv",format='ascii', delimiter=',', comment='#')
    table_h2 = Table.read(f"./data/huang2_{name_to_txt(pulsar_name)}.csv",format='ascii', delimiter=',', comment='#')

 
    ax = plt.subplot()
    table = table_g
    ax.plot(table["e_ref"], table["total"]/ 4 * np.pi,'-b',label='Total', linewidth=2)
    ax.plot(table["e_ref"], table["pi"]/ 4 * np.pi, '--g', label='Pion decay', linewidth=2)
    ax.plot(table["e_ref"], table["ic"]/ 4 * np.pi, '-.r',label='Inverse Compton', linewidth=2)
    ax.plot(table["e_ref"], table["br"]/ 4 * np.pi,':k', label='Bremsstrahlung', linewidth=2)

    table = table_h
    ax.plot(table["e_ref"]* 1e6, table["flux"]* 1e6 / 4 * np.pi,'-.', color='gold', label='Huang (2022)')

    table = table_h2
    ax.plot(table["e_ref"]* 1e6, table["flux"]* 1e6 / 4 * np.pi, color='turquoise', label='Huang 2 (2022)')
    
    ax.xaxis.set_units(u.Unit(dict_plot_axis['units'][0]))
    ax.yaxis.set_units(u.Unit(dict_plot_axis['units'][1]))

    kwargs = {
        "ax": ax, 
        "sed_type": sed_type,
#         "uplims": True
    }
                        
    for index, dataset in enumerate(datasets):
        color = leg_style[dataset.name][0]
        marker = leg_style[dataset.name][2]

        label = set_label_datasets(dataset.name).replace('CTAO', f'{cta_source_name}') 

        dataset.data.plot(
            label = label, 
            marker = marker, 
            color=color,
            **kwargs
        )

    if models: 
        for index, model in enumerate(models):
            linestyle = leg_style[model.name][1]
            color = leg_style[model.name][0]
            spectral_model = model.spectral_model
            if  PeVatron == 1:
                energy_bounds = [7e-2, 2e3] * u.TeV
            else: energy_bounds = [7e-2, 3e2] * u.TeV


            spectral_model.plot(label = f"{model.name}".replace('CTAO', f'{cta_source_name}'), energy_bounds=energy_bounds,  linestyle = linestyle,  marker = ',', color="black", **kwargs)
            spectral_model.plot_error(energy_bounds=energy_bounds,**kwargs)
            
    ax.set_ylim(dict_plot_limits['ylim'])
    ax.set_xlim(dict_plot_limits['energy_bounds'])
    

    ax.legend(**dict_leg_place)
    
    plt.xlabel(dict_plot_axis['label'][0])   
    plt.ylabel(dict_plot_axis['label'][1])
    
    if file_path:
        plt.savefig(file_path, bbox_inches='tight')

        
#    plt.grid(which="both")
    plt.show()
    
    return