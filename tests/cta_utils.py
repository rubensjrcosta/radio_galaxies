from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ..spectrum import SpectrumObservation
from ..spectrum.utils import calculate_predicted_counts
from ..spectrum.core import PHACountsSpectrum
from ..utils.energy import EnergyBounds
from ..utils.random import get_random_state

from ..spectrum.models import AbsorbedSpectralModel, TableModel

__all__ = [
    'Target',
    'ObservationParameters',
    'CTAObservationSimulation',
]


class Target(object):
    """Class storing source information

    Parameters
    ----------
    name : `str`
        Name of the source
    model : `~gammapy.spectrum.models.SpectralModel`
        Model of the source
    redshift : `~astropy.units.Quantity`
        Redshift of the source
    ebl_model_name: `str`
        EBL model (franceschini, dominguez, finke or None)
    """

    def __init__(self, name=None,
                 model=None,
                 redshift=None,
                 ebl_model_name=None):
        self.name = name
        self.model = model
        self.redshift = redshift
        self.ebl_model_name = ebl_model_name
        self.abs_model = None
        filename = None
        if ebl_model_name in 'franceschini':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_franceschini.fits.gz'
        elif ebl_model_name in 'dominguez':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_dominguez11.fits.gz'
        elif ebl_model_name in 'finke':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/frd_abs.fits.gz'
        else:
            pass

        if redshift is not None:
            absorption = TableModel.read_xspec_model(filename, redshift)
            self.abs_model = AbsorbedSpectralModel(spectral_model=model,
                                                   table_model=absorption)
        else:
            self.abs_model = model

    def __str__(self):
        """Target report (`str`)."""
        ss = '*** Target parameters ***\n'
        ss += 'Name={}\n'.format(self.name)
        for par in self.model.parameters.parameters:
            ss += '{}={} {}\n'.format(par.name, str(par.value), par.unit)
        ss += 'Redshift={}'.format(self.redshift)
        return ss

    def from_fermi_lat_catalogue(name):
        raise NotImplementedError


class ObservationParameters(object):
    """Class storing observation parameters

    Parameters
    ----------
    alpha : `~astropy.units.Quantity`
        Normalisation between ON and OFF regions
    livetime :  `~astropy.units.Quantity`
        Observation time
    emin :  `~astropy.units.Quantity`
        Minimal energy for simulation
    emax :  `~astropy.units.Quantity`
        Maximal energy for simulation
    """

    def __init__(self, alpha=None, livetime=None,
                 emin=None, emax=None):
        self.alpha = alpha
        self.livetime = livetime
        self.emin = emin
        self.emax = emax

    def __str__(self):
        """Observation summary report (`str`)."""
        ss = '*** Observation parameters summary ***\n'
        ss += 'alpha={} [{}]\n'.format(self.alpha.value, self.alpha.unit)
        ss += 'livetime={} [{}]\n'.format(self.livetime.value, self.livetime.unit)
        ss += 'emin={} [{}]\n'.format(self.emin.value, self.emin.unit)
        ss += 'emax={} [{}]\n'.format(self.emax.value, self.emax.unit)
        return ss


class CTAObservationSimulation(object):
    """Class dedicated to simulate observation from one set
    of IRF and one target.

    TODO : Should be merge with `~gammapy.spectrum.SpectrumSimlation`

    Parameters
    ----------
    perf : `~gammapy.scripts.CTAPerf`
        CTA performance
    target : `~gammapy.scripts.Target`
        Source
    """

    @staticmethod
    def simulate_obs(perf, target, obs_param):
        """
        Simulate observation with given parameters

        Parameters
        ----------
        perf : `~gammapy.scripts.CTAPerf`
            CTA performance
        target : `~gammapy.scripts.Target`
            Source
        obs_param : `~gammapy.scripts.ObservationParameters`
            Observation parameters
        """
        livetime = obs_param.livetime
        alpha = obs_param.alpha.value
        emin = obs_param.emin
        emax = obs_param.emax

        model = target.abs_model

        # Create energy dispersion
        e_reco_min = perf.bkg.energy.lo[0]
        e_reco_max = perf.bkg.energy.hi[-1]
        e_reco_bin = perf.bkg.energy.nbins
        e_reco_axis = EnergyBounds.equal_log_spacing(e_reco_min,
                                                     e_reco_max,
                                                     e_reco_bin,
                                                     'TeV')

        # Compute expected counts
        reco_energy = perf.bkg.energy
        bkg_rate_values = perf.bkg.data.data * livetime.to('s')
        predicted_counts = calculate_predicted_counts(model=model,
                                                      aeff=perf.aeff,
                                                      livetime=livetime,
                                                      edisp=perf.rmf,
                                                      e_reco=e_reco_axis)

        # Randomise counts
        rand = get_random_state('random-seed')
        on_counts = rand.poisson(predicted_counts.data.data.value)  # excess
        bkg_counts = rand.poisson(bkg_rate_values.value)  # bkg in ON region
        off_counts = rand.poisson(bkg_rate_values.value / alpha)  # bkg in OFF region

        on_counts += bkg_counts  # evts in ON region

        counts_kwargs = dict(energy_lo=reco_energy.lo,
                             energy_hi=reco_energy.hi,
                             livetime=livetime,
                             creator='gammapy')
        on_vector = PHACountsSpectrum(data=on_counts,
                                      backscal=1,
                                      **counts_kwargs)

        off_vector = PHACountsSpectrum(energy_lo=reco_energy.lo,
                                       energy_hi=reco_energy.hi,
                                       data=off_counts,
                                       livetime=livetime,
                                       backscal=1. / alpha,
                                       is_bkg=True,
                                       creator='gammapy')

        
        obs = SpectrumObservation(on_vector=on_vector,
                                  off_vector=off_vector,
                                  aeff=perf.aeff,
                                  edisp=perf.rmf)

        # Set threshold according to the closest energy reco from bkg bins
        idx_min = np.abs(reco_energy.lo - emin).argmin()
        idx_max = np.abs(reco_energy.lo - emax).argmin()
        obs.lo_threshold = reco_energy.lo[idx_min]
        obs.hi_threshold = reco_energy.lo[idx_max]

        return obs

    @staticmethod
    def plot_simu(simu, target):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(10, 5))

        # Spectrum plot
        energy_range = [0.01 * u.TeV, 100 * u.TeV]
        target.abs_model.plot(ax=ax1, energy_range=energy_range,
                              label='Model')
        plt.text(0.55, 0.65, target.__str__(),
                 style='italic', transform=ax1.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        ax1.set_xlim([energy_range[0].value, energy_range[1].value])
        ax1.set_ylim(1.e-17, 1.e-5)
        ax1.grid(which='both')
        ax1.legend(loc=0)

        # Counts plot
        on_off = simu.on_vector.data.data.value
        off = 1. / simu.off_vector.backscal * simu.off_vector.data.data.value
        excess = on_off - off
        bins = simu.on_vector.energy.lo.value
        x = simu.on_vector.energy.nodes.value
        ax2.hist(x, bins=bins, weights=on_off,
                 facecolor='blue', alpha=1, label='ON')
        ax2.hist(x, bins=bins, weights=off,
                 facecolor='green', alpha=1, label='OFF')
        ax2.hist(x, bins=bins, weights=excess,
                 facecolor='red', alpha=1, label='EXCESS')
        ax2.legend(loc='best')
        ax2.set_xscale('log')
        ax2.set_xlabel('Energy [TeV]')
        ax2.set_ylabel('Expected counts')
        ax2.set_xlim([energy_range[0].value, energy_range[1].value])
        ax2.set_ylim([0.0001, on_off.max()*(1+0.05)])
        ax2.vlines(simu.lo_threshold.value, 0, 1.1 * on_off.max(),
                  linestyles='dashed')
        ax2.grid(which='both')
        plt.text(0.55, 0.05, simu.__str__(),
                 style='italic', transform=ax2.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        plt.tight_layout()
