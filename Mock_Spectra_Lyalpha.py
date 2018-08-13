import numpy as np
from matplotlib import pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import io as lsio
from linetools.spectralline import AbsLine
from linetools.analysis import voigt as ltav
from numpy import random, round
from astropy import units as u
from astropy import constants as const
from bokeh.io import output_notebook, show, output_file
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import Range1d
from pyigm.abssys.igmsys import IGMSystem
from pyigm.fN import mockforest
import pdb
from pyigm.fN.fnmodel import FNModel

from mpi4py import MPI


def main():
    """ The main function for the code

                Parameters
                ----------

                Returns
                -------

        """
    #np.set_printoptions(threshold=np.nan)
    model_wave_z = []
    bins = 0.
    redshifti = 2.7
    wmin = 912
    wmax = 1170
    z = []
    sample_size = 2

    k = np.loadtxt('/Users/Jenny/documents/sdss_data/z_3.6_3.7.txt')
    i = range(0, sample_size, 1)

    #z_redshift = [3.65 for i in range(sample_size)]
    z_redshift = k[i,0]
    


    z_err = 500/3/10**5*z_redshift

    


    #call the function of generating redshift





    fN_default = FNModel.default_model()

    # call the function of creating mock spectra

    stacking_spectrum(z_random(z_redshift, sample_size, z_err, wmin, wmax)[0],
                      z_random(z_redshift, sample_size, z_err, wmin, wmax)[1],
                      z_random(z_redshift, sample_size, z_err, wmin, wmax)[2])






def z_random(z_obs, sample_size, z_err, wmin, wmax):
    """ The main function for generate z and observed wavelength

                Parameters
                ----------
                z_obs : input observed redshift
                z_err: error for the observed redshift
                wmin : minimum input rest frame wavelength
                wmax: maximum input rest frame wavelength
                sample_size: sample size for mock spectra


                Returns
                -------
                Array of observed wavelength and the redshift of the source
        """

    #z = np.linspace(redshifti - bins, redshifti + bins, sample_size)
    zactual = []
    zactual = np.random.normal(z_obs, z_err, sample_size)
    # deltaz = np.array(zactual) - np.array(z_redshift)
    model_wave_total_in = []
    model_wave_z_in = []
    for item in np.array(zactual):
        wminz = wmin * (1 + item)
        wmaxz = wmax * (1 + item)
        model_wave = np.linspace(wminz, wmaxz, 1080) * u.AA
        model_wave_total_in.append(model_wave)

    return model_wave_total_in, zactual, z_obs


def stacking_spectrum(mock_wave_total, z_actual, z_obs):
    """ The main function for generate the stacking spectrum
                Parameters
                ----------
                mock_wave_total : input observed wavelength
                z_r: central redshift



                Returns
                -------
                Array of rest frame wavelength and stacking fluxes
    """
    #fN_I14 = FNModel('Gamma')
    mock_spec_total = []
    mean_weighted_flux = []
    mock_spec_total_flux = []
    mock_spec_wave = []

    fN_default = FNModel.default_model()

 #getting start for running in parallel
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        inter = Intervals(len(z_actual), size)
    else:
        inter=None
    inter = comm.bcast(inter, root=0)
    start = inter[rank][0]
    end = inter[rank][1]

    print(rank, start, end, size)

    #pdb.set_trace()
# start to run in parallel for generating mock spectra

    for i in range(start,end):

        mock_spec, HI_comps, _ = mockforest.mk_mock(mock_wave_total[i], z_actual[i], fN_default, bfix=None, add_conti=False, s2n=1000.)
    #mock_spec_total.append(mock_spec)

        plt.ylim(-20,100)
        mock_spec_total_flux.append(mock_spec.flux*30)

        mock_spec_wave.append(mock_spec.wavelength/(1+z_obs[i]))
        #print(HI_comps)
        #plt.plot(mock_spec.wavelength/(1+z_obs[i]), mock_spec.flux)
        #plt.show()

    all_wave = comm.gather(mock_spec_wave, root=0)
    all_flux = comm.gather(mock_spec_total_flux, root=0)

# combine and stack the data
    if rank == 0:
        all_wave = np.array(all_wave[0])
      #all_wave = np.hstack(all_wave)
        all_flux = np.array(all_flux[0])
      #all_flux = np.hstack(all_flux)

      #print(all_wave)
      #print(all_flux)

        stacking_wave = np.mean(all_wave, axis=0)
        stacking_flux = np.mean(all_flux, axis=0)
        plt.ylim(-20, 100)
        plt.title('mock stacking spectra_1200_s/n=1000_flat_continuum_z_err0_brandom_z3.65_P14_1')
        plt.xlabel('wavelength')
        plt.ylabel('fluxes')
        plt.plot(stacking_wave, stacking_flux)

        #np.savetxt('flat_z3.65_z_err0_brandom_1000_1200_wave_P14_1.txt', stacking_wave, delimiter=" ", newline='\n')
        #np.savetxt('flat_z3.65_z_err0_brandom_1000_1200_flux_P14_1.txt', stacking_flux, delimiter=" ", newline='\n')

        plt.show()

        return  stacking_wave, stacking_flux




#function for running in parallel

def Intervals(n, size):
    interval = np.floor(n/size)
    rem = n%size
    inter = np.zeros([size,2])
    for i in range(size):
        if rem !=0:
            add=interval+1
            rem+=-1
        else:
            add=interval
        inter[i][0]=inter[i-1][1]
        inter[i][1]=inter[i][0]+add

    return inter.astype(int)





if __name__ == '__main__':
    main()
