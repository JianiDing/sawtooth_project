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
print (3)
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
    bins = 0.1
    redshifti = 2.4
    wmin = 900
    wmax = 1170
    z = []
    sample_size = 3
    #call the function of generating redshift

    z_random(z,bins, sample_size, redshifti, wmin, wmax)
    global fN_default
    fN_default = FNModel.default_model()

    # call the function of creating mock spectra

    stacking_spectrum(z_random(z,bins, sample_size, redshifti, wmin, wmax)[0],
                      z_random(z,bins, sample_size, redshifti, wmin, wmax)[1])




##general redshift distribution of the source


def z_random(z, bins, sample_size, redshifti, wmin, wmax):
    """ The main function for generate z and observed wavelength

                Parameters
                ----------
                bins : bins for the redshift
                redshifti: central redshift
                wmin : minimum input rest frame wavelength
                wmax: maximum input rest frame wavelength
                sample size: sample size for mock spectra


                Returns
                -------
                Array of observed wavelength and the redshift of the source
        """
    z = np.linspace(redshifti - bins, redshifti + bins, sample_size)
    model_wave_total_in = []
    model_wave_z_in = []
    for item in z:
      wminz = wmin*(1+item)
      wmaxz = wmax*(1+item)
      model_wave = np.linspace(wminz,wmaxz,1000)*u.AA
      model_wave_total_in.append(model_wave)
    #model_wave_z_in.append(item)
    return model_wave_total_in, z


def stacking_spectrum(mock_wave_total, z_r):
 """ The main function for generate the stacking spectrum
                Parameters
                ----------
                mock_wave_total : input observed wavelength
                z_r: central redshift



                Returns
                -------
                Array of rest frame wavelength and stacking fluxes
 """

 mock_spec_total = []
 mean_weighted_flux = []
 mock_spec_total_flux = []
 mock_spec_wave = []
 stacking_flux = []
 stacking_wave = []

 #getting start for running in parallel
 comm = MPI.COMM_WORLD
 rank = comm.Get_rank()
 size = comm.Get_size()
 if rank == 0:
  inter = Intervals(len(z_r), size)
 else:
  inter=None
  inter = comm.bcast(inter, root=0)
  start = inter[rank][0]
  end = inter[rank][1]

  print(rank, start, end, size)


# start to run in parallel for generating mock spectra

  for i in range(start,end):

    mock_spec, HI_comps, _ = mockforest.mk_mock(mock_wave_total[i], z_r[i], fN_default, s2n=20.)
    #mock_spec_total.append(mock_spec)

    plt.ylim(-20,100)
    mock_spec_total_flux.append(mock_spec.flux)

    mock_spec_wave.append(mock_spec.wavelength/(1+z_r[i]))

    #plt.plot(mock_spec.wavelength/(1+z_r[i]), mock_spec.flux)
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
      plt.title('mock stacking spectra_3000_s/n=20_flat_continumm')
      plt.xlabel('wavelength')
      plt.ylabel('fluxes')
      plt.plot(stacking_wave, stacking_flux)

      #np.savetxt('flat_20_3000_wave.txt', stacking_wave, delimiter=" ", newline='\n')
      #np.savetxt('flat_20_3000_flux.txt', stacking_flux, delimiter=" ", newline='\n')
      #print (stacking_wave, stacking_flux)
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
