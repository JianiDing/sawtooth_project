import specdb
import numpy as np
from specdb.specdb import SpecDB
from matplotlib import pylab as plt
from scipy import stats
from linetools.spectra.utils import smash_spectra
from linetools.spectra.xspectrum1d import XSpectrum1D
from random import choices
import itertools as it
import scipy.stats
from scipy import stats



def bootstrap_selection(zlow, zhigh, wavelow, wavehigh,meta_tbl,sdb14):
    '''
    function for generate mean xspectrum and all the bootstrap spectrum
    :param zlow: lower limit for redshift
    :param zhigh: upper limit for redshift
    :param wavelow: lower limit for wavelength range
    :param wavehigh: upper limit for wavelength range
    :return: bootstrap xspectrum: stacking, individal mean redhsift of bootstrap, mean flux, mena redshift
    '''

    #getting spectrum for each redshift bin from igmspec

    count = 0
    meta = meta_tbl[(meta_tbl['Z_PIPE'] > zlow)
                    & (meta_tbl['Z_PIPE_ERR'] > 0.)
                    & (meta_tbl['Z_PIPE'] < zhigh)]

    #& (abs(meta_tbl['Z_VI'] - meta_tbl['Z_PIPE']) < 60 / 1216 * meta_tbl['Z_VI'])
    print (len(meta))
    spec = sdb14.spec_from_meta(meta)


    wavelength = []

    flux = []

    length = []
    error = []

    z = []
    ave_stn = []

    #finding out the maximum length for each spectrum within certain wavelength range

    for i in range(0, len(meta['Z_PIPE'])):

        mask2 = (np.array(spec[i].wavelength) < wavehigh * (1 + meta['Z_PIPE'][i])) & (
            np.array(spec[i].wavelength) > wavelow * (1 + meta['Z_PIPE'][i]))
        length.append(len(spec[i].wavelength[mask2]))
        mask_s_t_n = (np.array(spec[i].wavelength) < 1150 * (1 + meta['Z_PIPE'][i])) & (
            np.array(spec[i].wavelength) > 1075 * (1 + meta['Z_PIPE'][i]))
        # snr cut for snr >3
        signal = spec[i].flux[mask_s_t_n]
        s_t_n = signal / spec[i].sig[mask_s_t_n]
        ave_stn.append(np.median(s_t_n))


    #picking out the wavelength range we need
    for i in range(0, len(meta['Z_PIPE'])):
        mask3 = (np.array(spec[i].wavelength) < wavehigh * (1 + meta['Z_PIPE'][i])) & (
            np.array(spec[i].wavelength) > wavelow * (1 + meta['Z_PIPE'][i]))
        mask_mean = (np.array(spec[i].wavelength) < 1460 * (1 + meta['Z_PIPE'][i])) & (
            np.array(spec[i].wavelength) > 1440 * (1 + meta['Z_PIPE'][i]))

    #mask all the prominet skylines and generate the final flux, error, wavelength array for each spectrum within the redshift bin and wavelength range
        if (len(spec[i].wavelength[mask3] / (1 + meta['Z_PIPE'][i])) > (np.max(length) - 3)) & (ave_stn[i] > 2.2):
            # mean_error = np.sqrt(np.sum(np.array(spec[i].sig[mask1])) ** 2) / len(spec[i].flux[mask1])
            mask_mean_sky = (np.array(spec[i].wavelength) < 5590) & (np.array(spec[i].wavelength) > 5570)
            flux_error = spec[i].sig[mask3][:(np.max(length) - 11)]
            #print(len(flux_error))
            mask4 = (np.array(spec[i].wavelength[mask3][:(np.max(length) - 11)]) < 5590) & (
            np.array(spec[i].wavelength[mask3][:(np.max(length) - 11)]) > 5570)
            mask5 = (np.array(spec[i].wavelength[mask3][:(np.max(length) - 11)]) < 4370) & (
            np.array(spec[i].wavelength[mask3][:(np.max(length) - 11)]) > 4350)
            spec[i].flux[mask_mean_sky] = np.zeros(len(spec[i].flux[mask_mean_sky]))
            mean = np.mean(spec[i].flux[mask_mean])
            flux_error[mask4] = np.zeros(len(flux_error[mask4]))
            flux_error[mask5] = np.zeros(len(flux_error[mask5]))
            if ((np.array(spec[i].flux[mask3][:(np.max(length) - 11)]) / (mean)) < -8.0).any():
                flux_error = np.zeros(len(flux_error))
                count = count + 1
            error.append(flux_error)
            flux.append(np.array(spec[i].flux[mask3][:(np.max(length) - 11)]) / (mean))

            wavelength.append(spec[i].wavelength[mask3][:(np.max(length) - 11)] / (1 + meta['Z_PIPE'][i]))
            z.append(meta['Z_PIPE'][i])

    #generate the xspectrum
    stacking = []
    stacking_wave = np.mean(np.array(wavelength), axis=0)
    stacking_wave_final = np.repeat([stacking_wave], len(wavelength), axis = 0)

    wave = np.array(wavelength)
    sp = XSpectrum1D(np.array(stacking_wave_final), np.array(flux), np.array(error), verbose=False)
    totalindex = range(0, len(flux))
    redshift = []
    redshift_mean = np.mean(z)
    #genrate the final combine spectra for each redshift bin
    mean = smash_spectra(sp,'average')
    #generate the total bootstrap spectra (250) for 10000 times
    number = int(len(z)*0.9)
    bootstrapindex = sorted(choices(totalindex, k=number) for i in range(times))
    print ('boop',len(bootstrapindex),len(bootstrapindex[0]))
    for ii in range(0, len(bootstrapindex)):
        spstack = sp[bootstrapindex[ii]]
        stacking.append(smash_spectra(spstack,'average'))
        redshift.append(np.mean([z[x] for x in bootstrapindex[ii]]))





    '''
    plt.plot(mean.wavelength, mean.flux, label="z=3.9-4.0")
    plt.xlabel('rest frame wavelength')
    plt.ylabel('flux')
    plt.legend()
    plt.show()
    '''
    return stacking,redshift,mean,redshift_mean






def bootstrap_flux(zlow, deltaz, wmin, wmax):
    '''

    :param zlow:
    :param deltaz:
    :param wmin:
    :param wmax:
    :return: the covariance matrix for flux
    '''
    total_spec,redshift,mean,redshift_mean = bootstrap_selection(zlow, zlow + deltaz, wmin, wmax)
    flux_bootstrap = []
    for i in range(0, len(total_spec)):
        flux_bootstrap.append(total_spec[i].flux-mean.flux)

    flux_stack = np.vstack(np.transpose(flux_bootstrap))
    flux_bootstrap_final = np.dot(flux_stack, np.transpose(flux_stack)) / (times - 1)



    return flux_bootstrap_final,flux_stack,mean,redshift_mean



def delta_tau(zlow,deltaz,zrel,wmin,wmax):

    total_spec,redshiftz,mean,redshift_mean = bootstrap_selection(zlow, zlow + deltaz, wmin, wmax)

    # relative_flux_3136 = np.array(mock31_flux) / np.array(mock36_flux)
    total_spec_rel,redshiftrel,meanrel,redshift_mean_rel = bootstrap_selection(zrel, zrel + deltaz, wmin, wmax)

    # relative_error = (np.array(spectrum_selection(zlow,zlow+deltaz,wmin,wmax)[1])/np.array(spectrum_selection(zrel,zrel+deltaz,wmin,wmax)[1]))*(np.sqrt((np.array(spectrum_selection(zlow,zlow+deltaz,wmin,wmax)[2])/np.array(spectrum_selection(zlow,zlow+deltaz,wmin,wmax)[1]))**2+
    # (np.array(spectrum_selection(zrel,zrel+deltaz,wmin,wmax)[2])/np.array(spectrum_selection(zrel,zrel+deltaz,wmin,wmax)[1]))**2))

    meantau = -np.log(np.array(mean.flux) / np.array(meanrel.flux))
    redshifti = np.array(mean.wavelength) / 1215.67 * (redshift_mean + 1) - 1

    redshiftirel = np.array(meanrel.wavelength) / 1215.67 * (redshift_mean_rel + 1) - 1

    redshiftbinmean = scipy.stats.binned_statistic(redshifti, redshifti, statistic='mean', bins=3)
    redshiftrelbinmean = scipy.stats.binned_statistic(redshifti, redshiftirel, statistic='mean', bins=3)
    tau_binmean = scipy.stats.binned_statistic(redshifti, meantau, statistic='mean', bins=3)
    # print( 'rel',relative_flux)
    deltatau = []
    wave = []

    flux=[]
    outputbin = []
    outputindi = []
    base = []
    for i in range(0, len(total_spec)):
        deltataui = -np.log(np.absolute(np.array(total_spec[i].flux) / np.array(total_spec_rel[i].flux)))
        deltatau.append(-np.log(np.array(total_spec[i].flux) / np.array(total_spec_rel[i].flux)))
        wave.append(total_spec[i].wavelength)
        flux.append(total_spec[i].flux / total_spec_rel[i].flux)

        redshift = np.array(total_spec[i].wavelength) / 1215.67 * (redshiftz[i] + 1) - 1

        redshift2 = np.array(total_spec_rel[i].wavelength) / 1215.67 * (redshiftrel[i] + 1) - 1


        redshiftbin = scipy.stats.binned_statistic(redshift, redshift, statistic='mean', bins=3)
        redshiftrelbin = scipy.stats.binned_statistic(redshift, redshift2, statistic='mean', bins=3)
        tau_bin = scipy.stats.binned_statistic(redshift, deltataui, statistic='mean', bins=3)
        base = np.append(np.random.normal(0.363, 0.020, 1),np.random.normal(0.373, 0.018, 1))
        base = np.append(base,np.random.normal(0.441, 0.023, 1))



        outputbin.append(np.array(tau_bin[0]+base)-np.array(tau_binmean[0]+base))
        outputindi.append(np.array(deltataui)-np.array(meantau))






    wave_final = np.mean(wave,axis=0)


    return redshifti, redshiftbinmean, np.transpose(outputbin), np.transpose(outputindi),








def bootstrap_absolutetau(zlow, deltaz, wmin, wmax):
    conti_model = np.loadtxt("/Users/Jenny/Documents/pycodes/Mock_spectra/new_measurement_0211/conti_model_new.txt")

    


        
    mu = 0.36
    sigma = 0.012
    total_spec, redshift, mean, redshift_mean = bootstrap_selection(zlow, zlow + deltaz, wmin, wmax)


    abstau = np.random.normal(mu, sigma, times)
    tau_bootstrap = []
    deltatau_bootstrap = []
    xalpha = []
    tautotal = []
    for i in range(0, len(total_spec)):


        x2 = np.array(total_spec[i].wavelength) / np.array(1215.67 / (1 + np.array(redshift[i]))) - 1
        xi = np.array(mean.wavelength) / np.array(1215.67 / (1 + np.array(redshift_mean))) - 1





        tau = -np.log(total_spec[i].flux/conti_model)


        tau_mean = -np.log(mean.flux/conti_model)
        #print (tau,tau_mean)
        xalphamean = np.ravel(xi)
        xalpha.append(np.ravel(x2))
        tautotal.append(np.ravel(tau))
        #delta_taui = tautotal - np.mean(tautotal[np.where((ref - 0.005 < xalpha) & (xalpha < ref + 0.005))])
        #reltau = np.ravel(tau_mean)[np.where((ref - 0.005 < xalphamean) & (xalphamean < ref + 0.005))]
        #delta_mean = tau_mean - reltau
        #print (reltau)
        #tau_totali = delta_taui + abstau[i]
        #tau_totalmean = delta_mean + mu

        #tau_bootstrap.append(tau_totali - tau_totalmean)
        #deltatau_bootstrap.append(delta_taui-delta_mean)
    #flux_stack = np.vstack(np.transpose(tau_bootstrap))

    #flux_bootstrap_tau = np.dot(flux_stack, np.transpose(flux_stack)) / (times - 1)
    #deltatest = np.vstack(np.transpose(deltatau_bootstrap))
    #deltatestfinal = np.dot(deltatest, np.transpose(deltatest)) / (times - 1)
    return xalpha,tautotal




def bootstrap_bin_err(zlow, deltaz, wmin, wmax):
    conti_model2 = np.loadtxt("/Users/Jenny/Documents/pycodes/Mock_spectra/new_measurement_0211/conti_model_new.txt")
    total_spec, redshift, mean, redshift_mean = bootstrap_selection(zlow, zlow + deltaz, wmin, wmax)
    tau_bootstrap = []
    deltatau_bootstrap = []
    for i in range(0, len(total_spec)):
        x2 = np.array(total_spec[i].wavelength) / np.array(1215.67 / (1 + np.array(redshift[i]))) - 1
        xi = np.array(mean.wavelength) / np.array(1215.67 / (1 + np.array(redshift_mean))) - 1



        tau = -np.log(total_spec[i].flux / conti_model2)
        tau_mean = -np.log(mean.flux / conti_model2)
        # print (tau,tau_mean)
        xalpha = np.ravel(x2)
        xalphame = np.ravel(xi)
        ind = np.unravel_index(np.argsort(xalpha, axis=None), xalpha.shape)
        indme = np.unravel_index(np.argsort(xalphame, axis=None), xalphame.shape)
        tautotal = np.ravel(tau)
        taume = np.ravel(tau_mean)

        xalphabin = stats.binned_statistic(xalpha[ind], xalpha[ind], statistic='mean',bins = 15)
        xalphabinme = stats.binned_statistic(xalphame[indme], xalphame[indme], statistic='mean',bins = 15)
        taubin = stats.binned_statistic(xalpha[ind], tautotal[ind], statistic='mean', bins=15)
        taubinme = stats.binned_statistic(xalphame[indme], taume[indme], statistic='mean', bins=15)
        tau_bootstrap.append(taubin[0] - taubinme[0])


    flux_stack = np.vstack(np.transpose(tau_bootstrap))

    flux_bootstrap_tau = np.dot(flux_stack, np.transpose(flux_stack)) / (times - 1)

    return flux_bootstrap_tau, flux_stack

def main():
#loading spectra without BAL features

   db_file = '/Users/Jenny/documents/pycodes/specdb/specdb/data/DB/IGMspec_DB_v03.hdf5'
   sdb = SpecDB(db_file=db_file)
   sdb14 = sdb['BOSS_DR14']
   meta_tbl = sdb['BOSS_DR14'].meta

   times = 10000
#data = np.loadtxt("Lyman_alpha_result_deltatau_1.txt")

   dr14index = np.loadtxt('/Users/Jenny/documents/pycodes/specdb/specdb/data/DB/BAL_selection_dr14_z>2.1_new.txt')
   index = []
   for i in range (len(dr14index)):
        index.append(int(dr14index[i]))
   meta_tbl = meta_tbl[(meta_tbl['Z_PIPE'] > 2.1)
                    & (meta_tbl['Z_PIPE_ERR'] > 0.)
                    & (abs(meta_tbl['Z_VI'] - meta_tbl['Z_PIPE']) < 60 / 1216 * meta_tbl['Z_VI'])]
   meta_tbl = meta_tbl[index]
   print(len(meta_tbl))

#gathering data


   for ii in it.chain(np.arange(2.5,4.0,0.1),np.arange(4.0,4.8,0.2)):
#for ii in np.arange(4.4,4.6,0.2):

        if ii < 4.0:


            xalphai,taui= bootstrap_absolutetau(ii, 0.1, 1075, 1150)
            np.savetxt("dr14_co_10000_0.9_testing_tau_1075_toltestdeltatau_"+str(ii)+"_.txt", np.c_[xalphai,taui], delimiter="  ",
                   newline='\n')
        else:


            xalphai, taui = bootstrap_absolutetau(ii, 0.2, 1075, 1150)
            np.savetxt("dr14_co_10000_0.9_testing_tau_1075_toltestdeltatau_" + str(ii) + "_.txt", np.c_[xalphai, taui],
                   delimiter="  ",newline='\n')



if __name__=='__main__':
    main()