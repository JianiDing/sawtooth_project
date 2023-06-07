import matplotlib.pyplot as plt
plt.style.use(['classic'])
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
from theano.compile.ops import as_op
import theano.tensor as tt
from scipy import stats
from scipy.stats import binned_statistic
from scipy.linalg import block_diag,eigh
plt.rcParams['font.family']='stixgeneral'
plt.rcParams.update({'font.size':16})
from pymc3 import summary
#checking verison of pymc3

#inputting data
#data_base = np.loadtxt("Lyman_alpha_result_balout_z=3.2-3.3_testing.txt")
data_z = np.loadtxt("zratiosnr22.txt")
data_all = np.loadtxt("./dr14_1075-1150_co_10000_0.9_testing_snr2.2_waveflux.txt")
#data_error = np.loadtxt("test1.txt")
#covariance = np.loadtxt("test2.txt")
#gp = np.loadtxt("./jiani0126/gp_dr14_1114.txt")
data_z = data_z[:,0]
print (data_z)
i = range(0, len(data_all))
data_indi_flux = np.split(data_all[i,1], len(data_z))
data_indi_wave = np.split(data_all[i,0], len(data_z))




#picking the data from 2.8 < z < 4.8
data_indi_flux = data_indi_flux[3:]
data_indi_wave = data_indi_wave[3:]

data_z = data_z[3:]
#print (np.min(covariance_split[7]))

#using schaye metal correction
data_indi_tau = [0.127,0.164,0.203,0.251,0.325]
data_redshift_base = [2.0,2.2,2.4,2.6,2.8]
data_error_base = [0.023,0.013,0.009,0.01,0.012]
data_indi_flux_final = np.append(data_indi_flux,np.array(data_indi_tau))
#data_error_final=np.append(data_error_split,np.array(data_error_base))


#arranging error for the Faucher-giguere data
data_error_base = block_diag(np.array(data_error_base))
contrain_co1 = np.zeros((5,5))
indice = np.diag_indices(5)

contrain_co1[indice] = np.array(data_error_base**2)
print (contrain_co1)

#load the covariance matrix
covtol1 = np.loadtxt("dr14_1075-1150_co_10000_0.9_testing_snr2.2_tol_1.txt")


#covtol = np.append(covtol1,covtol2)

covtolsplit = np.split(covtol1,19)

covtest= []
for ii in range(3,len(covtolsplit)):
    covtest.append(np.transpose(np.split(covtolsplit[ii],10000)))

final_cov = np.dot(np.vstack(covtest),np.transpose(np.vstack(covtest)))/(len(np.vstack(covtest))-1)

# costruct the final covariance matrix
covt = []
covt.append(final_cov)
covt.append(contrain_co1)
# costruct the final covariance matrix
cov_final = block_diag(*covt)

print (np.array(cov_final))

# begin running MCMC to find the best-fitted continuum and parameters to describe tau

n = 22
n1 = 12
n2 = n - n1


# assuming function of tau
def func(x, amp, index):
    func = amp * ((1 + x) / (1 + 3.05)) ** index
    return func


# setting up pymc3 model
@as_op(itypes=[tt.dscalar] * (n), otypes=[tt.dvector])
def fc(*para):
    # converting restframe wavelength to absorption redhsift
    # x2 = np.array(data_indi_wave) / np.array(1215.67 / (1+np.array(np.split(data_z, len(data_z))))) - 1
    # xinter = binned_statistic(np.ravel(x2),np.ravel(x2),bins = n2)[0]
    x2 = np.array(data_indi_wave) / np.array(1215.67 / (1 + np.array(np.split(data_z, len(data_z))))) - 1
    x2t = np.sort(np.ravel(x2))
    xinterl = binned_statistic(x2t[x2t < 3.5], x2t[x2t < 3.5], bins=n2 - 2)[0]
    xinterh = binned_statistic(x2t[x2t > 3.5], x2t[x2t > 3.5], bins=2)[0]
    xinter = np.append(xinterl, xinterh)
    tau = interp1d(xinter, para[n1:n], fill_value='extrapolate')
    tau_1 = tau(x2)

    # wavelength for continuum model
    xi = np.linspace(data_indi_wave[5][0], data_indi_wave[5][len(data_indi_wave[5]) - 1], n1)

    result_final = interp1d(xi, np.array(para[:n1]), kind='cubic')(data_indi_wave[5]) * np.exp(-np.array(tau_1))

    # adding the constraint from Faucher2008
    x1 = np.array(data_redshift_base)
    # print (result_final)
    result_final1 = np.append(result_final, tau(x1))

    return np.ravel(result_final1)


if __name__ == '__main__':
    with pm.Model() as model:
        # the prior distribution of all the parameters
        para = []
        n = 22
        n1 = 12
        n2 = n - n1
        k1 = 0.18
        for i in range(0, n1):
            para.append(pm.Uniform(str(i), 0.6, 2.8))
        for i in range(n1, n):
            para.append(pm.Uniform(str(i), k1, k1 + 0.6))
            k1 = k1 + 0.045
        # print (para)

        fc = fc(*para)
        # fit the model to observed data
        pymc_data = pm.MvNormal(str('output_data'), mu=fc, cov=np.array(cov_final),
                                observed=np.ravel(np.array(data_indi_flux_final)))

        trace = pm.sample(90000, burn=6000, tune=5000, chain=1, cores=2)

pm.traceplot(trace)
plt.show()


def plot_MCMC_model(n, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    n = n
    n1 = 12
    n2 = n - n1

    mean = []

    for i in range(0, n):
        mean.append(summary(trace)['mean'][i])

    x2 = np.array(data_indi_wave) / np.array(1215.67 / (1 + np.array(np.split(data_z, len(data_z))))) - 1
    x2t = np.sort(np.ravel(x2))
    xinterl = binned_statistic(x2t[x2t < 3.5], x2t[x2t < 3.5], bins=n2 - 2)[0]
    xinterh = binned_statistic(x2t[x2t > 3.5], x2t[x2t > 3.5], bins=2)[0]
    xinter = np.append(xinterl, xinterh)
    tau = interp1d(xinter, mean[n1:n], fill_value='extrapolate')
    tau_1 = tau(x2)

    # wavelength for continuum model
    xi = np.linspace(data_indi_wave[5][0], data_indi_wave[5][len(data_indi_wave[5]) - 1], n1)

    result_final = interp1d(xi, np.array(mean[:n1]), kind='cubic')(data_indi_wave[5]) * np.exp(-np.array(tau_1))

    # adding the constraint from Faucher2008
    x1 = np.array(data_redshift_base)
    # print (result_final)
    result_model = np.append(result_final, tau(x1))

    fig, ax = plt.subplots(3, 2, figsize=(18, 20), gridspec_kw={'wspace': 0.15, 'hspace': 0.2}, sharex=False,
                           sharey=False)

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.9)
    k = 10

    degrees = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3)]
    for index in range(ax.shape[0]):
        for jndex in range(ax.shape[1]):
            ax[index][jndex].plot(np.array(xdata[k] / 1215.67 * (1 + data_z[k])) - 1, ydata[k],
                                  label=str('{:.2f}'.format(data_z[k])) + ' data')
            ax[index][jndex].plot(np.array(xdata[k] / 1215.67 * (1 + data_z[k])) - 1,
                                  result_model[k * 282:(k + 1) * 282], label='model')

            ax[index][jndex].set_xlabel('redshift')
            ax[index][jndex].set_ylabel('flux')
            ax[index][jndex].legend(prop={'size': 20})

            k = k + 1

    plt.savefig('model_fittednew_2.pdf')

    plt.show()
    plt.plot(data_redshift_base, tau(x1), label='mean model')
    plt.plot(data_redshift_base, data_indi_tau, label='data')
    plt.errorbar(data_redshift_base, data_indi_tau, yerr=np.array([0.018, 0.013, 0.009, 0.01, 0.012]), capsize=2.0,
                 label='faucher2008 data')
    plt.savefig('model_fitted4.png')
    plt.show()

    # plt.fill_between(xdata, mu - sig, mu + sig, color='lightgray')



    return xi, xinter, result_model, tau_1, tau, interp1d(xi, np.array(mean[:n1]), kind='cubic')(data_indi_wave[5])


xcon,xtau,model,taut,tau,conti = plot_MCMC_model(22,data_indi_wave, data_indi_flux, trace)

#saving the continuum function
np.savetxt("conti_model_new_1125",conti, delimiter = " ", newline = '\n')

#saving the continuum function
np.savetxt("tau_model_new_1125",taut,delimiter = " ", newline = '\n')

np.savetxt("taux_model_new_1125",xtau,delimiter = " ", newline = '\n')

np.savetxt("conx_model_new_1125",xcon,delimiter = " ", newline = '\n')






