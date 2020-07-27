import numpy as np
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *


## Set Parameters
alpha = 0.5
N = 15000
DATALAG = 6
N_SHUFFLES = 10
N_LAGS = 30
lag_search_range = range(1, N_LAGS)
N_OBSERVATIONS = 3
n_bins = 6


## Prepare arrays to store results
lTE_array = np.zeros(shape=(N_OBSERVATIONS, N_LAGS-1, 2))
lZ_array =  np.zeros(shape=(N_OBSERVATIONS, N_LAGS-1, 2))
TE_array = np.zeros(shape=(N_OBSERVATIONS, N_LAGS-1, 2))
Z_array =  np.zeros(shape=(N_OBSERVATIONS, N_LAGS-1, 2))


## Generate Multiple Realisations and Take Average Values
for i, observation in enumerate(range(N_OBSERVATIONS)):
    print('Realisation: ' + str(i))

    ## Stochastic Parameterisation to Average over Multiple Realisations
    S1 = np.random.normal(200, 10)
    S2 = np.random.normal(200, 10)
    mu1 = np.random.normal(0.1, 0.1)
    mu2 = np.random.normal(0.1, 0.1)
    sigma1 = np.random.normal(0.25, 0.01)
    sigma2 = np.random.normal(0.1, 0.01)

    
    ## Loop Over Defined Lag Values
    for j, LAG in enumerate(lag_search_range):    
        print('Lag: ' + str(LAG))

        ## Generate coupled Gaussian Random Walks
        DF = coupled_random_walks(  S1 = S1, S2 = S2, T = np.sqrt(N) * 1E-6, 
                                    N = N + LAG, mu1 = mu1, mu2 = mu2, 
                                    sigma1 = sigma1, sigma2 = sigma2, 
                                    alpha = alpha, epsilon = 0,
                                    lag = DATALAG, seed=None)

        ## Difference data using log-returns
        DF = np.log(DF).diff().iloc[LAG:]
        
        ## Initialise Object to Calculate TE
        TE = TransferEntropy(DF = DF,
                             endog = 'S2',
                             exog = 'S1',
                             lag = LAG
                            )
                    
        ## Calculate marginal equipartion
        AB = AutoBins(DF, LAG)
        equal_bins = AB.equiprobable_bins(n_bins)

        ## Update results with linear TE from X->Y and from Y->X
        TE.linear_TE(n_shuffles=N_SHUFFLES)
        lTE_array[i,j,:] = (TE.results['TE_linear_XY'], TE.results['TE_linear_YX'])  
        lZ_array[i,j,:] = (TE.results['z_score_linear_XY'], TE.results['z_score_linear_YX'])
        
        ## Update results with non-linear TE from X->Y and from Y->X
        TE.nonlinear_TE(pdf_estimator='histogram', bins=equal_bins, n_shuffles=N_SHUFFLES)
        TE_array[i,j,:] = (TE.results['TE_XY'], TE.results['TE_YX'])  
        Z_array[i,j,:] = (TE.results['z_score_XY'], TE.results['z_score_YX'])
            

## Plot Linear and Non-Linear TE
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12,6))

## TE
TE0 = axes[0].plot(lag_search_range, 
                   TE_array.mean(axis=0)[:,0], 
                   color=(31/255., 119/255., 180/255.),
                   linewidth=0.8, label='Non-linear TE',
                   linestyle='dashdot')

lTE0 = axes[0].plot(lag_search_range, 
                    lTE_array.mean(axis=0)[:,0], 
                    color=(10/255., 10/255., 10/255.),
                    linewidth=0.8, label='Linear TE',
                    linestyle='dotted')


## Z_Scores
z0 = axes[1].plot(lag_search_range, 
                  Z_array.mean(axis=0)[:,0],
                  color=(31/255., 119/255., 180/255.),
                  linewidth=0.8, label='Non-linear TE',
                  linestyle='dashdot')

lz0 = axes[1].plot(lag_search_range, 
                   lZ_array.mean(axis=0)[:,0], 
                   color=(10/255., 10/255., 10/255.),
                   linewidth=0.8, label='Linear TE',
                   linestyle='dotted')


## Format Plots
for a, axis in enumerate(axes):
    axis.set_xticks([0, 5, 10, 15, 20, 25, 30])
    axis.set_ylabel('Transfer Entropy (bits)')
    axis.legend(['Non-linear TE', 'Linear TE'], fontsize=7)

    for label in axis.xaxis.get_majorticklabels():
        label.set_fontsize(9) 
    for label in axis.yaxis.get_majorticklabels():
        label.set_fontsize(9)
    for label in axis.yaxis.get_minorticklabels():
        label.set_fontsize(9)
    axis.grid(False)

axes[0].set_title(r'TE with $\alpha=$' + str(alpha), fontsize=11)
axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
axes[1].set_yscale('log')
axes[1].set_ylabel('Significance (Z-score)')
axes[1].set_xlabel(r'lag = $k$')

plt.subplots_adjust(    left  = 0.1,    # the left side of the subplots of the figure
                        right = 0.9,    # the right side of the subplots of the figure
                        bottom = 0.1,   # the bottom of the subplots of the figure
                        top = 0.85,     # the top of the subplots of the figure
                        wspace = 0.5,   # the amount of width reserved for blank space between subplots
                        hspace = 0.5    # the amount of height reserved for white space between subplots
                        )
plt.show()
