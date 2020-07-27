import numpy as np
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *


## Set Parameters
LAG = 1      
N = 15000
N_SHUFFLES = 100
N_OBSERVATIONS = 10
N_BINS = 6
N_ALPHAS = 20
ALPHAS = np.linspace(0, 0.9, N_ALPHAS)
epsilon = 0.4
T = 10


## Prepare arrays to store results
lTE = np.zeros(shape=(N_ALPHAS, N_OBSERVATIONS, 2))
TE = np.zeros(shape=(N_ALPHAS, N_OBSERVATIONS, 2))
Z =  np.zeros(shape=(N_ALPHAS, N_OBSERVATIONS, 2))
lZ = np.zeros(shape=(N_ALPHAS, N_OBSERVATIONS, 2))

## Loop over all alphas
for i, SIMILARITY in enumerate(ALPHAS):

    ## Generate Multiple Realisations and Take Average Values
    for j in range(N_OBSERVATIONS):
        print('\n\n Alpha: '  + str(SIMILARITY) + '  Time Series #' + str(j) + ' with ' + str(N_SHUFFLES) + ' shuffles')
        
        ## Generate coupled time series under a logistic map
        S1 = np.random.normal(0.3, 0.1)
        S2 = np.random.normal(0.5, 0.15)

        DF = coupled_logistic_map(S1, S2, T, N, SIMILARITY, epsilon)

        ## Difference data using % change
        DF = DF.pct_change().iloc[1:]

        ## Calculate Linear TE using Geweke-Granger Causality Formula
        PC = TransferEntropy(DF = DF,
                             endog = 'S2',
                             exog = 'S1',
                             lag = LAG)
        PC.linear_TE(n_shuffles = N_SHUFFLES)

        ## Update results with linear TE from X->Y and from Y->X
        (lTE[i,j,0], lTE[i,j,1]) = (PC.results['TE_linear_XY'], PC.results['TE_linear_YX'])
        (lZ[i,j,0], lZ[i,j,1]) = (PC.results['z_score_linear_XY'], PC.results['z_score_linear_YX'])

        ## Note equiprobable bins is a misnomer; these bins are a marginal-equiprobable partition
        AB = AutoBins(DF,LAG)
        equal_bins = AB.equiprobable_bins(N_BINS)

        ## Calculate TE using information-theoretic method
        PC.nonlinear_TE(pdf_estimator = 'histogram',
                        bins = equal_bins,
                        n_shuffles = N_SHUFFLES)

        ## Update results with non-linear TE from X->Y and from Y->X
        (TE[i,j,0], TE[i,j,1]) = (PC.results['TE_XY'], PC.results['TE_YX'])
        (Z[i,j,0], Z[i,j,1]) = (PC.results['z_score_XY'], PC.results['z_score_YX'])
            


## Plot Results

fig, axes = plt.subplots(figsize=(12, 6), nrows=2, ncols=2)


## TE_XY
TE_XY = axes[0][0].plot(ALPHAS, TE.mean(axis=1)[:,0], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lTE_XY = axes[0][0].plot(ALPHAS, lTE.mean(axis=1)[:,0], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Linear TE',
                        linestyle='dotted')

## TE_YX
TE_YX = axes[0][1].plot(ALPHAS, TE.mean(axis=1)[:,1],
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lTE_YX = axes[0][1].plot(ALPHAS, lTE.mean(axis=1)[:,1], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Linear TE',
                        linestyle='dotted')

## Z_Scores XY
Z_XY = axes[1][0].plot(ALPHAS, Z.mean(axis=1)[:,0], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lZ_XY = axes[1][0].plot(ALPHAS, lZ.mean(axis=1)[:,0], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Linear TE',
                        linestyle='dotted')


## Z_Scores YX
Z_YX = axes[1][1].plot(ALPHAS, Z.mean(axis=1)[:,1], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lZ_YX = axes[1][1].plot(ALPHAS, lZ.mean(axis=1)[:,1], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Linear TE',
                        linestyle='dotted')

## Format Legend
lines = TE_XY + lTE_XY 
labels = labs = [l.get_label() for l in lines]
axes[0][0].legend(lines, labels, loc=0)
axes[0][1].legend(lines, labels, loc=0)

## Format plots
axes[0][1].set_ylim(axes[0][0].get_ylim())
axes[1][1].set_ylim(axes[1][0].get_ylim())

axes[0][0].set_ylabel('Transfer Entropy (bits)',fontsize=9)
axes[0][1].set_ylabel('Transfer Entropy (bits)',fontsize=9)
axes[1][0].set_ylabel('Significance (Z-Score)',fontsize=9)
axes[1][1].set_ylabel('Significance (Z-Score)',fontsize=9)

axes[0][0].set_title(r'Transfer Entropy from $X \rightarrow Y$',fontsize=10)
axes[0][1].set_title(r'Transfer Entropy from $Y \rightarrow X$',fontsize=10)
axes[1][0].set_title(r'Significance of TE from $X \rightarrow Y$',fontsize=10)
axes[1][1].set_title(r'Significance of TE from $Y \rightarrow X$',fontsize=10)


for ax in axes.flatten():
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xlabel(r'Coupling Strength $\alpha $',fontsize=9)    

plt.subplots_adjust(    left  = 0.1,    # the left side of the subplots of the figure
                        right = 0.9,    # the right side of the subplots of the figure
                        bottom = 0.1,   # the bottom of the subplots of the figure
                        top = 0.85,     # the top of the subplots of the figure
                        wspace = 0.5,   # the amount of width reserved for blank space between subplots
                        hspace = 0.5    # the amount of height reserved for white space between subplots
                        )
plt.show()
