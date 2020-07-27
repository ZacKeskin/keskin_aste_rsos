import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *


## Set Parameters
LAG = 1      
N_SHUFFLES = 100
WINDOW_SIZE = {'MS':24}
WINDOWS_STRIDE = {'M':1}
n_bins = 6

## Shortcuts for Column Names
SM = 'positive_volume_en'
Return = 'close'

## Choose Crypto File and load data
CRNC = 'ETH'
path = os.path.join(os.getcwd(), 'data', CRNC + '_merged.txt')
DF = pd.read_csv(path, index_col='time')[[SM, Return]]
DF.index = pd.to_datetime(DF.index, unit='s') 

## Forward-Fill Missing Data
DF = DF[[SM,Return]].replace(0, np.nan).fillna(method='ffill')


## Difference the data using log-returns
DF[SM] = np.log(DF[SM]).diff()
DF[Return] = np.log(DF[Return]).diff()
DF = DF[1:-1]


## Initialise TE Objects
linearPC = TransferEntropy(DF = DF,
                           endog = Return,
                           exog = SM,
                           lag = LAG,
                           window_size = WINDOW_SIZE,
                           window_stride = WINDOWS_STRIDE)

nonlinearPC = TransferEntropy(DF = DF,
                              endog = Return,
                              exog = SM,
                              lag = LAG,
                              window_size = WINDOW_SIZE,
                              window_stride = WINDOWS_STRIDE)


## Calculate marginal equipartion
DF = LaggedTimeSeries(DF, LAG).df
auto = AutoBins(DF, LAG)
equal_bins = auto.equiprobable_bins(n_bins)

## Calculate Linear TE using Granger Causality
linearPC.linear_TE(n_shuffles=N_SHUFFLES)
nonlinearPC.nonlinear_TE(pdf_estimator='histogram',
                         bins = equal_bins,
                         n_shuffles=N_SHUFFLES)


## Plot Results
fig, axes = plt.subplots(figsize=(12, 6.5), nrows=2, ncols=2) 

## TE_XY
TE_XY = axes[0][0].plot(nonlinearPC.results['TE_XY'], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lTE_XY = axes[0][0].plot(linearPC.results['TE_linear_XY'], 
                        color=(10/255., 10/255., 10/255.), 
                        linewidth=0.8,label='Linear TE',
                        linestyle='dotted')


## TE_YX
TE_YX = axes[0][1].plot(nonlinearPC.results['TE_YX'],
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lTE_YX = axes[0][1].plot(linearPC.results['TE_linear_YX'], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8,label='Linear TE',
                        linestyle='dotted')


## Z_Scores_XY
Z_XY = axes[1][0].plot(nonlinearPC.results['z_score_XY'], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lZ_XY = axes[1][0].plot(linearPC.results['z_score_linear_XY'], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Linear TE',
                        linestyle='dotted')


## Z_Scores_YX
Z_XY = axes[1][1].plot(nonlinearPC.results['z_score_YX'], 
                        color=(31/255., 119/255., 180/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dashdot')

lZ_XY = axes[1][1].plot(linearPC.results['z_score_linear_YX'], 
                        color=(10/255., 10/255., 10/255.),
                        linewidth=0.8, label='Non-linear TE',
                        linestyle='dotted')



## Format plots
axes[1][0].set_ylim(0.003)
#axes[0][0].set_ylim(axes[0][1].get_ylim())
axes[0][1].set_ylim(axes[0][0].get_ylim())

axes[1][0].set_ylim(axes[1][1].get_ylim())
axes[1][0].set_ylim(0,50)
axes[1][1].set_ylim(0,50)

## Format Legend
lines = TE_XY + lTE_XY
labels = labs = [l.get_label() for l in lines]


for ax in axes:
    ax[0].legend(lines, labels, loc=0, prop={'size': 7})
    ax[1].legend(lines, labels, loc=0, prop={'size': 7})

axes[0][0].set_ylabel('Transfer Entropy (bits)',fontsize=9)
axes[0][1].set_ylabel('Transfer Entropy (bits)',fontsize=9)
axes[1][0].set_ylabel('Significance (Z-Score)',fontsize=9)
axes[1][1].set_ylabel('Significance (Z-Score)',fontsize=9)

axes[0][0].set_title(r'Transfer Entropy from $Sentiment \rightarrow Returns$',fontsize=10)
axes[0][1].set_title(r'Transfer Entropy from $Returns \rightarrow Sentiment$',fontsize=10)
axes[1][0].set_title(r'Significance of TE from $Sentiment \rightarrow Returns$',fontsize=10)
axes[1][1].set_title(r'Significance of TE from $Returns \rightarrow Sentiment$',fontsize=10)


for ax in axes.flatten():
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlabel(str(WINDOW_SIZE.get('MS')) + '-Month Windows Ending', fontsize=9)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        

plt.subplots_adjust(left  = 0.1,    # the left side of the subplots of the figure
                    right = 0.9,    # the right side of the subplots of the figure
                    bottom = 0.15,  # the bottom of the subplots of the figure
                    top = 0.89,     # the top of the subplots of the figure
                    wspace = 0.5,   # the amount of width reserved for blank space between subplots
                    hspace = 0.65   # the amount of height reserved for white space between subplots
                    )

plt.show()

