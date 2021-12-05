# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:16:08 2021

@author: Spencer Dooley and Nik Cobb
"""
#import necessary modules and packages
import load_p300_data
import plot_p300_erps
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import fdr_correction

np.random.seed(123)


def calculate_erps(subject=3):
    '''
    Function that calls modules from plot_p300_erps.py 
    
    Parameters
    ----------
    subject : int32, optional
        The subject number we are looking at. The default is 3.

    Returns
    -------
    target_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was detected
    nontarget_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was not detected
    erp_times : 1-D array of float64 of size 384
        array of times beginning at the epoch start time and ending at the epoch end time by the step determined by the samples per epoch.
    eeg_epochs : 3-D array of float64 of size 900x384x8
        contains the epoched data, each epoch contains the number of samples per epoch by the number of channels.
    is_target_event : 1-D boolean array
        A boolean array of weather or not the event that occured was a target event
    target_epochs : 3-D Array of size 150x384x8
        contains the epoched data for target trials, each epoch contains the number of samples per epoch by the number of channels.
    nontarget_epochs : 3-D Array of size 750x384x8
        contains the epoched data for nontarget trials, each epoch contains the number of samples per epoch by the number of channels.

    '''
    #calling functions from other modules to get data and arrays we need
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_train_eeg(subject)
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp = plot_p300_erps.get_erps(eeg_epochs, is_target_event)
    #calculate the target and nontarget epochs using boolean indexing
    target_epochs = eeg_epochs[is_target_event]
    nontarget_epochs = eeg_epochs[~is_target_event]
    return target_erp, nontarget_erp, erp_times, eeg_epochs, target_epochs, nontarget_epochs


def calc_intervals(target_erps, nontarget_erps, target_epochs, nontarget_epochs):
    '''
    Function that calculate the 95% confidence intervals for target and non target erps

    Parameters
    ----------
    target_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was detected
    nontarget_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was not detected
    target_epochs : 3-D Array of size 150x384x8
        contains the epoched data for target trials, each epoch contains the number of samples per epoch by the number of channels.
    nontarget_epochs : 3-D Array of size 750x384x8
        contains the epoched data for nontarget trials, each epoch contains the number of samples per epoch by the number of channels.


    Returns
    -------
    target_confidence_interval : Array of arrays each of size 384x1
         The first array is an array of the lower bounds for the target confidence intervals and the second array holds the upper bounds.
    nontarget_confidence_interval :  Array of arrays each of size 384x1
         The first array is an array of the lower bounds for the non target confidence intervals and the second array holds the upper bounds.

    '''
    #calculate the standard error for both target and non target epochs across axis=0
    target_std_err = (np.std(target_epochs, axis=0))/np.sqrt(150)
    nontarget_std_err = (np.std(nontarget_epochs, axis=0))/np.sqrt(750)
    
    #calculate target and nontarget confidence intervals (value+/-2*standard error) arrays are [[lower bounds],[upper bounds]]
    target_confidence_interval = [target_erps-2*target_std_err, target_erps+2*target_std_err]
    nontarget_confidence_interval = [nontarget_erps-2*nontarget_std_err, nontarget_erps+2*nontarget_std_err]

    #return our intervals
    return target_confidence_interval, nontarget_confidence_interval


                           

def calculate_p_vals(eeg_epochs, target_epochs, nontarget_epochs, target_erp, nontarget_erp):
    '''
    Function that calculates p values for bootstrapped erp differences

    Parameters
    ----------
    eeg_epochs : 3-D array of float64 of size 900x384x8
        contains the epoched data, each epoch contains the number of samples per epoch by the number of channels.
    target_epochs : 3-D Array of size 150x384x8
        contains the epoched data for target trials, each epoch contains the number of samples per epoch by the number of channels.
    nontarget_epochs : 3-D Array of size 750x384x8
        contains the epoched data for nontarget trials, each epoch contains the number of samples per epoch by the number of channels.


    Returns
    -------
    pvals: Array of size 384x8
        Array containg the p values at each time point at each channel.

    '''
    def bootstrap(eeg_epochs, target_epochs, nontarget_epochs):
        '''
        Function that bootstraps target and non target erps

        Parameters
        ----------
        eeg_epochs : 3-D array of float64 of size 900x384x8
            contains the epoched data, each epoch contains the number of samples per epoch by the number of channels.
        target_epochs : 3-D Array of size 150x384x8
            contains the epoched data for target trials, each epoch contains the number of samples per epoch by the number of channels.
        nontarget_epochs : 3-D Array of size 750x384x8
            contains the epoched data for nontarget trials, each epoch contains the number of samples per epoch by the number of channels.

        Returns
        -------
        sample_erp_target : Array of size 384x8
            Our bootstrapped sampled target erp.
        sample_erp_nontarget : Array of size 384x8
            Our bootstrapped sampled nontarget erp.

        '''
        #create arrays of random numbers from 0 to 900 (150 random numbers for randomly sampled targets and 750 for randomly sampled nontargets)
        random_sample_target = np.random.randint(0,len(eeg_epochs),size=len(target_epochs))
        random_sample_nontarget = np.random.randint(0,len(eeg_epochs),size=len(nontarget_epochs))
        
        #using those random number arrays randomly sample from the 900 eeg epochs 150 'target' and 750 'nontarget
        random_epochs_target = eeg_epochs[random_sample_target, :, :]
        random_epochs_nontarget = eeg_epochs[random_sample_nontarget, :, :]
        
        #to get our sample erps, take the mean across our randomly sampled epochs
        sample_erp_target = random_epochs_target.mean(0)
        sample_erp_nontarget = random_epochs_nontarget.mean(0)
        
        #return our bootstraped erps 
        return sample_erp_target, sample_erp_nontarget
    
    
    def bootstrapStat():
        '''
        Function to calculate bootstrap differences
    
        Returns
        -------
        bootstrap_difference: Array of size 3000x384x8
            3-D array containing 3000 bootstrapped differences at each time point for each channel
    
        '''          
        #call our bootstrap function
        target_erp_bootstrapped, nontarget_erp_bootstrapped = bootstrap(eeg_epochs, target_epochs, nontarget_epochs)
        #calculate our bootstrapped differences 
        bootstrap_difference = target_erp_bootstrapped - nontarget_erp_bootstrapped
        return bootstrap_difference  
    
    bootstrapped_diffs = np.array([bootstrapStat() for _ in range(3000)])
    #calculate the actual differences at each timpoint on each channel
    difference_array = np.absolute(target_erp-nontarget_erp)
    
    #calc our p values (the number of times our bootsraped difference is greater than our actual observed difference divided by 3000 across all trials at each time point)
    pvals = np.sum(bootstrapped_diffs>difference_array, axis=0)/3000
    return pvals

def get_fdr_pvals(pvals):
    '''
    Function that uses FDR correction for multiple comparisons on our array of p values

    Parameters
    ----------
    pvals : Array of size 384x8
        Array containing p values at each timepoint for each channel.

    Returns
    -------
    fdr_pval : Array of size 384x8
        Array containing FDR corrected p values at each timepoint for each channel.

    '''
    #calculate our fdr corrected p values
    fdr_pval = fdr_correction(pvals)
    return fdr_pval



def plot_vals_intervals(erp_times, target_erps, nontarget_erps, target_confidence_interval, nontarget_confidence_interval, fdr_pval, subject):
    '''
    Function that plots the target and non target erps and their confidence intervals as well as if the difference was significant at that time point

    Parameters
    ----------
    target_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was detected
    nontarget_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was not detected
    target_confidence_interval : Array of arrays each of size 384x1
         The first array is an array of the lower bounds for the target confidence intervals and the second array holds the upper bounds.
    nontarget_confidence_interval :  Array of arrays each of size 384x1
         The first array is an array of the lower bounds for the non target confidence intervals and the second array holds the upper bounds.
    fdr_pval : Array of size 384x8
        An array of our FDR corrected p values.
    subject : int32
        The subject we are looking at.

    Returns
    -------
    None.

    '''
    #for every channel 
    for channel_index in range(1, int(np.size(target_erps, axis = 1))+1):
        
        #set up subplot structure, inserting new graph at indexed location
        plt.subplot(3, 3, channel_index)
        #plot the target 
        plt.plot(erp_times, target_erps[:, channel_index-1], color='blue', label='Target ERP')
        #use fill between between to plot the confidence intervals 
        plt.fill_between(erp_times,y1=target_confidence_interval[0][:,channel_index-1],y2=target_erps[:, channel_index-1], color='blue', alpha=.2)
        # plt.fill_between(erp_times,y1=target_confidence_interval[0][:,channel_index-1],y2=target_confidence_interval[1][:,channel_index-1], color='blue', alpha=.2)
        plt.fill_between(erp_times,y1=target_confidence_interval[1][:,channel_index-1],y2=target_erps[:, channel_index-1], color='blue', alpha=.2, label='95% CI')
        
        #plot the non target data for the correct channel
        plt.plot(erp_times, nontarget_erps[:, channel_index-1], color='orange', label='Non-Target ERP')
        #use fill between between to plot the confidence intervals 
        plt.fill_between(erp_times,y1=nontarget_confidence_interval[0][:,channel_index-1],y2=nontarget_erps[:, channel_index-1], color='orange', alpha=.2)
        plt.fill_between(erp_times,y1=nontarget_confidence_interval[1][:,channel_index-1],y2=nontarget_erps[:, channel_index-1], color='orange', alpha=.2, label='95% CI')
        
        #add our dotted lines at orgin
        plt.axhline(y=0,  color = 'black', linestyle=':')
        plt.axvline(x=0,  color = 'black', linestyle=':')
        
        #plot the fdr corrected pvals when they are less than .05 at the correct time point
        plt.scatter(x=erp_times[np.where(fdr_pval[1][:,channel_index-1]<.05)], y=np.zeros((len(erp_times[np.where(fdr_pval[1][:,channel_index-1]<.05)]))), color='black', label='FDR p-value < .05')
        #add axis and labels
        plt.title(f'Channel {channel_index-1}')
        plt.ylabel("Voltage uV")
        plt.xlabel("Time from flash onset(s)")

    #format plots
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((16, 10))
    plt.savefig(f'P300_S{subject}_erp.png')
    fig.savefig(f'P300_S{subject}_erp.png', dpi=500)
    plt.clf()

NUM_TIME_POINTS=384
NUM_CHANNELS = 8
#set up 3-D array to count what subjects are significant at what time points on what channels
significant_fdr_subject = np.zeros((8,NUM_TIME_POINTS,NUM_CHANNELS))

for subject_index, subject in enumerate(range(3,11)):
    # get target_erps, nontarget_erps, erp_times, eeg_epochs, is_target_event, target_epochs, nontarget_epochs
    target_erps, nontarget_erps, erp_times, eeg_epochs, target_epochs, nontarget_epochs=calculate_erps(subject)
    # get target_confidence_interval, nontarget_confidence_interval
    target_confidence_interval, nontarget_confidence_interval= calc_intervals(target_erps, nontarget_erps, target_epochs, nontarget_epochs)
    pvals = calculate_p_vals(eeg_epochs, target_epochs, nontarget_epochs, target_erps, nontarget_erps) 
    #get fdr p values
    fdr_pval = get_fdr_pvals(pvals)
    #determine weather subject was significant (using fdr p value) at given time point on given channel
    significant_fdr_subject[subject_index, :, :] = fdr_pval[0]
    #plot teh intervals and significance
    print(subject)
    plot_vals_intervals(erp_times, target_erps, nontarget_erps, target_confidence_interval, nontarget_confidence_interval, fdr_pval, subject)
    

#calcualte the number of subjects who were significant at each time point on each channel
num_subjects_significant_by_time = np.sum(significant_fdr_subject, axis=0)

for channel_index in range(1, NUM_CHANNELS+1):
    #set up subplot structure, inserting new graph at indexed location
    ax=plt.subplot(3, 3, channel_index)
    #plot the target and non target data for the correct channel
    plt.plot(erp_times, num_subjects_significant_by_time[:, channel_index-1], color='blue')
    # set the y range to 0 to the max number of subjects that were significant (so y axis can be shared)
    ax.set_ylim(0, np.amax(num_subjects_significant_by_time))
    
    
    plt.title(f'Channel {channel_index-1}')
    plt.ylabel("Number of Subjects Significant")
    plt.xlabel("Time from flash onset(s)")
    
    
plt.savefig('P300_Num_Significant_subjects.png')







