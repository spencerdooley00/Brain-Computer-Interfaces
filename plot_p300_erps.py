# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 12:43:38 2021

@author: Spencer Dooley and Kristina Sitcawich
"""
#%% cell 1 - import packages
# import packages needed for this lab
import numpy as np
import matplotlib.pyplot as plt


#%% cell 2 - extract event times and labels

def get_events(rowcol_id, is_target):
    '''
    get_events gives us arrays of the indices of when an event takes place, and weather or not 
    that event had a target letter involved

    Parameters
    ----------
    rowcol_id : array of int32
        integer array that shows the event type.
    is_target : boolean array
        boolean array representing if the current row/col flashed includes the target letter or not..

    Returns
    -------
    event_sample : 1-D array
        Array of indices where an event occured
    is_target_event : 1-D boolean array
        A boolean array of weather or not the event that occured was a target event

    '''
    # we want the indices of rowcol_id WHERE the DIFFERENCE is greater than 0
    event_sample = np.where(np.diff(rowcol_id)>0)[0]+1
    # now we want to know if, at each event, was it a target event
    is_target_event = is_target[event_sample]
    return event_sample, is_target_event

#%% cell 3 - extract the epochs

def epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=-0.5, epoch_end_time = 1):
    '''
    Function that epochs the eeg data and gives us a 3-D epoch array

    Parameters
    ----------
    eeg_time : 1-D array of float64 of size 61866
        Array of time of each sample taken (in seconds).
    eeg_data : 2-D array of float64 of size 61866x8
        Array of the raw EEG data over 8 different channels.
    event_sample : array of int64 of size 900
        array of the indices when events took place.
    epoch_start_time : int
        How may seconds before the event do we want to start an epoch. The default is -0.5.
    epoch_end_time : int
        How may seconds after the event do we want to end an epoch. The default is 1.

    Returns
    -------
    eeg_epochs : 3-D array of float64 of size 900x384x8
        contains the epoched data, each epoch contains the number of samples per epoch by the number of channels.
    erp_times : 1-D array of float64 of size 384
        array of times beginning at the epoch start time and ending at the epoch end time by the step determined by the samples per epoch.

    '''
    # calculate the samples per second by dividing 1 by the sampling frquency. Since the first time is 0, we can get this frequency by indexing at 1
    samples_per_second = 1/eeg_time[1]
    # calculate the seconds per epoch
    seconds_per_epoch = epoch_end_time-epoch_start_time
    # calculate the samples per epoch with the above two calculations - seconds will cancel using dimensional analysis
    samples_per_epoch = int(samples_per_second * seconds_per_epoch)
    
    # initialize empty np array
    eeg_epochs = np.array([])
    # for every sample in event sample, we want the epoch to range from the start epoch index to the end epoch index over all 8 channels
    for sample_index in event_sample:
        # to get start and end epoch indices we find the number of indices that corresponds to the start/end time from zero (ex. -0.5 is 192 indices before teh target) and add that to the initial sample index
        start_epoch = int(sample_index+(samples_per_second/(1/epoch_start_time)))
        end_epoch = int(sample_index+(samples_per_second/(1/epoch_end_time)))
        # epoch eeg_data over correct time range
        epoch_data = eeg_data[start_epoch:end_epoch, :]
        # append the epoch to the eeg_epochs array from before
        eeg_epochs = np.append(eeg_epochs, epoch_data)
    # reshape the eeg_epochs array to give us 3-D array of shape (number of epochs, sample per epoch, number of channels)
    eeg_epochs = np.reshape(eeg_epochs, [len(event_sample), samples_per_epoch, int(np.size(eeg_data, axis = 1))])
    # use np.arange to get an array starting at epoch start time and ending at epoch end time of step (total time in seconds of time range/samples/epoch)
    erp_times=np.arange(epoch_start_time, epoch_end_time, (epoch_end_time-epoch_start_time)/samples_per_epoch)
    return eeg_epochs, erp_times
#%% cell 4 - get ERPs
def get_erps(eeg_epochs, is_target_event):
    '''
    This function calculates target and non target ERPs by indexing eeg_epochs based on if an 
    event was a target event or non target event
    



    Parameters
    ----------
    eeg_epochs : 3D array of float64
        A 3D array of epoched eeg_data where each epoch is a new layer in the array
    is_target_event : 1D array of bool
        An 1D array where each sample is labeled either True or False based on whether the target was 
        detected (True) or was not detected (False)


    Returns
    -------
    target_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was detected
    nontarget_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target was not detected






    '''
    # target epochs is the eeg epochs where a target event occured (where is_target event is True)
    target_epochs = eeg_epochs[is_target_event]
    # non target epochs is the eeg epochs where an event but not a target event occured (where is_target event is false)
    nontarget_epochs = eeg_epochs[~is_target_event]
    
    # the tagret and non target erps are calculated by averaging the correspnding value across all epochs to find the mean response on each channel over all epochs for both target and nontarget events  
    target_erp = np.mean(target_epochs, axis=0)
    nontarget_erp = np.mean(nontarget_epochs, axis=0)
    
    return target_erp, nontarget_erp
#%% cell 5 - Plot ERPs
def plot_erps(target_erp, nontarget_erp, erp_times):
    '''
    

    This function plots the average of all eeg_epochs of each each eeg channel as a subplot where each plot 
    contains eeg data where the target_erps was detected in addition to eeg data where the target_erps was NOT detected
    


    Parameters
    ----------
    target_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target letter was flashed during an event
    nontarget_erps : 2D array of float64
        A 2D array consisting of eeg_data that corresponds to when a target letter was not flashed during the event
    erp_times : 1D array of float64
        1-D array of times beginning at epoch start time and ends at epoch end time where the step is
        determined by the samples per epoch.


    Returns
    -------
    None.

    '''



    # we want to populate a 3x3 subplot matrix with the corresponding channel, use a for loop to iterate over subplot location
    for channel_index in range(1, int(np.size(target_erp, axis = 1))+1):
        #set up subplot structure, inserting new graph at indexed location
        plt.subplot(3, 3, channel_index)
        #plot the target and non target data for the correct channel
        plt.plot(erp_times, target_erp[:, channel_index-1])
        plt.plot(erp_times, nontarget_erp[:, channel_index-1])
        # give plot a title, dotted lines at (0,0), and axis titles
        plt.title(f'Channel {channel_index-1}')
        plt.axhline(y=0,  color = 'black', linestyle=':')
        plt.axvline(x=0,  color = 'black', linestyle=':')
        plt.ylabel("Voltage uV")
        plt.xlabel("Time from flash onset(s)")
    #only giving the last graph a legend as shown in directions, and to not crowd the plots    
    plt.legend(['target', 'non-target'])


#%% Cell 6- discussion

'''
1. Why do we see repeated up-and-down patterns on many of the channels?

The repeated up and down patterns on the channels show the changes in electrical 
activity in the brain. The different patterns in different channels show the neurons 
firing in specific regions of the brain in response to a certain activity 
(in this case, the rows and columns of letters being flashed on a screen)


2. Why are they more pronounced in some channels but not others

Since each channel we see in the data is representing a differend part/region 
of the brain. Each region responds differently to different stimuli 
(what we are trying to decode!), thus this is why some channels have more 
pronounced patterns than others, these brain regions are responding stronger

3. Why does the voltage on some of the channels have a positive peak around 
half a second after a target flash?

This is most likely because of the P300 response, as this response occurs 
about half of a second after an oddball task

4. Which EEG channels (e.g. Cz) do you think these 
(the ones described in the last qestion) might be and why?
Looking at the plots output, channels 2, 3, 5, and 6 all have a positive peak 
around half a second after the target was flashed, fitting this description in
the last question.




'''

 




  


