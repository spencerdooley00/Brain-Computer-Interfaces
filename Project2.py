# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:00:24 2021

@author: spencer
"""
import numpy as np
import import_ssvep_data
import matplotlib.pyplot as plt


def get_predicted_labels(data_dict, eeg_epochs_fft, fft_frequencies, low_stimulation_freq, high_stimulation_freq, electrode_of_interest, threshold):
    '''
    Function that takes a given set of epoch ffts and our desired electrode of interest
    and finds the power at two stimulus frequencies of interest and predicts the trial is of the stimulus frequency with a higher power 

    Parameters
    ----------
    data_dict : dict of 6 fields
        Dictionary containing 6 fields, each holding data for an aspect of the eeg experiment.
    eeg_epochs_fft : Array of size (trials, channels, frequencies)
        3-D array holding epoched data in the frequency domain for each trial.
    fft_frequencies : Array of size (frequencies)
        Array containing the frequency corresponding to each column of the Fourier transform data.
    low_stimulation_freq : int32
        integer taken as user input representing the lower stimulation frequency we are investigating.
    high_stimulation_freq : int32
        integer taken as user input representing the higher stimulation frequency we are investigating..
    electrode_of_interest : String
        String representing the name of the channel/electrode we are calculating the accuracies and itrs for.
    threshold : float
        float representing the threshold (the difference between the higher and lower frequency), we will consider to be a correct prediction or not.

    Returns
    -------
    labels_pred : list of length (number of trials)
        List that holds the predictions we made. Each element of list is a string holding either the value '{lower_freq}hz' or '{higher_freq}hz'

    '''
    # calculate power from epoch ffts
    power = np.abs(eeg_epochs_fft)**2
    channels = data_dict['channels']
    # find the index of the frequency closest to the low and high stimulus frequencie
    index_low_freq = np.where(fft_frequencies==fft_frequencies[np.abs(fft_frequencies - low_stimulation_freq).argmin()])[0][0]
    index_high_freq = np.where(fft_frequencies==fft_frequencies[np.abs(fft_frequencies - high_stimulation_freq).argmin()])[0][0]
    
    # initialize empty list of differences in amplitudes
    amplitude_differences=[]
    # for each trial in the actual trials 
    for trial_index in range(len(eeg_epochs_fft)):
        # find index of the electrode we are interested in
        electrode_index = np.where(channels==electrode_of_interest)[0][0]
        # find the power at the low and high frequency
        power_low = power[trial_index, electrode_index,index_low_freq]
        power_high = power[trial_index, electrode_index,index_high_freq]
        # calculate the difference
        difference = power_high - power_low
        # append difference to list of all differences
        amplitude_differences=np.append(amplitude_differences, difference)
        
    # initialize list of predicted labels
    labels_pred = []
    # for each difference in the list of all differences...
    for amplitude_diff in amplitude_differences:
        # if our amplitude_diff is higher than our decision threshold predict high freq
        if amplitude_diff > threshold:
            labels_pred.append(f'{high_stimulation_freq}hz')
        #else predict low freq
        else:
            labels_pred.append(f'{low_stimulation_freq}hz')
    return labels_pred

def get_truth_labels(data_dict):
    '''
    Function that extracts the truth labels from the dictionary of data

    Parameters
    ----------
    data_dict : dict of 6 fields
        Dictionary containing 6 fields, each holding data for an aspect of the eeg experiment.
    
    Returns
    -------
    truth_labels : Array of length (trials)
        Array holding the truth labels for each trial. truth_labels[i] would indicate wheather trial i is of the lower or higher frequency.


    '''
    # use the event types feild to get the truth labels
    truth_labels = data_dict['event_types']
    return truth_labels


def calculate_accuracy(labels_pred, truth_labels):
    '''
    Function that calculates our prediction accuracy from the predicted labels and truth labels

    Parameters
    ----------
    labels_pred : list of length (number of trials)
        List that holds the predictions we made. Each element of list is a string holding either the value '{lower_freq}hz' or '{higher_freq}hz'

    truth_labels : Array of length (trials)
        Array holding the truth labels for each trial. truth_labels[i] would indicate wheather trial i is of the lower or higher frequency.

    Returns
    -------
    accuracy : float 
        Float representing the accuracy of our prediction as the proportion of correctly predicted trials divided by total number of trials.

    '''
    # initialize counter of correct labels
    num_labels_correct=0
    # for each label index and label in the predicted labels...
    for label_index, label in enumerate(labels_pred):
    # if the predicted label matches the truth label, increment correct labels counter by one, else, keep the same        
        if label==truth_labels[label_index]:
            num_labels_correct+=1
        else:
            num_labels_correct=num_labels_correct    
    accuracy=num_labels_correct/len(truth_labels)
    return accuracy

def calculate_itr(accuracy, duration, truth_labels):
    '''
    Function that uses the given ITR formula to calculate ITR given an accuracy and trial duration

    Parameters
    ----------
    accuracy : float 
        Float representing the accuracy of our prediction as the proportion of correctly predicted trials divided by total number of trials.
    duration : float
        Float representing the time window we are using for epoching data.
    truth_labels : Array of length (trials)
        Array holding the truth labels for each trial. truth_labels[i] would indicate wheather trial i is of the lower or higher frequency.


    Returns
    -------
    itr_time : float
        Float which is the ITR in bits per second for the given accuracy and trial duration.

    '''
    # calculate the number of classes present in our data - it will be the number of unique truth labels
    n = len(np.unique(truth_labels))
    p = accuracy
    # use itr formula given, if accuracy = 1, ITR blows up, set equal to 1
    if p == 1: 
        itr_trial=1
    else:
        itr_trial = np.log2(n) + p*np.log2(p) + (1-p) * np.log2((1-p)/(n-1))
    itr_time = itr_trial*(1/duration)
    return itr_time


def test_epoch_limits(data_dict, electrode_of_interest, start_times_to_test, end_times_to_test, low_stimulation_freq, high_stimulation_freq, threshold):
    '''
    Function that loops through all start and end times the user wishes to investigate 
    and calculates the accuracies and ITRs for each acceptable time window

    Parameters
    ----------
    data_dict : dict of 6 fields
        Dictionary containing 6 fields, each holding data for an aspect of the eeg experiment.
    electrode_of_interest : String
        String representing the name of the channel/electrode we are calculating the accuracies and itrs for.
    start_times_to_test : Array of size (times)
        Array input by the user which hold each start time that should be tested.
    end_times_to_test : Array of size (times)
        Array input by the user which hold each end time that should be tested.
    low_stimulation_freq : int32
        integer taken as user input representing the lower stimulation frequency we are investigating.
    high_stimulation_freq : int32
        integer taken as user input representing the higher stimulation frequency we are investigating..
    threshold : float
        float representing the threshold (the difference between the higher and lower frequency), we will consider to be a correct prediction or not.


    Returns
    -------
    accuracies : Array of size (start times tested, end times tested)
        Array holding the accuracies at each different start and end time combination.
    itr_array : Array of size (start times tested, end times tested)
        Array holding the ITRs at each different start and end time combination.

    '''
    # reverse start times array in order to correctly populate matrix [s20e1 s20e2 ... s20e20
    #                                                                  ...
    #                                                                  s1e1    ...     s1e20]
    start_times_to_test = start_times_to_test[::-1]
    # initialize arrays to hold accuracies and itrs for every time window
    accuracies = np.array([]).astype('float')
    itr_array = np.array([]).astype('float')
    # loop through all possible start and end time combinations
    for start_time in start_times_to_test:
        for end_time in end_times_to_test:
            # check if start, end time comboination is acceptable
            if end_time > start_time:
                # epoch data, calc, ffts, calculate accuracy and itr for given start and end time combo
                eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data_dict, epoch_start_time=start_time, epoch_end_time=end_time)
                fs = data_dict['fs']
                eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)
                labels_pred = get_predicted_labels(data_dict, eeg_epochs_fft, fft_frequencies, low_stimulation_freq, high_stimulation_freq, electrode_of_interest, threshold)
                truth_labels = get_truth_labels(data_dict)
                accuracy = calculate_accuracy(labels_pred, truth_labels)
                itr = calculate_itr(accuracy, duration = end_time-start_time, truth_labels=truth_labels)
                
        
            else:
                # if unacceptable combo, insert nan value
                accuracy = np.nan
                itr = np.nan
            
            accuracies = np.append(accuracies, accuracy)
            itr_array = np.append(itr_array, itr)
    # reshape arrays into 2-D matrices of size start times tested by end times tested
    accuracies = (np.reshape(accuracies, (len(start_times_to_test), len(end_times_to_test))))*100
    itr_array = np.reshape(itr_array, (len(start_times_to_test), len(end_times_to_test)))
    
    return accuracies, itr_array





def plot_results(accuracies, itr_array, electrode_of_interest, start_times_tested, end_times_tested, subject):
    '''
    Function that plots the accuracy and itr matrices as psuedocolor plots to
    visualize the two figures of merit at each start and end time combination

    Parameters
    ----------
    accuracies : Array of size (start times tested, end times tested)
        Array holding the accuracies at each different start and end time combination.
    itr_array : Array of size (start times tested, end times tested)
        Array holding the ITRs at each different start and end time combination.
    electrode_of_interest : String
        String representing the name of the channel/electrode we are calculating the accuracies and itrs for.
    start_times_tested : Array of size (times)
        Array input by the user which hold each start time that were tested.
    end_times_tested : Array of size (times)
        Array input by the user which hold each end time that were tested.

    Returns
    -------
    None.

    '''
    plt.suptitle(f'Accuracy and ITR at Various Start and End Times on Channel {electrode_of_interest} for Subject {subject}')
    plt.rcParams["figure.figsize"] = (14,8)  
    plt.subplot(1, 2, 1)
    # plt accuracy matrix - nan values shown as white to signify these start and end times are unacceptable 
    plt.imshow(accuracies, extent = (end_times_tested[0], end_times_tested[-1], start_times_tested[0], start_times_tested[-1]))
    plt.colorbar(label = 'Accuracy (% Correct)', fraction=0.046, pad=0.04)
    plt.xticks(np.arange(0,end_times_tested[-1]+1,2.5))
    plt.yticks(np.arange(0,start_times_tested[-1]+1,2.5))
    plt.xlabel('Epoch End Times (s)')
    plt.ylabel('Epoch Start Times (s)')
    plt.title('Accuracy for Various Epoch Time Windows')
    
    plt.subplot(1, 2, 2)
    # plt itr matrix - nan values shown as white to signify these start and end times are unacceptable 
    plt.imshow(itr_array, extent = (end_times_tested[0], end_times_tested[-1], start_times_tested[0], start_times_tested[-1]))
    plt.colorbar(label = 'ITR', fraction=0.046, pad=0.04)
    plt.xticks(np.arange(0,end_times_tested[-1]+1,2.5))
    plt.yticks(np.arange(0,start_times_tested[-1]+1,2.5))
    plt.xlabel('Epoch End Times (s)')
    plt.ylabel('Epoch Start Times (s)')
    plt.title('ITR for Various Epoch Time Windows')
    
    
    plt.tight_layout()
    












