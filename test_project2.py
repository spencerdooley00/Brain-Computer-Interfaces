# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:01:30 2021

@author: spencer
"""
import Project2
import numpy as np
import import_ssvep_data

data_directory="C:/Users/spenc/Documents/UVM/Fall 2021/BME 296/BME296Git/SsvepData/"
subject=1


## Calling functions from previous labs
data_dict = import_ssvep_data.load_ssvep_data(subject, data_directory)
eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20)
fs = data_dict['fs']
event_duration = data_dict['event_durations'][0]/fs
eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)



##### USER INPUT #####
# set your low and high stimulation frequency 
low_stimulation_freq = 12
high_stimulation_freq = 15
# set threshold (diff between )
threshold = 0.0
# set the increment of time for time arrays
time_step = 1
# create time arrays
start_times_to_test = np.arange(0, event_duration+.0001, time_step)
end_times_to_test = np.arange(0,event_duration+.0001, time_step)
# set what electrode/channel is being investigated
electrode_of_interest='Oz'



# call function to get predicted labels
labels_pred = Project2.get_predicted_labels(data_dict, eeg_epochs_fft, fft_frequencies, low_stimulation_freq, high_stimulation_freq, electrode_of_interest, threshold)

# call function to get truth labels
truth_labels = Project2.get_truth_labels(data_dict)

# test call function to calculate accuracy for default time window
accuracy = Project2.calculate_accuracy(labels_pred, truth_labels)

# call function to get accuracies and its matrices for all differnt time windows
accuracies, itr_array = Project2.test_epoch_limits(data_dict, electrode_of_interest, start_times_to_test, end_times_to_test, low_stimulation_freq, high_stimulation_freq, threshold)

# plot and visualize the results
Project2.plot_results(accuracies, itr_array, electrode_of_interest, start_times_to_test, end_times_to_test, subject)

















