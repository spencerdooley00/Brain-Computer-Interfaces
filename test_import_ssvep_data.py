# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:22:12 2021

@author: spenc
"""
import import_ssvep_data
data_directory="C:/Users/spenc/Documents/UVM/Fall 2021/BME 296/BME296Git/SsvepData/"
subject=2
data_dict = import_ssvep_data.load_ssvep_data(subject, data_directory)

channels_to_plot = ['Fz','Oz']
channels = data_dict['channels']

import_ssvep_data.plot_raw_data(data_dict,subject, channels_to_plot)


eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20)

fs = data_dict['fs']
eeg_epochs_fft, fft_frequencies = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)


import_ssvep_data.plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels_to_plot, channels)








