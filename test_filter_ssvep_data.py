# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:18:47 2021

@author: Spencer, Kai
"""
import import_ssvep_data
import filter_ssvep_data


subject=1
data_directory= 'C:/Users/spenc/Documents/UVM/Fall 2021/BME 296/BME296Git/SsvepData/'
data_dict = import_ssvep_data.load_ssvep_data(subject, data_directory)
fs = data_dict['fs']



# %%
filter_coefficients_12 = filter_ssvep_data.make_bandpass_filter(fs = fs, low_cutoff = 11, high_cutoff = 13, filter_type = "hann", filter_order = 1001)

filter_coefficients_15 = filter_ssvep_data.make_bandpass_filter(fs = fs, low_cutoff = 14, high_cutoff = 16, filter_type = "hann", filter_order = 1001)
# increasing the order of the filter increases the occilations of the gain of the filter
# impulse response. Decreasing the order of the filter reduces the osccilations. Increaseing 
# order of the filter created a steeper cutoff of the frequency respoce. A higher oder filter
# has a faster attenuation than a lower order filter. 
# %%

filtered_data_12 = filter_ssvep_data.filter_data(data_dict, filter_coefficients_12)
filtered_data_15 = filter_ssvep_data.filter_data(data_dict, filter_coefficients_15)


envelope_12 = filter_ssvep_data.get_envelope(data_dict, filtered_data_12, channel_to_plot = 'Oz', frequency_to_plot = '12')
envelope_15 = filter_ssvep_data.get_envelope(data_dict, filtered_data_15,  channel_to_plot = 'Oz', frequency_to_plot = '15')


filter_ssvep_data.plot_ssvep_amplitudes(data_dict, envelope_12, envelope_15, subject = subject, channel_to_plot = 'Oz', first_frequency = '12Hz', second_frequency = '15Hz')

# If we zoom in on the graph to the section where the impulse frequency changes we see
# that the amplitude of the 12Hz signal is higher during the 12Hz impulses. At the 
# 15 Hz impulse the amplidude of the 15Hz signal becomes higher. For the Oz channel
# the 12Hz signal is more consistantly higher than for the 15Hz signal. The response is similar 
# for the Fz channel the amplitude is consistantly higher for the 12Hz impulses than the 15Hz 
# is. 






